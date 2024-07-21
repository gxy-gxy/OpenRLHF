import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss, SwitchBalancingLoss, ValueLoss

import numpy as np
import logging
import sys

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
    stream=sys.stdout  # 设置输出流为标准输出
)

def pearson_correlation(x, y):
    """
    计算两个序列的皮尔逊相关系数
    
    参数:
    x (list or np.array): 第一个序列
    y (list or np.array): 第二个序列
    
    返回:
    float: 皮尔逊相关系数

    # 示例数据
    golden_scores = [4, 3, 5, 2, 1, 3, 4, 5, 2, 1]
    model_predictions = [3.5, 3, 4.5, 2, 1.5, 3, 4, 5, 2, 1]

    # 计算皮尔逊相关系数
    r = pearson_correlation(golden_scores, model_predictions)
    print(f"Pearson Correlation Coefficient: {r}")
    """
    # 将输入转换为numpy数组
    x = np.array(x)
    y = np.array(y)
    
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 计算分子和分母
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    # 计算皮尔逊相关系数
    r = numerator / denominator
    
    return r


class OfflineCriticModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        # if loss == "sigmoid":
        #     self.loss_fn = PairWiseLoss()
        #     self.strategy.print("LogSigmoid Loss")
        # else:
        #     self.loss_fn = LogExpLoss()
        #     self.strategy.print("LogExp Loss")
        
        self.loss_fn = ValueLoss()
        self.strategy.print("Value Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    # TODO 看一下这里是否要torch.no_grad
    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ):
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0(), total=self.epochs)
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
                total=len(self.train_dataloader)
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            
            # 除了获取采样到的response，还要获取每个response对应的reward
            # 这里应该以prompt为单位累计
            for prompts_id_len, inputs_batch, attention_masks_batch, rewards_batch, _ in self.train_dataloader:
                losses = []
                for i in range(len(inputs_batch)):
                    input_id_len = prompts_id_len[i]
                    loss_item = 0
                    for inputs, attention_masks, rewards in zip(inputs_batch[i], attention_masks_batch[i], rewards_batch[i]):
                        inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                        attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                        # ! 必须使用右pad action_mask才是正确的？
                        action_mask = attention_mask[:, input_id_len - 1 : -1]

                        values = self.model(
                            inputs,
                            action_mask=action_mask,
                            attention_mask=attention_mask,
                        )

                        rewards = torch.tensor(rewards, device=torch.cuda.current_device(), dtype=values.dtype)
                        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
                        # tensor([[312], [319]], device='cuda:0')
                        rewards = torch.zeros_like(values).scatter_(dim=1, index=eos_indices, src=rewards.unsqueeze(1).to(values.dtype))

                        advantages, returns = self.get_advantages_and_returns(
                            values,
                            rewards,
                            action_mask,
                            self.strategy.args.gamma,
                            self.strategy.args.lambd,
                        )
                        loss = self.loss_fn(values, values, returns, action_mask)/len(rewards_batch[i])
                        loss_item += loss.item()
                        self.strategy.backward(loss, self.model, self.optimizer)

                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                    losses.append(loss_item)
                # optional rm info
                logs_dict = {
                    "critic_loss": sum(losses)/len(losses),
                }
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            # step_bar.set_postfix(logs_dict)
            logging.info(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
            total=len(eval_dataloader)
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards_all = []
            values_all = []
            pearson_corr_sum = 0
            loss_sum = 0
            for prompts_id_len, inputs_batch, attention_masks_batch, rewards_batch, _ in self.eval_dataloader:
                for i in range(len(inputs_batch)):
                    input_id_len = prompts_id_len[i]
                    loss_item = 0
                    prompt_wise_rewards = []
                    prompt_wise_values = []
                    for inputs, attention_masks, rewards in zip(inputs_batch[i], attention_masks_batch[i], rewards_batch[i]):
                        inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                        attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                        # ! 必须使用右pad action_mask才是正确的？
                        action_mask = attention_mask[:, input_id_len - 1 : -1]

                        values = self.model(
                            inputs,
                            action_mask=action_mask,
                            attention_mask=attention_mask,
                        )
                        rewards_raw = torch.tensor(rewards, device=torch.cuda.current_device(), dtype=values.dtype)
                        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
                        # tensor([[312], [319]], device='cuda:0')

                        extracted_values = torch.gather(values, 1, eos_indices).view(-1)

                        rewards = torch.zeros_like(values).scatter_(dim=1, index=eos_indices, src=rewards_raw.unsqueeze(1).to(values.dtype))

                        advantages, returns = self.get_advantages_and_returns(
                            values,
                            rewards,
                            action_mask,
                            self.strategy.args.gamma,
                            self.strategy.args.lambd,
                        )

                        loss = self.loss_fn(values, values, returns, action_mask)/len(rewards_batch[i])
                        loss_item += loss.item()

                        prompt_wise_rewards.extend(rewards_raw.tolist())
                        prompt_wise_values.extend(extracted_values.tolist())
                        rewards_all.append(rewards_raw)
                        values_all.append(extracted_values)

                    pearson_corr = pearson_correlation(prompt_wise_rewards, prompt_wise_values)
                    pearson_corr_sum += pearson_corr/len(rewards_batch[i])

                loss_sum += loss_item
                step_bar.update()

            # acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()
            pearson_corr_mean = pearson_corr_sum / self.eval_dataloader.__len__()

            rewards = torch.cat(rewards_all).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            values = torch.cat(values_all).float()
            values = self.strategy.all_gather(values)
            value_mean = torch.mean(values)
            value_std = torch.std(values).clamp(min=1e-8)

            # save mean std
            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "pearson_corr_mean": pearson_corr_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
                "value_mean": value_mean.item(),
                "value_std": value_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            logging.info(logs)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state

