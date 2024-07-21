from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, process_multi_turn_dialogue, zero_pad_sequences


def preprocess_data(data, input_template=None, input_key=None, output_key=None, apply_chat_template=None):
    # custom dataset
    if input_key:
        raw_prompt = data[input_key]
        response = data[output_key]

        if apply_chat_template:
            prompt = apply_chat_template([
                {"role": "human", "content": raw_prompt}
            ], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template([
                {"role": "human", "content": raw_prompt},
                {"role": "assistant", "content": response}
            ], tokenize=False)[len(prompt) :]
            input_template = None
    else:
        # Open-Orca/OpenOrca
        if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + " " + data["question"]
            response = data["response"]
        # MaziyarPanahi/WizardLM_evol_instruct_V2_196k
        # jondurbin/airoboros-3.2
        elif exist_and_not_none(data, "conversations") and isinstance(data["conversations"], list):
            prompt = process_multi_turn_dialogue(
                data["conversations"][:-1], input_template=input_template, content_key="value", role_key="from"
            )
            response = data["conversations"][-1]["value"]
            input_template = None  # do not modified with input template again
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            response = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, response


class ValueDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
        response_micro_batch_size=1
    ) -> None:
        super().__init__()
        self.prompts = []
        self.responses = []
        self.prompt_ids_lens = []
        self.rewards = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.response_micro_batch_size = response_micro_batch_size
        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            raw_prompt = data["prompt"]
            responses_batch = []
            rewards_batch = data["rm_score"]
            for response in data["response"]:
                prompt, response = preprocess_data(
                    {input_key: raw_prompt, output_key: response},
                    None if pretrain_mode else input_template,
                    input_key,
                    output_key,
                    apply_chat_template=apply_chat_template,
                )
                if not self.pretrain_mode:
                    prompt_token = self.tokenizer(
                        prompt,
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    )
                    prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                else:
                    prompt_ids_len = 0

                if not self.pretrain_mode:
                    # filter the sample whose length is greater than max_length (2 for answer length)
                    if prompt_ids_len >= self.max_length - 2:
                        continue
                    if not prompt or not response:
                        continue
                responses_batch.append(response)
            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.responses.append(responses_batch)
            self.rewards.append(rewards_batch)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        # input_ids_batch
        # response_ids_batch
        # return input_batch, reward_batch
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response_batch = self.responses[idx]
        reward_batch = self.rewards[idx]

        text_batch = [(prompt + response).rstrip("\n") for response in response_batch]
        for i in range(len(text_batch)):
            if not text_batch[i].endswith(self.tokenizer.eos_token):
                text_batch[i] += " " + self.tokenizer.eos_token

        input_token_batch =[self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ) for text in text_batch]
        info = {"input": prompt, "output": response_batch}
        # to avoid EOS_token truncation
        for input_token in input_token_batch:
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True

        return prompt_ids_len, [input_token["input_ids"] for input_token in input_token_batch], [input_token["attention_mask"] for input_token in input_token_batch], reward_batch, info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids_all = []
        attention_masks_all = []
        rewards_all = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id_list, attention_mask_list, rewards_list, info in item_list:
            input_ids = []
            attention_masks = []
            rewards = []
            for input_start_idx in range(0, len(input_id_list), self.response_micro_batch_size):
                input_batch = input_id_list[input_start_idx:input_start_idx+self.response_micro_batch_size]
                attention_mask_batch = attention_mask_list[input_start_idx:input_start_idx+self.response_micro_batch_size]
                rewards_batch = rewards_list[input_start_idx:input_start_idx+self.response_micro_batch_size]
                input_batch = zero_pad_sequences(input_batch, "right", self.tokenizer.pad_token_id)
                attention_mask_batch = zero_pad_sequences(attention_mask_batch, "right")
                
                input_ids.append(input_batch)
                attention_masks.append(attention_mask_batch)
                rewards.append(rewards_batch)

            prompt_ids_lens.append(prompt_ids_len)
            input_ids_all.append(input_ids)
            attention_masks_all.append(attention_masks)
            rewards_all.append(rewards)

            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        # input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        # attention_masks = zero_pad_sequences(attention_masks, "right")
        # 这边返回按prompt 再按response的micro batch padding的结果就行
        return prompt_ids_lens, input_ids_all, attention_masks_all, rewards_all, infos
