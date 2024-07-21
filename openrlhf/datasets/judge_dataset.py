from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, process_multi_turn_dialogue, zero_pad_sequences


def preprocess_data(
    data, prompt_key=None, chosen_key=None, rejected_key=None, refined_critic_key=None, apply_chat_template=None
) -> str:
    prompt = data[prompt_key]
    chosen = data[chosen_key]
    rejected = data[rejected_key]
    refined_critic = data[refined_critic_key]
    return prompt, chosen, rejected, refined_critic


class JudgeDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo

        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.refined_critic = []
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_dpo = is_dpo

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)
        refined_critic_key = getattr(self.strategy.args, "refined_critic_key", None)

        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, refined_critic = preprocess_data(
                data, prompt_key, chosen_key, rejected_key, refined_critic_key, apply_chat_template
            )

            # prompt_ids_len for prompt mask
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(margin)

            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
