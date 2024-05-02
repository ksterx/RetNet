import os
import random
import time
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import interleave_datasets, load_dataset
from retnet.configuration_retnet import load_config_from_json
from retnet.modeling_retnet_test import RetNetForCausalLM
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer

rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ["SLURM_LOCALID"])
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]

print(
    f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}, Master Addr: {master_addr}, Master Port: {master_port}",
    flush=True,
)


# 分散学習のコード...
print("Is GPU available? =", torch.cuda.is_available())
print("How many GPUs available? =", torch.cuda.device_count())

device_string = PartialState().process_index
print("device_string :", device_string)


# Hard copy from https://github.com/huggingface/trl/blob/v0.8.1/trl/trainer/utils.py
class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question} ### Answer: {answer}"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
            append_concat_token ('bool', *optional*, defaults to True)
                If true, appends `eos_token_id` at the end of each sample being packed.
            add_special_tokens ('bool', *optional*, defaults to True)
                If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start."
                        )
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(
                buffer, add_special_tokens=self.add_special_tokens, truncation=False
            )["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        try:
            super()._save_checkpoint(model, trial, metrics)
        except Exception as e:
            print(e)
            print("Continue without saving checkpoint")


@dataclass
class MyArgs:
    model_size: str = "3b"
    dataset_name: str = "v1"
    text_col: str = "text"
    max_seq_length: int = 2048
    packing: bool = False


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()
    print("train_args :\n", train_args)
    print("args :\n", args)

    # Check resume

    if train_args.gradient_checkpointing_kwargs is None:
        train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if args.dataset_name == "v1":
        # Japanese
        ## Wikipedia
        # print("izumi-lab/wikipedia-ja-20230720")
        print("Spiral-AI/wikipedia")
        data_ja_wiki = load_dataset(
            "Spiral-AI/wikipedia", "hojichar", split="train", streaming=True
        )
        prob_ja_wiki = 4.72

        ## CC100
        print("izumi-lab/cc100-ja-filter-ja-normal")
        print("Spiral-AI/cc100")
        data_ja_cc100 = load_dataset(
            "Spiral-AI/cc100", "hojichar", split="train", streaming=True
        )
        prob_ja_cc100 = 10.0

        ## CalturaX (Oscar+mc4)
        wait_time = random.randint(0, 60)
        time.sleep(wait_time)
        print(f"uonlp/CulturaX (wait_time: {wait_time})")
        data_ja_calturaX = load_dataset(
            "Spiral-AI/CulturaX", "hojichar", split="train", token=True, streaming=True
        )
        prob_ja_calturaX = 93.0

        # English
        ## Wikipedia
        print("wikipedia.20220301.en")
        data_en_wiki = load_dataset(
            "wikipedia", "20220301.en", split="train", streaming=True
        )
        prob_en_wiki = 20.0

        ## CulturaX (Oscar+mc4)
        wait_time = random.randint(0, 60)
        time.sleep(wait_time)
        print(f"uonlp/CulturaX (wait_time: {wait_time})")
        data_en_calturaX = load_dataset(
            "uonlp/CulturaX", "en", split="train", token=True, streaming=True
        )
        prob_en_calturaX = 2846.0 * 0.01  # 28bn. tokens相当に調整

        print("...done")

        datasets = [
            data_ja_wiki,
            data_ja_cc100,
            data_ja_calturaX,
            data_en_wiki,
            data_en_calturaX,
        ]
        probs = [
            prob_ja_wiki,
            prob_ja_cc100,
            prob_ja_calturaX,
            prob_en_wiki,
            prob_en_calturaX,
        ]

        for d in datasets:
            d = d.shuffle()

        total_probs = sum(probs)
        probs = [x / total_probs for x in probs]

        train_dataset = interleave_datasets(
            datasets,
            probabilities=probs,
            ### seed=42, ## don't set seed number, which potentially fetch the same datasets on different nodes.
        )

    else:
        train_dataset = load_dataset(args.dataset_name, split="train", streaming=True)
        # eval_dataset = load_dataset(args.dataset_name, split="validation")

    # Prepare models
    config = load_config_from_json(f"configs/retnet-{args.model_size}/config.json")
    model = RetNetForCausalLM(config)

    tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
    # tokenizer.model_max_length = 16384
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    if args.packing:
        train_dataset = ConstantLengthDataset(
            tokenizer=tokenizer,
            dataset=train_dataset,
            dataset_text_field="text",
            seq_length=args.max_seq_length,
            shuffle=True,
            infinite=True,
        )

    trainer = MySFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # packing=args.packing,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    if train_args.do_train:
        trainer.train(resume_from_checkpoint=True)
        trainer.save_model()
    if train_args.do_eval:
        trainer.evaluate()

    # dist.destroy_process_group()


if __name__ == "__main__":
    main()

data_ja_calturaX = load_dataset(
    "json",
    # "Spiral-AI/CulturaX",
    # "hojichar",
    data_files="/nas/share/datasets/Spiral-AI+CulturaX/main/hojichar",
    split="train",
    # token=True,
    streaming=True,
)
