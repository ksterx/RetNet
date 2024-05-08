import os
import sys
from dataclasses import dataclass

import torch
from accelerate import PartialState
from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

if os.getenv("SPLLM_LIBRARY_PATH"):
    SPLLM_LIBRARY_PATH = os.getenv("SPLLM_LIBRARY_PATH")
    sys.path.append(SPLLM_LIBRARY_PATH)
else:
    raise ValueError("SPLLM_LIBRARY_PATH is not set")

if os.getenv("SPCHAT_LIBRARY_PATH"):
    SPCHAT_LIBRARY_PATH = os.getenv("SPCHAT_LIBRARY_PATH")
    sys.path.append(SPCHAT_LIBRARY_PATH)
else:
    raise ValueError("SPCHAT_LIBRARY_PATH is not set")

if os.getenv("WANDB_PROJECT"):
    PROJECT_NAME = os.getenv("WANDB_PROJECT")
else:
    raise ValueError("WANDB_PROJECT is not set")
if os.getenv("RESULT_DIR"):
    RESULT_DIR = os.getenv("RESULT_DIR")
else:
    raise ValueError("RESULT_DIR is not set")
if os.getenv("WANDB_NAME"):
    WANDB_NAME = os.getenv("WANDB_NAME")
else:
    raise ValueError("WANDB_NAME is not set")


from spchat.templates import TEMPLATE_REGISTRY
from spchat.utils import do_nothing, to_mt_prompt
from spllm.model.callbacks import ModelTreeCallback, SlackCallback
from spllm.model.training.utils import get_save_steps, set_seed
from spllm.model.utils import load_model_from_tree

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


@dataclass
class MyArgs:
    max_seq_length: int = 2048
    base_model_repo: str = "cyberagent/open-calm-3b"
    base_model_revision: str = "main"
    model_repo: str = "Spiral-AI/open-calm-3b-chat"
    model_revision: str = "main"


def main():
    parser = HfArgumentParser((TrainingArguments, MyArgs))
    train_args, args = parser.parse_args_into_dataclasses()
    print("train_args :\n", train_args)
    print("args :\n", args)

    if train_args.gradient_checkpointing_kwargs is None:
        train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    print("Spiral-AI/anonymous-chat-mt (base)")
    data_base = (
        load_from_disk("/nas/share/datasets/Spiral-AI+anonymous-chat-mt/main/base-AB")
        .select_columns(["messages", "speakers", "num_messages", "min", "max"])
        .map(
            lambda x: to_mt_prompt(
                x,
                template="calm2",
                template_registry=TEMPLATE_REGISTRY,
                target_speaker="A",
                do_split=False,
            )
        )
        .select_columns(["prompt"])
        .shuffle(seed=42)["train"]
        .train_test_split(test_size=100, seed=42)
    )

    print("Spiral-AI/anonymous-chat-mt (scenario)")
    data_scenario = (
        load_from_disk(
            "/nas/share/datasets/Spiral-AI+anonymous-chat-mt/main/scenario-AB"
        )
        .select_columns(["messages", "speakers", "num_messages", "min", "max"])
        .map(
            lambda x: to_mt_prompt(
                x,
                template="calm2",
                template_registry=TEMPLATE_REGISTRY,
                target_speaker="A",
                do_split=False,
            )
        )
        .select_columns(["prompt"])
        .shuffle(seed=42)["train"]
        .train_test_split(test_size=100, seed=42)
    )

    train_base, eval_base = data_base["train"], data_base["test"]
    train_scenario, eval_scenario = data_scenario["train"], data_scenario["test"]

    train_dataset = concatenate_datasets([train_base, train_scenario])

    eval_dataset = {"base": eval_base, "scenario": eval_scenario}

    repo_id = "cyberagent/open-calm-3b"
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    # tokenizer.pad_token = "?】"  # 51994

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=[48405, 35390, 20784, 27],
        instruction_template=[3558, 2316, 27],
    )

    # save_steps = get_save_steps(
    #     num_epochs=train_args.num_train_epochs,
    #     num_gpus=torch.cuda.device_count(),
    #     train_dataset=train_dataset,
    #     batch_size=train_args.per_device_train_batch_size,
    #     gradient_accumulation_steps=train_args.gradient_accumulation_steps,
    #     num_checkpoints=50,
    # )
    # print(f"Saving interval: {save_steps}")
    # train_args.save_steps = save_steps
    # train_args.eval_steps = save_steps

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="prompt",
        formatting_func=do_nothing(),
        packing=False,
        max_seq_length=args.max_seq_length,
        data_collator=data_collator,
    )

    trainer.add_callback(
        ModelTreeCallback(
            trainer=trainer,
            base_model=args.base_model_repo,
            base_model_revision=args.base_model_revision,
            model_name=args.model_repo,
            model_revision=args.model_revision,
            description=os.getenv("WANDB_NOTES"),
        )
    )
    trainer.add_callback(SlackCallback(trainer, "dev"))

    if train_args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    main()
