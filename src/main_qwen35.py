import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig

from data import load_train_dataset
from task import task_preprocess_functions
from trainer import CustomSFTTrainer
from utils import generate_labels, seed_everything


@dataclass
class Config:
    run_name: str = "planning_qwen35"
    report_to: str = "wandb"

    model: str = "Qwen/Qwen3.5-2B"
    task: str = "sft_text"
    data_path: str = "./data/train/train_coord_reasoning_coord_desc.jsonl"

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2

    save_model_path: Optional[str] = None
    load_model_path: Optional[str] = None
    epochs: int = 10
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: Optional[int] = 1

    seed: int = 42


def setup() -> Config:
    default_config: Config = OmegaConf.structured(Config)
    cli_args = OmegaConf.from_cli()
    return OmegaConf.merge(default_config, cli_args)


def freeze_vision_tower(model) -> None:
    vision_module = getattr(model, "visual", None)
    if vision_module is None:
        vision_module = getattr(model, "vision_tower", None)
    if vision_module is None:
        logging.warning("No vision module named 'visual' or 'vision_tower' was found.")
        return
    for parameter in vision_module.parameters():
        parameter.requires_grad = False


def create_text_collate_fn(processor: AutoProcessor):
    def collate_function(examples):
        texts = [
            processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]
        batch = processor(text=texts, return_tensors="pt", padding=True)

        answer_start_token_pattern = processor.tokenizer(
            "<|im_start|>assistant", return_tensors="pt"
        )["input_ids"][0]
        pad_token_idx = processor.tokenizer(
            "<|endoftext|>", return_tensors="pt"
        )["input_ids"][0]
        batch["labels"] = generate_labels(
            batch["input_ids"], answer_start_token_pattern, pad_token_idx
        )
        return batch

    return collate_function


if __name__ == "__main__":
    config = setup()
    seed_everything(seed=config.seed)

    if config.report_to == "wandb":
        wandb.init(project="ood_reasoning", name=config.run_name)

    log_file = f"./log_{config.run_name}.txt"
    if not config.save_model_path:
        config.save_model_path = f"./checkpoints/{config.run_name}"
    os.makedirs(config.save_model_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("========================================")
    logging.info(config)
    logging.info("========================================")

    processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
    assert processor.tokenizer.padding_side == "right"

    model_path = config.load_model_path or config.model
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    freeze_vision_tower(model)

    preprocess_function = task_preprocess_functions[config.task]
    train_dataset = load_train_dataset(config.data_path)
    train_dataset = [preprocess_function(sample) for sample in train_dataset]

    if config.task != "sft_text":
        raise ValueError("The Qwen3.5 entry point supports text-only supervision.")
    collate_fn = create_text_collate_fn(processor)

    report_to = ["wandb"] if config.report_to == "wandb" else []
    training_args = SFTConfig(
        output_dir=config.save_model_path,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        logging_dir="./logs/",
        logging_strategy="steps",
        report_to=report_to,
        run_name=config.run_name,
    )

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
