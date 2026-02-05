import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoProcessor
import logging

from omegaconf import OmegaConf
from typing import List, Optional
from dataclasses import dataclass

from trl import SFTConfig
from qwen_vl_utils import process_vision_info

from utils import seed_everything, place_input_image, generate_labels
from task import task_preprocess_functions
from data import load_train_dataset
from trainer import CustomSFTTrainer

import wandb


@dataclass
class Config():
    run_name: str = "planning"
    report_to: str = "wandb"

    model: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    
    task: str = 'sft_text'
    data_path: str = './data/train/train_desc.jsonl'
    
    image_size: Optional[int] = None
    
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2

    save_model_path: Optional[str] = None
    load_model_path: Optional[str] = None
    epochs: int = 10

    seed: int = 42


def setup() -> Config:
    default_config: Config = OmegaConf.structured(Config)
    cli_args = OmegaConf.from_cli()
    config: Config = OmegaConf.merge(default_config, cli_args)
    return config


def create_text_collate_fn(processor: AutoProcessor):

    def collate_function(examples):
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
        batch = processor(text=texts, return_tensors="pt", padding=True)
        
        answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
        pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        
        batch["labels"] = generate_labels(batch["input_ids"], answer_start_token_pattern, pad_token_idx)

        return batch
    
    return collate_function


def create_image_collate_fn(processor: AutoProcessor):

    def collate_function(examples):
        texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
        texts = [place_input_image(text) for text in texts]
        image_inputs, _ = process_vision_info(examples)
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
        pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        
        batch["labels"] = generate_labels(batch["input_ids"], answer_start_token_pattern, pad_token_idx)
        return batch
    
    return collate_function


if __name__ == "__main__":
    config = setup()
    seed_everything(seed=config.seed)

    wandb.init(project="ood_reasoning", name=config.run_name)
    
    log_file = f"./log_{config.run_name}.txt"
    
    if not config.save_model_path:
        config.save_model_path = f"./checkpoints/{config.run_name}"
        os.makedirs(config.save_model_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )

    logging.info('=='*20)
    logging.info(config)
    logging.info('=='*20)

    processor = AutoProcessor.from_pretrained(config.model)
    processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
    processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)
    assert processor.tokenizer.padding_side == "right"


    model_path = config.model
    model_config = Qwen2_5_VLConfig.from_pretrained(model_path) 


    if config.load_model_path is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, config=model_config, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.load_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(processor.tokenizer))

    for param in model.visual.parameters():
        param.requires_grad = False

    preprocess_function = task_preprocess_functions[config.task]

    epochs = config.epochs
    
    train_dataset = load_train_dataset(config.data_path)
    train_dataset = [preprocess_function(sample, image_size=config.image_size) for sample in train_dataset]

    if config.task == "sft_image":
        collate_fn = create_image_collate_fn(processor)
    elif config.task == "sft_text":
        collate_fn = create_text_collate_fn(processor)
    
    training_args = SFTConfig(
        output_dir=config.save_model_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        logging_dir='./logs/',
        logging_strategy='steps',
        report_to=["wandb"],
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