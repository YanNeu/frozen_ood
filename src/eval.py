import os
import json
import logging
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from omegaconf import OmegaConf
from typing import List, Optional
from dataclasses import dataclass

from qwen_vl_utils import process_vision_info
from mathruler.grader import extract_boxed_content

from utils import seed_everything, place_input_image
from task import task_test_preprocess_functions, simulate_vsp


def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

@dataclass
class Config():
    model: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    
    data_path: str = './data/test_id/test_level3_4_5_6_desc.jsonl'
    
    load_model_path: str = './checkpoints/sft_mirage_stage_mirage'
    
    seed: int = 42
    log_file: str = './test_log.txt'

    embed_suffix: Optional[str] = None

    save_dir: str = "results"

    task: str = "sft_text"
    max_new_tokens: int = 16192

def setup() -> Config:
    default_config: Config = OmegaConf.structured(Config)
    cli_args = OmegaConf.from_cli()
    config: Config = OmegaConf.merge(default_config, cli_args)
    return config


if __name__ == "__main__":

    config = setup()

    if "txt" in config.load_model_path or "table" in config.load_model_path or "grid" in config.load_model_path:
        config.task = "sft_text"
    seed_everything(seed=config.seed)

    os.makedirs(config.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(config.log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )

    run_name = config.load_model_path.split('/')[-1]
    run_name += f"_{config.data_path.split('/')[-1].replace('.jsonl', '')}"
    save_path = os.path.join(config.save_dir, f"results_{run_name}.json")

    logging.info('=='*20)
    logging.info(config)
    logging.info('=='*20)
    
    processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.load_model_path, device_map="auto", torch_dtype=torch.bfloat16)

    model.eval()

    with open(config.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    correct, invalid = 0, 0
    correct_dict, invalid_dict = {}, {}
    num_samples = {}
    correct_indices = {}
    answers = {}
    for i, sample in tqdm(enumerate(data)):
        
        map_size = len(sample["map_desc"])
        map_id = sample["map_id"]

        if not map_size in correct_dict: 
            correct_dict[map_size] = 0
            invalid_dict[map_size] = 0 
            num_samples[map_size] = 0
            correct_indices[map_size] = []
            answers[map_size] = {}

        preprocess_function = task_test_preprocess_functions[config.task]
        conversations = preprocess_function(sample)

        texts = [processor.apply_chat_template(conversations, tokenize=False)]
        texts = [place_input_image(text, sep_token=None) for text in texts]
        image_inputs, _ = process_vision_info(conversations)

        inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                tokenizer=processor.tokenizer,
            )
        
        decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        answer = decoded_output.split('<|im_start|>assistant')[-1]

        path_str = extract_boxed_content(answer)
        map_desc = sample.get("map_desc", [])
        result = simulate_vsp(map_desc, path_str)
        if result['success']: 
            correct += 1
            correct_dict[map_size] += 1
            correct_indices[map_size].append(map_id)
        elif result['invalid']: 
            invalid += 1
            invalid_dict[map_size] += 1
        print(f"success: {result['success']}")
        print(answer)
    
        answers[map_size][map_id] = answer
        num_samples[map_size] += 1
        print()

        if (i+1) % 20 == 0:
            logging.info(f"[{i+1}] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")
    logging.info(f"model: {config.load_model_path}")
    logging.info(f"[Final] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f}), Invalid: {invalid}/{i+1} ({invalid/(i+1):.3f})")

    num_total = sum(num_samples.values())
    results_dict = {
        "run": config.load_model_path,
        "total_correct": correct,
        "total_invalid": invalid, 
        "total_acc": correct/num_total,
        "correct_indices": correct_indices,
        "answers": answers
    }

    for map_size in correct_dict.keys():
        results_dict[f"{map_size}_correct"] = correct_dict[map_size]
        results_dict[f"{map_size}_invalid"] = invalid_dict[map_size]
        results_dict[f"{map_size}_acc"] = correct_dict[map_size]/num_samples[map_size]
        
    
    json.dump(results_dict, open(save_path, "w"), indent=4)

