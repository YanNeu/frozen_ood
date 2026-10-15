from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from mathruler.grader import extract_boxed_content
from transformers import AutoModelForCausalLM, AutoTokenizer

from task import simulate_vsp
from utils import seed_everything


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


@dataclass
class Config:
    model: str
    load_model_path: str
    data_path: str
    save_dir: str
    log_file: str
    seed: int = 42
    max_new_tokens: int = 4096
    do_sample: bool = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    print_answers: bool = False
    attn_implementation: Optional[str] = None


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate a text-only causal LM on FrozenLake text prompts.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--load_model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--do_sample", type=parse_bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--print_answers", type=parse_bool, default=False)
    parser.add_argument("--attn_implementation", default=None)
    return Config(**vars(parser.parse_args()))


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def result_save_path(config: Config) -> str:
    run_name = os.path.basename(config.load_model_path.rstrip("/\\"))
    run_name += f"_{os.path.basename(config.data_path).replace('.jsonl', '')}"
    if not config.do_sample:
        run_name += "_greedy"
    return os.path.join(config.save_dir, f"results_{run_name}.json")


def generation_kwargs(config: Config, tokenizer: AutoTokenizer) -> dict:
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if config.do_sample:
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
    return kwargs


def build_prompt(tokenizer: AutoTokenizer, sample: dict) -> str:
    messages = [{"role": "user", "content": sample["text_input"]}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_eval(config: Config, tokenizer: AutoTokenizer, model) -> dict:
    os.makedirs(config.save_dir, exist_ok=True)
    data = load_jsonl(config.data_path)

    correct = 0
    invalid = 0
    correct_dict: dict[int, int] = {}
    invalid_dict: dict[int, int] = {}
    num_samples: dict[int, int] = {}
    correct_indices: dict[int, list[str]] = {}
    answers: dict[int, dict[str, str]] = {}
    gen_kwargs = generation_kwargs(config, tokenizer)

    for i, sample in enumerate(data):
        map_size = len(sample["map_desc"])
        map_id = str(sample["map_id"])
        correct_dict.setdefault(map_size, 0)
        invalid_dict.setdefault(map_size, 0)
        num_samples.setdefault(map_size, 0)
        correct_indices.setdefault(map_size, [])
        answers.setdefault(map_size, {})

        text = build_prompt(tokenizer, sample)
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][prompt_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        path_str = extract_boxed_content(answer)
        result = simulate_vsp(sample.get("map_desc", []), path_str)

        if result["success"]:
            correct += 1
            correct_dict[map_size] += 1
            correct_indices[map_size].append(map_id)
        elif result["invalid"]:
            invalid += 1
            invalid_dict[map_size] += 1

        answers[map_size][map_id] = answer
        num_samples[map_size] += 1

        if config.print_answers:
            print(f"success: {result['success']}")
            print(answer)
            print()
        if (i + 1) % 20 == 0:
            logging.info(
                "[%s] Accuracy: %s/%s (%.3f), Invalid: %s/%s (%.3f)",
                i + 1,
                correct,
                i + 1,
                correct / (i + 1),
                invalid,
                i + 1,
                invalid / (i + 1),
            )

    num_total = sum(num_samples.values())
    logging.info(
        "[Final] model=%s Accuracy: %s/%s (%.3f), Invalid: %s/%s (%.3f)",
        config.load_model_path,
        correct,
        num_total,
        correct / num_total if num_total else 0.0,
        invalid,
        num_total,
        invalid / num_total if num_total else 0.0,
    )

    results_dict = {
        "run": config.load_model_path,
        "data_path": config.data_path,
        "demo_path": None,
        "num_shots": 0,
        "demo_seed": 0,
        "do_sample": config.do_sample,
        "temperature": config.temperature if config.do_sample else None,
        "top_p": config.top_p if config.do_sample else None,
        "top_k": config.top_k if config.do_sample else None,
        "total_samples": num_total,
        "total_correct": correct,
        "total_invalid": invalid,
        "total_acc": correct / num_total if num_total else 0.0,
        "correct_indices": correct_indices,
        "answers": answers,
    }
    for map_size in correct_dict:
        results_dict[f"{map_size}_correct"] = correct_dict[map_size]
        results_dict[f"{map_size}_invalid"] = invalid_dict[map_size]
        results_dict[f"{map_size}_acc"] = correct_dict[map_size] / num_samples[map_size]

    with open(result_save_path(config), "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    return results_dict


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    log_dir = os.path.dirname(config.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(config.log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("========================================")
    logging.info(config)
    logging.info("========================================")

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "pretrained_model_name_or_path": config.load_model_path,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    logging.info("Requested attention implementation: %s", config.attn_implementation or "<auto>")
    logging.info("Loaded attention implementation: %s", getattr(model.config, "_attn_implementation", None))
    logging.info("Loaded model class: %s", model.__class__.__name__)
    text_config = getattr(model.config, "text_config", None)
    layer_types = getattr(text_config, "layer_types", None) or getattr(model.config, "layer_types", None)
    if layer_types:
        logging.info("Layer types: %s", layer_types)
    model.eval()
    run_eval(config, tokenizer, model)


if __name__ == "__main__":
    main()
