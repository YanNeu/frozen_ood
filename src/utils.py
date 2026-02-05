import torch
import numpy as np
import random


def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def place_input_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", image_placeholder="<image>", sep_token="<|im_start|>assistant") -> str:

    if sep_token is not None:
        assert sep_token in text

        t1, t2 = text.split(sep_token)

        if image_placeholder in t1:
            t1 = t1.replace(image_pad, '')
            t1 = t1.replace(image_placeholder, image_pad)

        return t1 + sep_token + t2
    else:
        return text.replace(image_pad, '').replace(image_placeholder, image_pad)


def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:
    seq_len = row.size(0)
    pat_len = pattern.size(0)

    for start_idx in range(seq_len - pat_len + 1):
        if torch.all(row[start_idx:start_idx + pat_len] == pattern):
            return start_idx
    return -1


def generate_labels(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
) -> torch.Tensor:
    batch_size, seq_length = input_ids.shape

    labels = input_ids.clone()

    for b in range(batch_size):
        row = labels[b]

        start_idx = find_subsequence(row, start_sequence)

        if start_idx == -1:
            # mask everything when sub-sequence was not found
            row[:] = -100
        else:
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len # position after subsequence

            # mask everything before and including the sub-sequence
            row[:end_of_subseq] = -100
        
        row[row == pad_token_idx] = -100

    return labels


