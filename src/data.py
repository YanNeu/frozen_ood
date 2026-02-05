import json
import numpy as np
from datasets import Dataset


def load_train_dataset(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)
