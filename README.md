# **On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks**

**Yannic Neuhaus**<sup>1</sup>, **Chanoknan Tanchotikul**<sup>2</sup>, **Nicolas Flammarion**<sup>3</sup>, **Matthias Hein**<sup>1</sup>, **Francesco Croce**<sup>4</sup>

<sup>1</sup>*University of Tübingen*
<sup>2</sup>*Aalto University*
<sup>3</sup>*EPFL*
<sup>4</sup>*ELLIS Institute Finland, Aalto University*


[![arXiv](https://img.shields.io/badge/arXiv-2602.15460-b31b1b.svg)](https://arxiv.org/abs/2602.15460)

## Data

Download the complete datasets
[here](https://nc.mlcloud.uni-tuebingen.de/index.php/s/gFz9tc2wFryF9Wn) and
extract the archive in the repository root so that its contents are placed
under `./data`.

### Input representations

The datasets provide image, natural-language description, ASCII grid, ASCII
table, numeric-coordinate, and spreadsheet-coordinate representations. Their
JSONL file names use `image`, `desc`, `grid`, `table`, `coord`, and `sheet`,
respectively.

<p align="center">
  <img width="800" src="./assets/input_representations.png" alt="Examples of input representations">
</p>

### Training data

Training targets range from final answers and natural-language reasoning to
grid- and table-state traces, coordinate-state traces, traces combining
coordinates with natural-language descriptions, and A* search traces written
with numeric or spreadsheet-style coordinates. The corresponding file names
include `reasoning_desc`, `reasoning_grid`, `reasoning_table`,
`reasoning_coord`, `reasoning_coord_desc`, `reasoning_astar_coord`, or
`reasoning_astar_sheet`.

The coordinate-and-description training data also include solution-augmentation
variants. `k3` and `k5` provide up to three or five valid paths per map,
`allopt` provides every optimal-length solution, and `direction_targeted` and
`turn_targeted` add paths with specified direction or turn counts.

<p align="center">
  <img width="800" src="./assets/reasoning_traces_all_steps.png" alt="Examples of reasoning traces">
</p>

### Directory layout

- `train/`, `test_id/`, and `test_ood/` contain the map-size training and
  evaluation data.
- `train_10x10_fixed/` and `test_10x10_fixed/` contain the fixed-size
  start-goal-distance data.
- `direction_complex/` and `turn_complex/` contain structural evaluations
  bucketed by shortest-path direction and turn complexity.
- `train_hard_examples/` contains the 10-, 100-, and 1000-example
  direction-complexity continued-fine-tuning sets.
- `images/` contains the rendered maps used by the image-input datasets.

The data directory includes a README with the complete file-name conventions
and record schema.


## Environment

```bash
git clone https://github.com/YanNeu/frozen_ood.git
cd frozen_ood
conda env create -f environment.yml
conda activate frozen_ood
```

For Qwen3.5, install the pinned dependency overlay:

```bash
pip install --no-build-isolation -r requirements-qwen35.txt
```

The Qwen3.5 entry points leave `attn_implementation` unset by default so that
the model uses its native hybrid linear/full-attention configuration. The
optional CUDA packages in the overlay provide the accelerated linear-attention
path; without them, Transformers falls back to a slower PyTorch implementation.

## Fine-tuning

Use `src/main.py` to fine-tune Qwen2.5-VL. Set `task=sft_text` for text inputs
or `task=sft_image` for image inputs. For example:

```bash
python src/main.py epochs=10 task=sft_text \
  data_path="./data/train/train_grid.jsonl" \
  run_name="sft_grid"
```

For description and grid-based reasoning traces:

```bash
python src/main.py epochs=10 task=sft_text \
  data_path="./data/train/train_grid_reasoning_grid_desc.jsonl" \
  run_name="sft_grid_reas_grid_desc"
```

For coordinate-based conditions:

```bash
python src/main.py epochs=10 task=sft_text \
  data_path="./data/train/train_coord_reasoning_astar_coord.jsonl" \
  run_name="sft_coord_astar"
```

Use `src/main_qwen35.py` to fine-tune Qwen3.5 on text inputs:

```bash
python src/main_qwen35.py \
  model=Qwen/Qwen3.5-2B \
  epochs=10 \
  task=sft_text \
  data_path="./data/train/train_coord_reasoning_astar_coord.jsonl" \
  run_name="qwen35_2b_coord_astar"
```

Continued fine-tuning on the hard-example files uses `load_model_path`. For
example:

```bash
python src/main.py \
  load_model_path="./checkpoints/sft_coord_astar" \
  epochs=10 \
  task=sft_text \
  data_path="./data/train_hard_examples/train/train_dirs4_astar_n1000.jsonl" \
  run_name="sft_coord_astar_dirs4_n1000"
```

## Evaluation

Use `src/eval.py` to evaluate Qwen2.5-VL checkpoints:

```bash
python src/eval.py \
  load_model_path="./checkpoints/sft_coord_astar" \
  data_path="./data/test_ood/test_level7_coord.jsonl" \
  save_dir="./results_ood"
```

To evaluate a structural dataset:

```bash
python src/eval.py \
  load_model_path="./checkpoints/sft_coord_astar" \
  data_path="./data/direction_complex/test_ood/test_dirs4_level7_coord.jsonl" \
  save_dir="./results_direction_complex"
```

Use `src/eval_qwen35.py` to evaluate Qwen3.5 with its native text chat
template:

```bash
python src/eval_qwen35.py \
  --model Qwen/Qwen3.5-2B \
  --load_model_path Qwen/Qwen3.5-2B \
  --data_path ./data/test_ood/test_level7_coord.jsonl \
  --save_dir ./results/qwen35_2b_zero_shot \
  --log_file ./logs/qwen35_2b_zero_shot.log \
  --max_new_tokens 30000 \
  --temperature 0.7 \
  --top_p 0.9
```

Both evaluators extract an action sequence from a boxed answer and score it by
simulation. Exact agreement with a reference action string is not required.
Boxed answers are extracted with MathRuler.

## Citation
```
@article{neuhaus2026oodreasoning,
      title={On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks}, 
      author={Yannic Neuhaus and Chanoknan Tanchotikul and Nicolas Flammarion and Matthias Hein and Francesco Croce},
      journal={arXiv preprint arXiv:2602.15460},
      year={2026},
}
```

## Acknowledgement
The code in this repository is based on [VSP](https://arxiv.org/abs/2407.01863) and [Mirage](https://www.arxiv.org/abs/2506.17218)
