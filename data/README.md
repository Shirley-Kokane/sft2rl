# Data

Preprocessed parquet splits for the two domains studied in the paper. Each row has the schema `data_source, prompt, solution, ability, reward_model, extra_info` (chat-formatted `prompt`, boxed/lettered ground truth in `reward_model.ground_truth`).

## DeepMath (Math domain)

Derived from [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) via [`examples/data_preprocess/deepmath_dataset.py`](../examples/data_preprocess/deepmath_dataset.py).

| Split | File | Rows |
|---|---|---|
| SFT train | `deepmath/sft/train_deepmath.parquet` | 7,012 |
| RL train | `deepmath/rl/train_deepmath.parquet` | 7,012 |
| Test (held-out) | `deepmath/test_deepmath.parquet` | 1,559 |
| Extra benchmarks | `deepmath/{aime,amc,math,minerva,olympiad_bench}.parquet` | 27 / 82 / 496 / 240 / 373 |

## OpenScience (Science domain)

Derived from [OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) via [`examples/data_preprocess/openscience.py`](../examples/data_preprocess/openscience.py).

| Split | File | Rows |
|---|---|---|
| SFT train | `openscience/sft/train_openscience.parquet` | 4,500 |
| RL train | `openscience/rl/train_openscience.parquet` | 4,500 |
| Test (held-out) | `openscience/test_openscience.parquet` | 3,500 |
| Extra benchmarks | `openscience/{gpqa,gpqa_diamond}.parquet` | 448 / 198 |

## Pruning

Both domains are filtered before splitting so that SFT/RL training fits a fixed sequence-length budget and stays within a moderate difficulty band, rather than using the raw source datasets as-is:

- **Length**: prompts and solutions are length-filtered at preprocessing time (DeepMath: prompt < 1024 chars, solution ≤ 12,500 chars; OpenScience: prompt < 1024 chars, solution ≤ 7,168 chars) to fit the `max_prompt_length` / `max_response_length` budgets used at train time (2,500 / 10,000–12,048 tokens, see `scripts/train_sft.sh` and `scripts/train_grpo.sh`).
- **Difficulty**: DeepMath keeps only problems a strong reference model (Qwen3-32B) solves at least 50% of the time pass@8, excluding problems that are effectively unsolvable; OpenScience keeps only examples whose extracted answer matches the ground truth exactly under an additional LLM-judge consistency check.
- **Disjointness**: SFT train, RL train, and test are non-overlapping samples of the filtered pool, so RL never trains on data seen during SFT and the same held-out test set is used to compare all checkpoints/configs.
