

This repo is forked from [volcengine/verl](https://github.com/volcengine/verl), the HybridFlow RL training library for LLMs. It is used here as the training/eval backbone for the empirical study in **"How Much Supervised Fine-Tuning Is Enough? Dissecting the Impact of SFT Extent on Downstream RL Performance"**

## What this study is about

Staged post-training (SFT followed by RL) is the standard recipe for adapting a base/distilled model to a domain, but there's little guidance on *how much* SFT to do before handing off to RL. Too little SFT leaves the model poorly grounded; too much SFT collapses the model's output distribution (low entropy, low diversity) and leaves RL with little room to explore.

This work introduces the **Rigidity Index** — an inverse-entropy measure of how concentrated a model's top-k token probability mass is at decoding time — and tracks it alongside accuracy across SFT checkpoints. The **inflection point**, where rigidity starts growing faster than accuracy, marks the SFT checkpoint that empirically yields the best downstream RL (GRPO) performance: RL initialized near this point achieves the strongest final accuracy while retaining exploration capacity; RL initialized well past it shows diminishing accuracy gains and does not recover the lost diversity.

Key metrics used throughout the experiments (implemented in [`verl/trainer/base_utils.py`](verl/trainer/base_utils.py)):

- **Rigidity Index** — normalized entropy of the top-k decoding distribution, averaged over tokens/batch (see `rigidity_index_from_logits`).
- **Type-Token Ratio (TTR)**, similarity/BLEU scores — output diversity across sampled generations (`similarity_score_cal`).
- **KL divergence to the pretrained/reference model** — distributional drift from fine-tuning (`check_kl_to_checkpoint`).

Domains, models, and data:

- **Math**: [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K) (7k SFT train / RL train, evaluated on held-out DeepMath, OlympiadBench, MATH-500)
- **Science**: [OpenScience](https://huggingface.co/datasets/nvidia/OpenScience) (4k SFT train / RL train, evaluated on held-out OpenScience, GPQA-Diamond)
- **Models**: DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B
- **RL algorithm**: GRPO, with entropy regularization disabled and only a small reference-policy KL-loss term, to isolate the pure effect of RL optimization

## Setup

```bash
# Clone and enter the repo
git clone <this-repo-url>
cd verl_mt

# Create an environment (Python 3.10+)
conda create -n verl_mt python=3.10 -y
conda activate verl_mt

# Install verl and its dependencies
pip install -e .
pip install -r requirements.txt

# Rollout backend (pick one; vLLM is used in the paper's experiments)
pip install -r requirements_sglang.txt   # if using SGLang instead
```

See [`docs/start/install.html`](https://verl.readthedocs.io/en/latest/start/install.html) for backend-specific installation notes (FSDP, vLLM).

### Data

Preprocessed parquet data for both domains lives under [`data/`](data):

```
data/deepmath/{sft,rl}/{train,test}_deepmath.parquet
data/openscience/{sft,rl}/{train,test}_openscience.parquet
```

Each domain's SFT and RL splits are disjoint, and the same held-out test set is reused across all checkpoints/configs for consistent comparison, matching the experimental setup in the paper.

## Experiments

### 1. Supervised Fine-Tuning (SFT)

SFT is run with verl's FSDP SFT trainer via [`scripts/train_sft.sh`](scripts/train_sft.sh), checkpointing every epoch so that accuracy and Rigidity Index can be tracked across SFT extent:

```bash
./scripts/train_sft.sh deepmath deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

Pass `openscience` as the first argument to run the Science domain, and `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` as the second to use the 7B variant. Full config options are in [`verl/trainer/config/sft_trainer.yaml`](verl/trainer/config/sft_trainer.yaml).

### 2. Rigidity / diversity metrics on SFT checkpoints

For each saved SFT checkpoint, generate rollouts and compute the Rigidity Index, TTR, and similarity/BLEU diversity metrics:

```bash
python -m verl.trainer.rigidity_only \
    --model_path <path_to_sft_checkpoint> \
    --data_path <rollouts_parquet> \
    --temperature 1.0 \
    --positive_only
```

The same metrics are computed inline during generation/eval via [`verl/trainer/main_generation.py`](verl/trainer/main_generation.py) (see below), which is the path used to produce the per-checkpoint accuracy-vs-rigidity curves and identify the inflection point for each domain/model size.

### 3. Reinforcement Learning (GRPO)

RL is initialized from a chosen SFT checkpoint and trained with GRPO (entropy bonus disabled, only a small KL-loss term against the reference policy) via [`scripts/train_grpo.sh`](scripts/train_grpo.sh):

```bash
./scripts/train_grpo.sh deepmath ./checkpoints/model/deepmath-sft-DeepSeek-R1-Distill-Qwen-1.5B/global_step_150
```

Run this once per SFT checkpoint of interest (e.g. epochs 1 through 5, produced by `train_sft.sh`) to reproduce the SFT-extent-vs-RL-performance curves. Pass `openscience` as the first argument, with the matching SFT checkpoint path, to cover the Science domain.

### 4. Evaluation and reporting

[`verl/trainer/main_generation.py`](verl/trainer/main_generation.py) generates responses for a benchmark, scores pass@1 / pass@n, and writes per-checkpoint metrics (accuracy, rigidity, similarity/TTR/BLEU) to CSV:

```bash
python -m verl.trainer.main_generation \
    model.path=<path_to_checkpoint> \
    data.path=data/deepmath/test_deepmath.parquet \
    data.output_path=<output_dir>/generation_<model_ckpt>_<ckpt_name>.parquet \
    rollout.temperature=1.0
```

Results are written under `checkpoints/evals/` (see `pass_<dataset>_format_<bool>_<model_ckpt>_<ckpt_name>.csv`), aggregating `model_path, dataset, pass@1, pass@n, response_length, temperature, avg/min similarity, ttr, avg/min bleu` for downstream plotting of accuracy vs. Rigidity Index.

### Converting FSDP checkpoints to Hugging Face format

Trained actor checkpoints are sharded FSDP checkpoints; convert them to a standard HF checkpoint before running generation/eval elsewhere:

```bash
python convert_fsdp_hf.py <fsdp_actor_ckpt_dir> <base_hf_model_dir> <output_hf_dir>
```

[`convert_model.sh`](convert_model.sh) shows an end-to-end example of syncing checkpoints and converting a sweep of RL global steps to HF format.

## Repository layout (additions on top of upstream verl)

- `scripts/train_sft.sh` — SFT launch script (FSDP SFT trainer) for the DeepMath/OpenScience domains.
- `scripts/train_grpo.sh` — GRPO RL launch script, initialized from an SFT checkpoint.
- `verl/trainer/base_utils.py` — Rigidity Index, KL-to-reference, and diversity (TTR/BLEU/Jaccard) metric implementations.
- `verl/trainer/rigidity_only.py` — standalone script to compute Rigidity Index over saved rollouts for a given checkpoint.
- `verl/trainer/main_generation.py` — generation + evaluation entrypoint, extended to log rigidity and diversity metrics per checkpoint.
- `data/deepmath/`, `data/openscience/` — SFT/RL/test parquet splits for the Math and Science domains.
- `rllm/` — dataset loaders, reward functions, and tooling used to prepare and score the domain data.
- `convert_fsdp_hf.py`, `convert_model.sh` — FSDP-to-HuggingFace checkpoint conversion utilities.
- `checkpoints/evals/` — aggregated per-checkpoint accuracy/rigidity/diversity CSVs used for analysis and plotting.

## Upstream verl

For everything else — supported algorithms (PPO, GRPO, RLOO, DAPO, etc.), backend support (FSDP/FSDP2/Megatron, vLLM/SGLang), multi-turn tool use, and general documentation — see the upstream [verl documentation](https://verl.readthedocs.io/en/latest/index.html) and [volcengine/verl](https://github.com/volcengine/verl).


