# QK-RESTORE

Code for **QK-RESTORE** — zero-training recall recovery for hybrid linear-attention models after Chain-of-Thought SFT — and the full **HALO distillation pipeline** (Stage 1–3 + CoT-SFT) that produces the models.

---

## Repository Structure

```
qk-restore/
├── qk_restore.py             QK-RESTORE: generate all variant checkpoints
├── eval_qkrr.sh              End-to-end eval pipeline (generate → NIAH → results table)
├── halo/
│   ├── stage1.py             Stage 1: hidden-state alignment (MSE distillation)
│   ├── stage2.py             Stage 2: end-to-end KL divergence distillation
│   ├── stage3.py             Stage 3: long-context fine-tuning (no teacher)
│   ├── stage_cot_sft.py      CoT-SFT: chain-of-thought supervised fine-tuning
│   ├── build_hf_ckpt.py      Convert training checkpoint → HuggingFace format
│   ├── arguments.py          Typed argument class (tap-based)
│   ├── preparation.py        Accelerator, dataloaders, optimizers setup
│   ├── utils.py              Parameter counting helpers
│   ├── configs/
│   │   ├── accelerate/       Accelerate / DeepSpeed configs
│   │   ├── model/            HypeNet model configs (2B, 5B, 9B)
│   │   └── training/         Per-stage training hyperparameters
│   ├── modeling/
│   │   └── hypenet/          HypeNetForCausalLM, LightningAttention, GDN, KDA, config
│   │                         vllm_model.py — vLLM-compatible model (MambaBase integration)
│   ├── trainer/
│   │   └── trainer.py        LMTrainer (main training loop)
│   ├── optim/                WSD and cosine LR schedulers
│   └── scripts/
│       ├── mix-data/         Stage 1/2/3 launch scripts (2B, 5B, 9B)
│       ├── cot_sft/          CoT-SFT launch scripts
│       └── analysis/         QK-RESTORE eval scripts + vLLM serve
├── layer-selection/
│   ├── convert.py            Build single-layer-linearized hybrid model for sweep
│   ├── layer_analysis.py     Score layer importance (HALO metric)
│   ├── register_hypenet.py   Register HypeNetForCausalLM with AutoClasses
│   ├── configs/
│   │   ├── ds.yaml           DeepSpeed / 7-GPU accelerate config (for eval)
│   │   └── infer.yaml        8-GPU multi-GPU accelerate config (for larger models)
│   ├── modeling/
│   │   └── hypenet/          Same modeling package as halo/ (local copy for eval)
│   └── scripts/
│       ├── layer_sweep_2b.sh  Full 28-layer sweep for HypeNet-2B
│       ├── layer_sweep_5b.sh  Full 36-layer sweep for HypeNet-5B
│       ├── layer_sweep_9b.sh  Full 36-layer sweep for HypeNet-9B
│       ├── layer_analysis_2b.sh   Score 2B results
│       ├── layer_analysis_5b.sh   Score 5B results
│       ├── layer_analysis_9b.sh   Score 9B results
│       └── evaluation/
│           ├── eval_niah.sh   NIAH eval for a single model (up to 262K context)
│           └── eval_math.sh   Math eval (mathqa_local, aqua_rat)
├── environment_hybridrl.yml  conda env export (training: HALO stages 1–3, CoT-SFT)
├── environment_hype5.yml     conda env export (eval + layer-selection)
└── README.md
```

---

## HALO: Distillation Pipeline

HALO converts a pre-trained Transformer (teacher) into a **HypeNet** hybrid model with LightningAttention RNN layers replacing most full-attention layers.

### Three-stage training

| Stage | Script | Purpose | Loss |
|-------|--------|---------|------|
| 1 | `stage1.py` | Hidden-state alignment — each RNN layer learns to mimic the corresponding teacher attention layer's output | MSE |
| 2 | `stage2.py` | End-to-end KL distillation — the full student is trained against teacher logits | KL divergence |
| 3 | `stage3.py` | Long-context fine-tuning — no teacher, extends context to 16K+ | CE |
| SFT | `stage_cot_sft.py` | Chain-of-thought supervised fine-tuning on math reasoning data | CE (prompt masked) |

### Quick start

```bash
cd halo/

# Stage 1: hidden-state alignment
bash scripts/mix-data/stage1_hypenet_2b.sh

# Stage 2: KL distillation (requires Stage 1 checkpoint)
bash scripts/mix-data/stage2_hypenet_2b.sh \
    stage1_run_name=<your_stage1_run> stage1_ckpt=ckpt_20000

# Stage 3: long-context fine-tuning (requires Stage 2 checkpoint)
bash scripts/mix-data/stage3_hypenet_2b.sh \
    stage2_run_name=<your_stage2_run> stage2_ckpt=ckpt_20000

# Convert to HuggingFace format (run after each stage as needed)
python build_hf_ckpt.py \
    --ckpt_path=results/hypenet/<run_name> \
    --config_path=configs/model/hypenet/hypenet-2b-mix_old_selection.json \
    --tok_path=/path/to/qwen3-1.7B \
    --ckpt=ckpt_500 --stage=3 \
    --out_path=ckpts/hypenet-2b

# CoT-SFT
bash scripts/cot_sft/cot_sft_2b.sh init_from=ckpts/hypenet-2b/ckpt_500
```

### Model architecture

Model configs live in `configs/model/hypenet/`. The key field is `mixer_types`, a per-layer list of `"attn"` or `"lightning-attn"`. Example for HypeNet-2B (`hypenet-2b-mix_old_selection.json`): 28 layers total, 7 full-attention layers at positions 2, 4, 8, 9, 14, 25, 26.

Teacher model: `qwen3-1.7B` (all experiments).

### Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "ckpts/hypenet-2b/ckpt_500",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("ckpts/hypenet-2b/ckpt_500", trust_remote_code=True)
```

---

## QK-RESTORE

After CoT-SFT, the full-attention layers' W^Q and W^K drift toward local patterns, degrading long-context recall. QK-RESTORE copies W^Q and W^K back from the pre-SFT checkpoint while keeping everything else at post-SFT values.

### Usage

```bash
# Generate all variant checkpoints
python qk_restore.py \
    --pre_sft  /path/to/pre_sft_checkpoint \
    --post_sft /path/to/post_sft_checkpoint \
    --out_dir  /path/to/variants/ \
    --recall_layers 2,4,8,9,14,25,26 \
    --configs  QKRR-all,SALR-all,Baseline

# Generate + evaluate end-to-end
bash eval_qkrr.sh \
    pre_sft=/path/to/pre_sft \
    post_sft=/path/to/post_sft \
    variant_dir=/path/to/variants \
    gpus=0,1,2,3 \
    run_math=1
```

### Variants

| Config | What is restored |
|--------|-----------------|
| `QKRR-all` | Q, K of **all** full-attention layers — **main method** |
| `KONLY-all` | Only K (ablation) |
| `VONLY-all` | Only V, O (ablation — should not help recall) |
| `QKRR-Procrustes` | Enforce R_new = R_pre while minimising ‖W^Q_new − W^Q_post‖_F |

### `--recall_layers`

Pass the full-attention layer indices ordered from most to least recall-critical. For HypeNet-2B the default is `2,4,8,9,14,25,26`. For other model sizes, identify full-attention layers from the `mixer_types` field in the model config, then run a layer-sweep to determine recall importance ordering.

### Key flags

**`qk_restore.py`**

| Flag | Default | |
|------|---------|---|
| `--recall_layers` | `2,4,8,9,14,25,26` | Layer indices, most- to least-critical |
| `--dtype` | `bfloat16` | |
| `--procrustes_mode` | `fix_k` | `fix_k` or `fix_q` |

**`eval_qkrr.sh`**

| Variable | Default | |
|----------|---------|---|
| `niah_lens` | `4096,8192,16384,32768,65536` | Context lengths to evaluate |
| `max_length` | `131072` | Context window passed to lm_eval |
| `gpus` | `0,1,2,3,4,5,6,7` | |
| `skip_gen` | `0` | Set to `1` to skip checkpoint generation |
| `run_math` | `0` | Set to `1` to also run math evaluation |

---

## Layer Selection

Before running HALO training, the `mixer_types` config (which layers remain full-attention vs. linearized) must be determined. `layer-selection/` implements a layer importance sweep for this purpose.

### Method

For each layer l in the model, build a hybrid model with that single layer replaced by a LightningAttention RNN, then evaluate:
- **Commonsense**: `hellaswag`, `arc_easy`, `arc_challenge`
- **Retrieval**: `ruler_qa_squad`, `fda`, `swde`

Layers that cause the largest capability drops when linearized are kept as full-attention.

Scoring uses the **HALO metric**: `score(l) = (rt_max - rt_l) / (cs_max - cs_l + ε)`, capped at 50. Higher score = more important to keep as full attention.

### Quick start

```bash
cd layer-selection/

# Step 1: run the layer sweep (builds & evaluates one model per layer)
bash scripts/layer_sweep_2b.sh \
    ORIG_MODEL=/path/to/qwen3-1.7B \
    RNN_HF_MODEL=/path/to/stage1_hf/ckpt_20000 \
    RESULTS_BASE_DIR=results/hypenet_2b/layer_sweep

# Step 2: score layer importance
bash scripts/layer_analysis_2b.sh \
    RESULTS_BASE_DIR=results/hypenet_2b/layer_sweep
```

The analysis prints a ranked table and writes `figures/*.png`. The top-K layers from this ranking become the `"attn"` entries in `halo/configs/model/hypenet/hypenet-2b-mix_old_selection.json`.

### Standalone eval scripts

```bash
# NIAH (needle-in-a-haystack) up to 262K context
bash scripts/evaluation/eval_niah.sh
```

---

## Requirements

```
torch >= 2.0
transformers >= 4.40
accelerate
deepspeed
lm_eval       (for evaluation)
tap-python    (for arguments.py)
datasets
swanlab / tensorboard  (for experiment tracking)
```

Full environment exports are included:

| File                       | Purpose                          |
| ----------------------------| ----------------------------------|
| `environment_hybridrl.yml` | HALO training (stages 1–3, eval) |
| `environment_hype5.yml`    | CoT-training                     |

```bash
conda env create -f environment_hybridrl.yml
conda env create -f environment_hype5.yml
```

---

## vLLM Integration

HypeNet requires a custom vLLM registration to serve with `vllm serve` or use with vLLM-based RL frameworks (e.g. Verl). The integration is implemented in `halo/modeling/hypenet/vllm_model.py` and targets **vLLM ≥ 0.19.0**.

### Key design

- `HypeNetLinearAttn` inherits `MambaBase` — vLLM allocates a `(max_batch, H, D, D)` recurrent state buffer per lightning-attention layer automatically
- `HypeNetForCausalLMVLLM` implements `HasInnerState` + `IsHybrid`, registered as `"HypeNetForCausalLM"` in vLLM's `ModelRegistry`
- State routing: vLLM passes `attn_metadata` as a `dict[layer_prefix, LinearAttentionMetadata]` (vLLM 0.19+ API); each layer reads its own entry by prefix


---

## Citation

```bibtex
@article{qk-restore-2026,
  title   = {QK-RESTORE: Zero-Training Recall Recovery for Hybrid Linear-Attention Models},
  author  = {Xinyu Zhou and others},
  year    = {2026},
}
```
