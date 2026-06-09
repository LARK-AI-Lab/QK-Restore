#!/usr/bin/env python3
"""
qk_restore.py — QK-RESTORE: Post-SFT Recall Recovery via QK-Restoration

Zero-training, post-hoc repair for hybrid linear-attention models whose recall
degrades after Chain-of-Thought SFT.  Restores W^Q and W^K from the pre-SFT
checkpoint while keeping W^V, W^O, FFN, and all linear-attention layers at
post-SFT values.

Variants produced
-----------------
  QKRR-all        — restore Q, K of ALL full-attention layers  [main]
  KONLY-all       — restore only W^K (ablation)
  VONLY-all       — restore only W^V, W^O (ablation — should not help recall)
  QKRR-Procrustes — enforce R_new = R_pre while minimising ‖W^Q_new − W^Q_post‖_F

Usage
-----
    python qk_restore.py \\
        --pre_sft  /path/to/pre_sft_checkpoint \\
        --post_sft /path/to/post_sft_checkpoint \\
        --out_dir  /path/to/output/ \\
        [--recall_layers 2,4,8,9,14,25,26]

Output
------
Each variant is saved as a self-contained HuggingFace checkpoint directory
under out_dir/ and can be loaded with AutoModelForCausalLM.from_pretrained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Default paths (HypeNet-2B)
# ---------------------------------------------------------------------------

_PRE_SFT_DEFAULT  = "/path/to/results/hypenet-Stage3/hf/ckpt_500"
_POST_SFT_DEFAULT = "/path/to/results/hypenet_CoT_SFT/ckpt_100"
_OUT_DIR_DEFAULT  = "/path/to/results/qk_restore_output"

# Full-attention layer indices for HypeNet-2B, ordered by recall importance.
_HYPENET_2B_FULL_ATTN_LAYERS: list[int] = [2, 4, 8, 9, 14, 25, 26]


# ---------------------------------------------------------------------------
# Parameter key helpers
# ---------------------------------------------------------------------------

def _qk_keys(layer_idx: int) -> list[str]:
    """Q and K projection + per-head norm keys for a full-attention layer."""
    base = f"model.layers.{layer_idx}.self_attn"
    keys: list[str] = []
    for proj in ("q_proj", "k_proj"):
        keys += [f"{base}.{proj}.weight", f"{base}.{proj}.bias"]
    for norm in ("q_norm", "k_norm"):
        keys += [f"{base}.{norm}.weight", f"{base}.{norm}.bias"]
    return keys


def _k_only_keys(layer_idx: int) -> list[str]:
    base = f"model.layers.{layer_idx}.self_attn"
    return [f"{base}.k_proj.weight", f"{base}.k_proj.bias",
            f"{base}.k_norm.weight", f"{base}.k_norm.bias"]


def _vo_only_keys(layer_idx: int) -> list[str]:
    base = f"model.layers.{layer_idx}.self_attn"
    return [f"{base}.v_proj.weight", f"{base}.v_proj.bias",
            f"{base}.o_proj.weight", f"{base}.o_proj.bias"]


# ---------------------------------------------------------------------------
# Core weight operations
# ---------------------------------------------------------------------------

def _restore_weights(
    post_sd: dict[str, torch.Tensor],
    pre_sd: dict[str, torch.Tensor],
    keys: list[str],
) -> dict[str, torch.Tensor]:
    """Copy *keys* from *pre_sd* into a clone of *post_sd*. Missing keys are skipped."""
    result = {k: v.clone() for k, v in post_sd.items()}
    restored, skipped = 0, 0
    for key in keys:
        if key in pre_sd and key in result:
            result[key].copy_(pre_sd[key])
            restored += 1
        else:
            skipped += 1
    print(f"    restored {restored} tensors, skipped {skipped}")
    return result


# ---------------------------------------------------------------------------
# Procrustes-constrained restoration
# ---------------------------------------------------------------------------

def _apply_procrustes_to_sd(
    post_sd: dict[str, torch.Tensor], pre_sd: dict[str, torch.Tensor],
    layers: list[int], mode: str = "fix_k", regularize: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Enforce R_new = R_pre while minimising ‖W^Q_new − W^Q_post‖_F (Lagrange closed form)."""
    result = {k: v.clone() for k, v in post_sd.items()}
    for li in layers:
        base  = f"model.layers.{li}.self_attn"
        q_key = f"{base}.q_proj.weight"
        k_key = f"{base}.k_proj.weight"
        if any(k not in post_sd or k not in pre_sd for k in (q_key, k_key)):
            print(f"  [WARN] Layer {li}: keys not found — skipping")
            continue

        qp, kp = post_sd[q_key].float(), post_sd[k_key].float()
        qr, kr = pre_sd[q_key].float(),  pre_sd[k_key].float()
        R_pre  = qr @ kr.T

        if mode == "fix_k":
            gram  = kr @ kr.T + regularize * torch.eye(kr.shape[0], device=kr.device, dtype=kr.dtype)
            q_new = qp + (R_pre - qp @ kr.T) @ torch.linalg.inv(gram) @ kr
            k_new = kr
        elif mode == "fix_q":
            gram  = qr @ qr.T + regularize * torch.eye(qr.shape[0], device=qr.device, dtype=qr.dtype)
            k_new = kp + (R_pre.T - kp @ qr.T) @ torch.linalg.inv(gram) @ qr
            q_new = qr
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'fix_k' or 'fix_q'.")

        err = (q_new @ k_new.T - R_pre).norm().item()
        print(f"  Layer {li}: mode={mode}, constraint error={err:.2e}")
        result[q_key].copy_(q_new.to(post_sd[q_key].dtype))
        result[k_key].copy_(k_new.to(post_sd[k_key].dtype))
    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_variant(
    base_model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    state_dict: dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    base_model.load_state_dict(state_dict)
    base_model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_DEFAULT_CONFIGS = "QKRR-all,KONLY-all,VONLY-all,QKRR-Procrustes"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QK-RESTORE: post-SFT recall recovery via QK restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pre_sft",  default=_PRE_SFT_DEFAULT)
    p.add_argument("--post_sft", default=_POST_SFT_DEFAULT)
    p.add_argument("--out_dir",  default=_OUT_DIR_DEFAULT)
    p.add_argument("--configs",  default=_DEFAULT_CONFIGS,
                   help="Comma-separated variants to generate "
                        "(QKRR-all, KONLY-all, VONLY-all, QKRR-Procrustes)")
    p.add_argument("--recall_layers", default=None,
                   help="Comma-separated full-attention layer indices, most- to least-critical")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--procrustes_mode", default="fix_k", choices=["fix_k", "fix_q"])
    p.add_argument("--procrustes_reg",  type=float, default=1e-6)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    recall_layers = (
        [int(x) for x in args.recall_layers.split(",")]
        if args.recall_layers else _HYPENET_2B_FULL_ATTN_LAYERS
    )

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  QK-RESTORE")
    print(f"  pre_sft       : {args.pre_sft}")
    print(f"  post_sft      : {args.post_sft}")
    print(f"  out_dir       : {out_dir}")
    print(f"  recall_layers : {recall_layers}")
    print(f"  configs       : {configs}")
    print(f"{'='*60}\n")

    print("[1/3] Loading post-SFT checkpoint ...")
    post_model = AutoModelForCausalLM.from_pretrained(
        args.post_sft, torch_dtype=torch_dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.post_sft, trust_remote_code=True)
    post_sd = {k: v.clone() for k, v in post_model.state_dict().items()}

    print("[2/3] Loading pre-SFT checkpoint ...")
    pre_model = AutoModelForCausalLM.from_pretrained(
        args.pre_sft, torch_dtype=torch_dtype, trust_remote_code=True
    )
    pre_sd = {k: v.clone() for k, v in pre_model.state_dict().items()}
    del pre_model

    print("[3/3] Generating variants ...\n")

    for config_name in configs:
        print(f"── {config_name}")
        out_path = out_dir / config_name.lower().replace("-", "_").replace(".", "_")

        if out_path.exists():
            print(f"  [SKIP] already exists\n")
            continue

        if config_name == "QKRR-all":
            keys = [k for li in recall_layers for k in _qk_keys(li)]
            new_sd = _restore_weights(post_sd, pre_sd, keys)

        elif config_name == "KONLY-all":
            keys = [k for li in recall_layers for k in _k_only_keys(li)]
            new_sd = _restore_weights(post_sd, pre_sd, keys)

        elif config_name == "VONLY-all":
            keys = [k for li in recall_layers for k in _vo_only_keys(li)]
            new_sd = _restore_weights(post_sd, pre_sd, keys)

        elif config_name == "QKRR-Procrustes":
            new_sd = _apply_procrustes_to_sd(post_sd, pre_sd, recall_layers,
                                              mode=args.procrustes_mode,
                                              regularize=args.procrustes_reg)

        else:
            print(f"  [WARN] unknown config '{config_name}' — skipping\n")
            continue

        _save_variant(post_model, tokenizer, new_sd, out_path)
        print()

    post_model.load_state_dict(post_sd)

    print(f"\n{'='*60}")
    print(f"  Done. Variants saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
