"""
convert.py — Build a hybrid HypeNet model for one layer configuration.

For each layer in a sweep, this script:
  1. Loads the original full-attention model (orig_model)
  2. Loads an all-RNN Stage 1 model (rnn_hf_model) or rebuilds one from a
     raw Stage 1 checkpoint (rnn_base_path + ckpt_num)
  3. Replaces the selected layer(s) with RNN mixers
  4. Saves the resulting hybrid model as a HuggingFace checkpoint

Usage
-----
python convert.py \\
    --orig_model    /path/to/qwen3-1.7B \\
    --rnn_hf_model  /path/to/stage1_hf/ckpt_20000 \\
    --output_root   /tmp/hybrid_layer4 \\
    --layer_indices 4 \\
    --rnn_type      lightning-attn

# Or using a raw Stage 1 checkpoint instead of rnn_hf_model:
python convert.py \\
    --orig_model   /path/to/qwen3-1.7B \\
    --rnn_base_path /path/to/stage1_run \\
    --ckpt_num     20000 \\
    --output_root  /tmp/hybrid_layer4 \\
    --layer_indices 4 \\
    --rnn_type     lightning-attn
"""

import os
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from modeling.hypenet.modeling_hypenet import HypeNetForCausalLM
from modeling.hypenet.configuration_hypenet import HypeNetConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, name, device):
    """Run a deterministic single-prompt generation for sanity checking."""
    model.eval()
    model.to(device)
    model.to(dtype=torch.bfloat16)

    prompt = "My name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== {name} ===")
    print(text)
    print("=" * 60)
    return text


def load_rnn_checkpoint(ckpt_path):
    """
    Load and normalize a raw Stage 1 checkpoint (safetensors).

    Strips prefixes:
      student_layer.*   -> *
      model.model.*     -> model.*
      model.lm_head     -> lm_head

    Ties lm_head if missing.
    """
    raw_state = load_file(ckpt_path)
    state = {}

    for k, v in raw_state.items():
        if "teacher_layer." in k:
            continue
        k = (
            k.replace("student_layer.", "")
             .replace("model.model.", "model.")
             .replace("model.lm_head", "lm_head")
        )
        state[k] = v

    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    return state


# ---------------------------------------------------------------------------
# Hybrid construction
# ---------------------------------------------------------------------------

def create_hybrid_model(
    attention_model,
    rnn_only_model,
    base_config_path,
    student_config_path,
    rnn_type,
    layer_indices,
):
    """
    Create a hybrid model by replacing the selected layers with RNN mixers.

    All models share the HypeNetForCausalLM class; behavior is controlled
    purely by config.mixer_types.
    """
    config = HypeNetConfig.from_json_file(base_config_path)
    student_config = HypeNetConfig.from_json_file(student_config_path)

    config.mixer_types = ["attn"] * config.num_hidden_layers
    for idx in layer_indices:
        config.mixer_types[idx] = rnn_type

    if rnn_type in ["lightning-attn", "lightning_attn"]:
        config.lightning_nh                = student_config.lightning_nh
        config.lightning_nkv               = student_config.lightning_nkv
        config.lightning_head_dim          = student_config.lightning_head_dim
        config.lightning_use_rope          = student_config.lightning_use_rope
        config.lightning_use_qk_norm       = student_config.lightning_use_qk_norm
        config.lightning_use_output_gate   = student_config.lightning_use_output_gate
        config.lightning_use_output_norm   = student_config.lightning_use_output_norm
    elif rnn_type == "kda":
        config.kda_num_heads  = student_config.kda_num_heads
        config.kda_head_dim   = student_config.kda_head_dim
        config.kda_use_rope   = student_config.kda_use_rope
        config.kda_use_qk_norm = student_config.kda_use_qk_norm
    elif rnn_type == "gdn":
        config.gdn_nh          = student_config.gdn_nh
        config.gdn_nkv         = student_config.gdn_nkv
        config.gdn_use_rope    = student_config.gdn_use_rope
        config.gdn_use_qk_norm = student_config.gdn_use_qk_norm
        config.gdn_use_gate    = student_config.gdn_use_gate
        config.gdn_use_short_conv  = student_config.gdn_use_short_conv
        config.gdn_activation  = student_config.gdn_activation

    hybrid_model = HypeNetForCausalLM(config)
    hybrid_state = hybrid_model.state_dict()

    # 1) Non-layer parameters from attention model
    for k, v in attention_model.state_dict().items():
        if "model.layers." not in k:
            hybrid_state[k] = v

    # 2) Full-attention layers (not replaced)
    for k, v in attention_model.state_dict().items():
        if "model.layers." in k:
            layer_id = int(k.split(".")[2])
            if layer_id not in layer_indices:
                hybrid_state[k] = v

    # 3) RNN layers (replaced)
    for k, v in rnn_only_model.state_dict().items():
        if "model.layers." in k:
            layer_id = int(k.split(".")[2])
            if layer_id in layer_indices:
                hybrid_state[k] = v

    hybrid_model.load_state_dict(hybrid_state, strict=True)
    return hybrid_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    orig_model,
    rnn_base_path,
    ckpt_num,
    output_root,
    layer_indices,
    rnn_type,
    rnn_hf_model=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config_path = f"{rnn_base_path}/orig_config.json"
    if rnn_hf_model is not None:
        student_config_path = os.path.join(rnn_hf_model, "config.json")
    else:
        rnn_ckpt_path = f"{rnn_base_path}/ckpt_{ckpt_num}/model.safetensors"
        student_config_path = f"{rnn_base_path}/student_config.json"

    print(f"[INFO] Attention base model : {orig_model}")
    if rnn_hf_model is not None:
        print(f"[INFO] RNN HF model        : {rnn_hf_model}")
    else:
        print(f"[INFO] RNN checkpoint       : {rnn_ckpt_path}")
    print(f"[INFO] Replace layers       : {layer_indices}")
    print(f"[INFO] RNN mixer type        : {rnn_type}")

    tokenizer = AutoTokenizer.from_pretrained(orig_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attention-only model (source for non-replaced layers)
    attention_model = HypeNetForCausalLM.from_pretrained(
        orig_model,
        torch_dtype=torch.bfloat16,
    )
    run_inference(attention_model, tokenizer, "Attention model", device)

    # RNN-only model (source for replaced layers)
    if rnn_hf_model is not None:
        rnn_only_model = HypeNetForCausalLM.from_pretrained(
            rnn_hf_model,
            torch_dtype=torch.bfloat16,
        )
        allowed_rnn_types = {
            rnn_type,
            rnn_type.replace("-", "_"),
            rnn_type.replace("_", "-"),
        }
        invalid_layers = [
            idx
            for idx, mixer_type in enumerate(rnn_only_model.config.mixer_types)
            if mixer_type not in allowed_rnn_types
        ]
        if invalid_layers:
            raise ValueError(
                f"rnn_hf_model is not an all-RNN Stage 1 export. "
                f"Non-{rnn_type} layers found at: {invalid_layers}"
            )
    else:
        rnn_config = HypeNetConfig.from_json_file(student_config_path)
        rnn_config.mixer_types = [rnn_type] * rnn_config.num_hidden_layers
        rnn_only_model = HypeNetForCausalLM(rnn_config)
        rnn_state = load_rnn_checkpoint(rnn_ckpt_path)
        rnn_only_model.load_state_dict(rnn_state, strict=True)

    run_inference(rnn_only_model, tokenizer, "RNN-only model", device)

    # Build hybrid
    hybrid_model = create_hybrid_model(
        attention_model=attention_model,
        rnn_only_model=rnn_only_model,
        base_config_path=base_config_path,
        student_config_path=student_config_path,
        rnn_type=rnn_type,
        layer_indices=layer_indices,
    )

    # Save
    os.makedirs(output_root, exist_ok=True)
    hybrid_model.config.register_for_auto_class()
    hybrid_model.model.register_for_auto_class("AutoModel")
    hybrid_model.register_for_auto_class("AutoModelForCausalLM")
    tokenizer.save_pretrained(output_root)
    hybrid_model.save_pretrained(output_root, safe_serialization=True)
    print(f"✅ Hybrid model saved to: {output_root}")

    run_inference(hybrid_model, tokenizer, "Hybrid model", device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a hybrid HypeNet model with selected RNN layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--orig_model",    type=str, required=True,
                        help="Path to the original full-attention HF model")
    parser.add_argument("--rnn_base_path", type=str, default=None,
                        help="Path to the Stage 1 training run directory "
                             "(required when --rnn_hf_model is not provided)")
    parser.add_argument("--ckpt_num",      type=int, default=None,
                        help="Stage 1 checkpoint step number")
    parser.add_argument("--output_root",   type=str, required=True,
                        help="Output directory for the hybrid HF checkpoint")
    parser.add_argument("--rnn_hf_model",  type=str, default=None,
                        help="Optional: path to a pre-exported all-RNN Stage 1 HF "
                             "checkpoint.  When provided, rnn_base_path/ckpt_num "
                             "are not needed.")
    parser.add_argument("--layer_indices", type=str, default="4",
                        help="Comma-separated layer indices to replace, e.g. 0,4,8")
    parser.add_argument("--rnn_type",      type=str, default="lightning-attn",
                        help="Mixer type for replaced layers (lightning-attn, gdn, kda)")

    args = parser.parse_args()
    if args.rnn_hf_model is None and (args.rnn_base_path is None or args.ckpt_num is None):
        parser.error("Either --rnn_hf_model or both --rnn_base_path and --ckpt_num are required.")

    layer_indices = [int(x) for x in args.layer_indices.split(",") if x]

    main(
        orig_model=args.orig_model,
        rnn_base_path=args.rnn_base_path,
        ckpt_num=args.ckpt_num,
        output_root=args.output_root,
        layer_indices=layer_indices,
        rnn_type=args.rnn_type,
        rnn_hf_model=args.rnn_hf_model,
    )
