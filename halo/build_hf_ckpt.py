import torch
import json
from modeling.hypenet import HypeNetForCausalLM, HypeNetConfig
from modeling.hypenet.modeling_hypenet import HypeNetModel
from transformers import AutoTokenizer
from torch import Tensor
from pathlib import Path
from safetensors.torch import load_file
from tap import Tap


def load_state_dict(ckpt_dir: Path) -> dict[str, Tensor]:  # type: ignore
    # Try to load from a safetensors checkpoint.
    safetensors_path = ckpt_dir / 'model.safetensors'
    if safetensors_path.exists():
        return load_file(safetensors_path)

    # Try to load from a FSDP checkpoint.
    fsdp_path = ckpt_dir / 'pytorch_model_fsdp.bin'
    if fsdp_path.exists():
        state_dict = torch.load(fsdp_path, map_location="cpu")
        return state_dict

    # Try to load from a DeepSpeed checkpoint.
    ds_index_path = ckpt_dir / 'output_dir/pytorch_model.bin.index.json'
    if ds_index_path.exists():
        with open(ds_index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        unique_files = set(weight_map.values())

        state_dict = {}
        for shard_file in sorted(unique_files):
            shard_path = ckpt_dir / 'output_dir' / shard_file
            shard = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard)

        return state_dict

    raise ValueError(f"No checkpoint found in {ckpt_dir}")


def normalize_stage1_state_dict(
    state_dict: dict[str, Tensor],
    config: HypeNetConfig,
    force_all_student: bool = False,
) -> dict[str, Tensor]:
    """
    Convert a Stage 1 HiddenStateAligner checkpoint into a standalone HypeNet
    state dict.

    For attention layers, keep `teacher_layer.*`.
    For converted RNN layers, keep `student_layer.*`.
    """
    new_state_dict: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model."):]

        if ".layers." in key:
            if force_all_student:
                if "teacher_layer" in key:
                    continue
                key = key.replace("student_layer.", "")
            else:
                layer_idx = int(key.split(".layers.")[1].split(".")[0])
                mixer_type = config.mixer_types[layer_idx]

                if mixer_type == "attn":
                    if "student_layer" in key:
                        continue
                    key = key.replace("teacher_layer.", "")
                else:
                    if "teacher_layer" in key:
                        continue
                    key = key.replace("student_layer.", "")

        new_state_dict[key] = value

    return new_state_dict


def load_with_stage1_tolerance(
    model: HypeNetForCausalLM,
    state_dict: dict[str, Tensor],
    config: HypeNetConfig,
) -> None:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise ValueError(f"Unexpected keys when loading stage1 checkpoint: {unexpected_keys}")

    for missing_key in missing_keys:
        # Some configs add output gates to attention layers after initialization.
        if config.attn_use_output_gate and ".self_attn.o_gate." in missing_key:
            continue
        raise ValueError(f"Missing key when loading stage1 checkpoint: {missing_key}")


class Args(Tap):
    only_state_dict: int = 0
    stage: int = 3
    stage1_all_rnn: int = 0
    rnn_type: str = "lightning-attn"
    tok_path: str = '/path/to/qwen3-1.7b'
    ckpt_path: str = "results/hypenet/e1_stage3_hypenet-2b_stage3_"
    config_path: str = "configs/model/hypenet/hypenet-2b.json"
    out_path: str = "ckpts/hypenet-2b_stage3"
    ckpt: str = "ckpt_500"
    attn_logits_scaling: str = ""  # e.g. "hype 500"; leave empty to keep config value


args = Args().parse_args()
ckpt_path = Path(args.ckpt_path) / args.ckpt
tok_path = Path(args.tok_path)
config_path = Path(args.config_path)
if args.stage == 1 and not config_path.exists():
    fallback_config_path = ckpt_path.parent / "student_config.json"
    if fallback_config_path.exists():
        print(f"Stage 1 detected, falling back to student config at {fallback_config_path}")
        config_path = fallback_config_path
out_path = Path(args.out_path) / args.ckpt
out_path.mkdir(exist_ok=True, parents=True)
assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
assert config_path.exists(), f"Config {config_path} does not exist"
assert tok_path.exists(), f"Tokenizer {tok_path} does not exist"

print(f"Loading tokenizer from {tok_path}")
tokenizer = AutoTokenizer.from_pretrained(tok_path)
print(f"Loading config from {config_path}")
config = HypeNetConfig.from_json_file(config_path)
if args.stage == 1 and bool(args.stage1_all_rnn):
    config.mixer_types = [args.rnn_type] * config.num_hidden_layers

print(config)

print("Instantiating model")
model = HypeNetForCausalLM(config=config)

print(f"Loading state dict from {ckpt_path}")
state_dict = load_state_dict(ckpt_path)
print(list(state_dict.keys()))

if args.stage == 1:
    state_dict = normalize_stage1_state_dict(
        state_dict,
        config,
        force_all_student=bool(args.stage1_all_rnn),
    )
elif args.stage == 2:
    new_state_dict = {}
    for key, val in state_dict.items():
        if 'teacher_model' in key:
            continue
        key = key.replace('student_model.', '')
        print(f"Inserting parameter {key}...")
        new_state_dict[key] = val.to(torch.bfloat16)
    state_dict = new_state_dict

if bool(args.only_state_dict):
    print(f"Saving state dict to {out_path}")
    torch.save(state_dict, out_path / 'state_dict.pt')
    exit()

if 'lm_head.weight' not in state_dict:
    assert config.tie_word_embeddings, "lm_head.weight is not in the state dict, so tie_word_embeddings must be True"
    print("Tying word embeddings...")
    state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']

print("Loading parameters into model")
if args.stage == 1:
    load_with_stage1_tolerance(model, state_dict, config)
else:
    model.load_state_dict(state_dict)
model = model.to(torch.bfloat16)

# register_for_auto_class on the top-level model triggers save_pretrained to copy
# modeling_hypenet.py and all its relative-imported dependencies (cache.py,
# lightning_attn.py, gdn.py, kda.py, modeling_qwen3.py, configuration_hypenet.py).
# NOTE: model.model.register_for_auto_class("AutoModel") is intentionally omitted —
# save_pretrained only processes type(model), so the AutoModel entry would be silently
# dropped. We write the complete auto_map manually after saving instead.
print("Registering model")
model.register_for_auto_class("AutoModelForCausalLM")

print(f"Saving model to {out_path}")
model.save_pretrained(out_path)
print(f"Saving tokenizer to {out_path}")
tokenizer.save_pretrained(out_path)

# Patch config.json: write complete auto_map and optionally fix attn_logits_scaling.
config_json_path = out_path / "config.json"
with open(config_json_path) as f:
    cfg = json.load(f)

config_module = HypeNetConfig.__module__.split(".")[-1]       # "configuration_hypenet"
model_module = HypeNetModel.__module__.split(".")[-1]         # "modeling_hypenet"
causal_module = HypeNetForCausalLM.__module__.split(".")[-1]  # "modeling_hypenet"
cfg["auto_map"] = {
    "AutoConfig": f"{config_module}.{HypeNetConfig.__name__}",
    "AutoModel": f"{model_module}.{HypeNetModel.__name__}",
    "AutoModelForCausalLM": f"{causal_module}.{HypeNetForCausalLM.__name__}",
}
print(f"Patched auto_map: {cfg['auto_map']}")

if args.attn_logits_scaling:
    cfg["attn_logits_scaling"] = args.attn_logits_scaling
    print(f"Patched attn_logits_scaling: {cfg['attn_logits_scaling']}")

with open(config_json_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Done.")
