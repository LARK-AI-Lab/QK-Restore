"""
CoT-SFT Stage for supervised fine-tuning with chain-of-thought reasoning.
Follows the same pattern as stage1/2/3 training pipeline.
"""
from pathlib import Path
from accelerate.utils import set_seed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk, load_dataset, concatenate_datasets
import json

from trainer.trainer1 import LMTrainer
from modeling.hypenet import HypeNetForCausalLM, HypeNetConfig
from modeling.hypenet import build_hybrid_from_ckpt
from preparation import (
    get_args,
    get_accelerator,
    prepare_optimizers,
    load_train_config,
)


INSTRUCTION_FOLLOWING = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def build_messages(example):
    """Convert problem and answer to chat format."""
    user_prompt = (
        example["problem"]
        + "\n\n"
        + INSTRUCTION_FOLLOWING
    )
    
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["answer"]}
    ]
    
    return {"messages": messages}


def apply_template(example, tokenizer):
    """Apply chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def tokenize_fn(example, tokenizer, max_length):
    """Tokenize text with prompt masking for loss computation."""
    user_prompt = example["messages"][0]["content"]
    
    prompt_messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    full_text = example["text"]
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length
    )
    
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    
    # Mask prompt tokens in labels (set to -100 for loss masking)
    labels = tokenized["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    
    tokenized["labels"] = labels
    
    return tokenized


def collate_fn(batch):
    """Collate batch with padding - includes position_ids for trainer compatibility."""
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )
    
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )
    
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask,
        batch_first=True,
        padding_value=0
    )
    
    # Create position_ids: [0, 1, 2, ..., seq_len-1] for each sequence
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def prepare_cot_dataset(dataset, tokenizer, max_length, num_proc=4):
    """Prepare CoT dataset following preparation.py patterns."""
    # Build messages format
    dataset = dataset.map(
        build_messages,
        num_proc=num_proc,
        desc="Building messages"
    )
    
    # Apply chat template
    dataset = dataset.map(
        lambda x: apply_template(x, tokenizer),
        num_proc=num_proc,
        desc="Applying chat template"
    )
    
    # Tokenize with prompt masking
    dataset = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, max_length),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return dataset


def _load_mixed_cot_dataset(args, accelerator):
    """Load and mix MiroMind (local disk) + OpenR1-Math-220k (HuggingFace)."""

    # --- MiroMind ---
    accelerator.print(f"Loading MiroMind from {args.data_path} ...")
    miromind = load_from_disk(args.data_path)
    if isinstance(miromind, dict):
        miromind = miromind.get("train") or miromind.get("training")
    if miromind is None:
        raise ValueError(f"No train split found in {args.data_path}")
    miromind = miromind.select_columns(["problem", "answer"])
    accelerator.print(f"  MiroMind: {len(miromind):,} examples")

    # --- OpenR1-Math-220k ---
    # `solution` is the full CoT chain; `answer` is just the boxed final answer.
    openr1_hf = "open-r1/OpenR1-Math-220k"
    accelerator.print(f"Loading OpenR1-Math-220k from {openr1_hf} ...")
    openr1 = load_dataset(openr1_hf, split="train")
    openr1 = openr1.select_columns(["problem", "solution"])
    openr1 = openr1.rename_column("solution", "answer")
    accelerator.print(f"  OpenR1-Math-220k: {len(openr1):,} examples")

    # --- Mix & shuffle ---
    combined = concatenate_datasets([miromind, openr1])
    combined = combined.shuffle(seed=42)
    accelerator.print(f"  Combined total: {len(combined):,} examples")
    return combined


def get_cot_dataloaders(args, tokenizer, accelerator):
    """
    Load and prepare CoT dataloaders.
    Follows get_dataloaders pattern from preparation.py
    """
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None

    if args.data_name == "mixed-cot":
        train_ds = _load_mixed_cot_dataset(args, accelerator)
    else:
        accelerator.print(f"Loading dataset from {args.data_path}")
        dataset = load_from_disk(args.data_path)
        if isinstance(dataset, dict):
            train_ds = dataset.get("train", dataset.get("training"))
        else:
            train_ds = dataset
        if train_ds is None:
            raise ValueError(f"No 'train' split found in dataset from {args.data_path}")

    accelerator.print(f"Preparing training dataset ({len(train_ds):,} examples)...")
    train_ds = prepare_cot_dataset(train_ds, tokenizer, args.max_len)
    
    # Create training dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    
    # Prepare validation dataloader if available
    val_loader = None
    if args.validation_data_path is not None:
        accelerator.print(f"Loading validation dataset from {args.validation_data_path}")
        val_dataset = load_from_disk(args.validation_data_path)
        
        if isinstance(val_dataset, dict):
            val_ds = val_dataset.get("validation", val_dataset.get("val"))
        else:
            val_ds = val_dataset
        
        if val_ds is not None:
            accelerator.print(f"Preparing validation dataset...")
            val_ds = prepare_cot_dataset(val_ds, tokenizer, args.max_len)
            
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )
    
    return train_loader, val_loader



def main():
    # Load configuration following stage3 pattern
    args = get_args()
    load_train_config(args)
    set_seed(args.seed)
    
    accelerator = get_accelerator(args)
    
    accelerator.print("================ args ================")
    accelerator.print(args)
    accelerator.print("======================================")
    
    # Make output dir and dump args (same as stage3)
    output_dir = Path(args.output_dir, args.proj_name, args.run_name)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))
    
    # Calculate effective batch size (same as stage3)
    tokens_per_iter = (
        accelerator.num_processes
        * args.grad_accum_steps
        * args.batch_size
        * args.max_len
    )
    accelerator.print(f"Tokens per batch: {tokens_per_iter:,}")
    accelerator.print(f"Process: {accelerator.num_processes}")
    accelerator.print(f"Grad accum: {args.grad_accum_steps}")
    accelerator.print(f"Batch size: {args.batch_size}")
    accelerator.print(f"Max len: {args.max_len:,}")
    
    # Load tokenizer
    accelerator.print(f"Loading tokenizer from {args.init_from}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tok_path,
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (support multiple init methods like stage3)
    accelerator.print("Building model...")
    
    if args.init_method == 'build_from_stage3':
        # Load HyperNet from stage3 checkpoint
        model = build_hybrid_from_ckpt(args, accelerator, args.init_from)
    elif args.init_method == 'HypeNetForCausalLM.from_pretrained':
        assert args.init_from is not None
        model = HypeNetForCausalLM.from_pretrained(args.init_from)
    elif args.init_method == 'state_dict':
        assert args.model_config is not None
        assert args.init_from is not None
        config = HypeNetConfig.from_json_file(Path(args.model_config))
        model = HypeNetForCausalLM(config=config)
        accelerator.print("Loading state dict")
        state_dict = torch.load(args.init_from)
        accelerator.print("Loading state dict into model")
        model.load_state_dict(state_dict)
    elif args.init_method == 'AutoModelForCausalLM.from_pretrained':
        model = AutoModelForCausalLM.from_pretrained(
            args.init_from,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Unknown init_method: {args.init_method}")
    
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    
    model = model.to(torch.bfloat16)
    
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model parameters: {n_params:,}")
    
    accelerator.print("================ model ================")
    accelerator.print(model.config)
    accelerator.print("=======================================")
    
    # Save model config (same as stage3)
    if accelerator.is_main_process:
        config_path = output_dir / "config.json"
        accelerator.print(f"Saving model config to {config_path}")
        model.config.to_json_file(config_path)
    
    # Prepare optimizers (same as stage3)
    accelerator.print("Preparing optimizers...")
    optimizer, lr_scheduler = prepare_optimizers(
        model=model,
        args=args,
        accelerator=accelerator,
    )
    
    # Compile model if requested (same as stage3)
    if bool(args.compile):
        assert args.device != "mps", "torch.compile not supported on MPS"
        accelerator.print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Prepare dataloaders using new function following preparation.py patterns
    accelerator.print("Preparing dataloaders...")
    train_loader, val_loader = get_cot_dataloaders(args, tokenizer, accelerator)
    
    # Prepare trainer (same as stage3)
    accelerator.print("Preparing LMTrainer for CoT-SFT...")
    trainer = LMTrainer(
        args=args,
        output_dir=output_dir,
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    accelerator.print("===== Start CoT-SFT training =====")
    trainer.train()
    accelerator.print("===== Done CoT-SFT training =====")


if __name__ == "__main__":
    main()
