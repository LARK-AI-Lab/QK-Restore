from pathlib import Path
import json
from typing import Tuple
from datetime import datetime

import torch
from accelerate import Accelerator
from torch import nn, Tensor
from torch.utils.data import DataLoader
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import DataLoaderConfiguration
from torch.optim import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from optim.lr_scheduler import WSDScheduler

from arguments import Args
from data import get_data
import sys


def load_train_config(args: Args):
    """
    Load from `args.train_config` if it exists, and use the values
    when the argument is not provided.
    """
    if args.train_config is not None and Path(args.train_config).exists():
        print(f"Loading train config from {args.train_config}")
        config = json.load(open(args.train_config, "r"))
        for k, v in config.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    else:
        print(f"WARNING: train config {args.train_config} does not exist.")
        print("It is highly recommended to provide a train config.")


def get_accelerator(args: Args, find_unused_parameters: bool = False) -> Accelerator:
    assert args.grad_accum_steps is not None
    assert args.run_name is not None
    assert args.proj_name is not None

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters,
    )

    dataloader_config = DataLoaderConfiguration(
        use_stateful_dataloader=False,
    )

    project_dir = f"./results/{args.proj_name}"
    log_with = args.report_to.split(",")
    accelerator = Accelerator(
        project_dir=project_dir,
        log_with=log_with,  # type: ignore
        gradient_accumulation_steps=args.grad_accum_steps,
        # This argument allows us to step the LR scheduler manually.
        # This is needed because internally, accelerator expects the
        # LR scheduler to step more frequently when using multiple
        # processes or when using gradient accumulation (e.g.,
        # the warmup steps should be scaled by
        # num_processes * gradient_accumulation).
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs] if find_unused_parameters else [],
    )
    accelerator.print(f"Project directory: {project_dir}")
    accelerator.print(f"Reporting to: {log_with}")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    hps = args.as_dict()
    run_name = f"{args.run_name}_{formatted_time}"
    accelerator.init_trackers(
        project_name=args.proj_name,
        config=hps,
        init_kwargs={
            "wandb": {
                "name": run_name,
            },
            "swanlab": {
                "experiment_name": run_name,
            },
        },
    )
    return accelerator

def default_collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """
    Default collate function that handles both tensor and list inputs.
    - If input is 1D (list or tensor): convert to tensor and stack
    - If input is 2D tensor (pre-batched): convert to tensor and concatenate
    """
    input_ids_list = []
    labels_list = []
    position_ids_list = []
    is_prebatched = False
    
    for i, example in enumerate(batch):
        # Convert lists to tensors if needed
        input_ids = example["input_ids"]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = input_ids.to(dtype=torch.long)
        
       
        # Check if data is pre-batched (2D tensors from fineweb_edu_100bt.py)
        if i == 0:
            is_prebatched = (input_ids.ndim == 2)
        
        input_ids_list.append(input_ids)
        
        labels = example["labels"]
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.to(dtype=torch.long)
        labels_list.append(labels)
        
        if "position_ids" in example:
            position_ids = example["position_ids"]
            if isinstance(position_ids, list):
                position_ids = torch.tensor(position_ids, dtype=torch.long)
            else:
                position_ids = position_ids.to(dtype=torch.long)
            position_ids_list.append(position_ids)
    
    # Combine based on data format
    if is_prebatched:
        # Pre-batched data: concatenate along first dimension
        input_ids = torch.cat(input_ids_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
    else:
        # Individual examples: stack along new batch dimension
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)
    
    result = {
        "input_ids": input_ids,
        "labels": labels,
    }
    
    if position_ids_list:
        if is_prebatched:
            result["position_ids"] = torch.cat(position_ids_list, dim=0)
        else:
            result["position_ids"] = torch.stack(position_ids_list)
    
    return result

def cu_seqlens_collate_fn(
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    seqlens = torch.tensor([it["input_ids"].size(0) for it in batch], dtype=torch.int32)

    cu = torch.empty(seqlens.numel() + 1, dtype=torch.int32)
    cu[0] = 0
    torch.cumsum(seqlens, dim=0, out=cu[1:])

    input_ids: Tensor = torch.cat([it["input_ids"] for it in batch], dim=0).to(
        dtype=torch.long
    )
    labels: Tensor = torch.cat([it["labels"] for it in batch], dim=0).to(
        dtype=torch.long
    )

    if input_ids.numel() == 0:
        raise ValueError("input_ids is empty after concatenation, unexpected.")

    return {
        "input_ids": input_ids.view(1, -1),  # (1, total_tokens)
        "labels": labels.view(1, -1),  # (1, total_tokens)
        "cu_seqlens": cu.view(1, -1),  # (1, B + 1)
    }


def get_dataloaders(
    args: Args, tok_path: str, accelerator: Accelerator
) -> Tuple[DataLoader, DataLoader | None]:
    assert args.data_name is not None
    assert args.data_path is not None
    assert args.max_len is not None
    assert args.batch_size is not None

    shift_labels_in_dataset = not bool(args.shift_labels_in_model)
    max_len = args.max_len

    train_ds = get_data(
        tokenizer_name=tok_path,
        data_name=args.data_name,
        data_path=args.data_path,
        max_len=max_len,
        is_main_process=accelerator.is_main_process,
        shift_labels=shift_labels_in_dataset,
        is_seq2seq=bool(args.is_seq2seq),
        stage=args.stage,
        **({} if args.math_dir is None else {"math_dir": args.math_dir}),
        **({} if args.web_token_ratio is None else {"web_token_ratio": args.web_token_ratio}),
    )  # type: ignore



    if args.use_cu_seqlens:
        collate_fn = cu_seqlens_collate_fn
    else:
        collate_fn = None # default_collate_fn

    train_loader = DataLoader(
        train_ds,  # type: ignore
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1 if args.data_n_workers is None else args.data_n_workers,
        collate_fn=collate_fn,
    )


    if args.validation_data_path is not None and args.validation_data_name is not None:
        val_ds = get_data(
            tokenizer_name=tok_path,
            data_name=args.validation_data_name,
            data_path=args.validation_data_path,
            max_len=max_len,
            is_main_process=accelerator.is_main_process,
            shift_labels=shift_labels_in_dataset,
            is_seq2seq=bool(args.is_seq2seq),
            stage=args.stage,
        )  # type: ignore
        if args.validation_data_size is not None:
            val_ds = val_ds.take(args.validation_data_size)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, collate_fn=collate_fn  # type: ignore
        )
    else:
        val_loader = None

    return train_loader, val_loader


def get_args() -> Args:
    args = Args().parse_args()

    if args.compile == 1:
        if args.grad_ckpt == 1:
            print(
                "Cannot use grad checkpoint and compile mode together, setting compile to 0."
            )
            args.compile = 0

        if args.model_name in ["gated_deltanet", "gated-deltanet"]:
            print(
                "Gated DeltaNet does not support torch.compile, setting compile to 0..."
            )
            args.compile = 0

        if args.model_name in ["rabbit"]:
            print("Rabbit does not support compilation, turning it off...")
            args.compile = 0

        if args.model_name in ["mamba2"]:
            print("Turning off compilation for Mamba2")
            args.compile = 0

    return args


def prepare_optimizers(model: nn.Module, args: Args, accelerator: Accelerator):
    """
    Returns: (optimizer, lr_scheduler)
    """
    assert args.lr is not None
    assert args.beta1 is not None
    assert args.beta2 is not None
    assert args.weight_decay is not None
    assert args.n_warmup_steps is not None
    assert args.n_train_steps is not None
    assert args.lr_scheduler is not None

    decay = set()
    no_decay = set()
    for name, param in model.named_parameters():
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.add(name)
        else:
            decay.add(name)

    trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen_params = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    accelerator.print(f"Trainable parameters: {len(trainable_params)}")
    accelerator.print(f"Trainable parameters: {list(trainable_params.keys())}")

    # Frozen params must be excluded from the optimizer entirely.
    # DeepSpeed ZeRO Stage 1/2 flattens gradients per group; passing
    # requires_grad=False params produces an empty tensor list and crashes.
    param_groups = [
        {
            "params": [p for n, p in trainable_params.items() if n in decay],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in trainable_params.items() if n in no_decay],
            "lr": args.lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    if args.lr_scheduler == "wsd":
        assert args.n_drop_steps is not None
        lr_scheduler = WSDScheduler(
            optimizer=optimizer,
            lr=args.lr,
            n_decay_iters=args.n_drop_steps,
            n_warmup_iters=args.n_warmup_steps,
            n_train_iters=args.n_train_steps,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.n_warmup_steps,
            num_training_steps=args.n_train_steps,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Invalid LR scheduler: {args.lr_scheduler}")
    return optimizer, lr_scheduler
