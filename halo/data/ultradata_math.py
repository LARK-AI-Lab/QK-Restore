"""
Streaming data loader for openbmb/UltraData-Math.

Loads two configs:
  - UltraData-Math-L3-Conversation-Synthetic
  - UltraData-Math-L3-QA-Synthetic

Both have a single `content` column with plain text.
They are interleaved with equal probability (configurable),
then tokenized and packed into fixed-length chunks exactly
like fineweb_edu_100bt.py.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from datasets import interleave_datasets, load_dataset, IterableDataset
from transformers import AutoTokenizer

CONFIGS = [
    "UltraData-Math-L3-Conversation-Synthetic",
    "UltraData-Math-L3-QA-Synthetic",
]


def build_dataset(
    tokenizer_name: str,
    data_path: Union[str, Path] = "openbmb/UltraData-Math",
    # mixing: equal weight by default
    config_weights: Optional[List[float]] = None,
    seed: int = 42,
    streaming: bool = True,
    # tokenisation / packing
    max_len: int = 4096,
    eos_token_id: Optional[int] = None,
    shift_labels: bool = False,
    token_ids_only: bool = True,
    return_list: bool = False,
    **kwargs,
) -> IterableDataset:
    """
    Build a streaming dataset from UltraData-Math (Conversation + QA configs).

    Parameters
    ----------
    data_path       : HuggingFace dataset repo ID (default: openbmb/UltraData-Math).
    config_weights  : per-config sampling probabilities; defaults to equal weights.
    max_len         : packed sequence length (tokens).
    shift_labels    : if True, input_ids = chunk[:-1], labels = chunk[1:].
    """
    print(f"Building tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None

    if config_weights is None:
        config_weights = [1.0 / len(CONFIGS)] * len(CONFIGS)
    assert len(config_weights) == len(CONFIGS), \
        f"Expected {len(CONFIGS)} weights, got {len(config_weights)}"

    # ------------------------------------------------------------------
    # 1. Load each config
    # ------------------------------------------------------------------
    datasets = []
    for cfg in CONFIGS:
        print(f"Loading {data_path} / {cfg} ...")
        ds = load_dataset(
            str(data_path),
            name=cfg,
            split="train",
            streaming=streaming,
        )
        # Both configs expose `content` as plain text; drop `uid`.
        ds = ds.select_columns(["content"])
        ds = ds.rename_column("content", "text")
        datasets.append(ds)

    # ------------------------------------------------------------------
    # 2. Interleave
    # ------------------------------------------------------------------
    mixed: IterableDataset = interleave_datasets(
        datasets,
        probabilities=config_weights,
        seed=seed,
        stopping_strategy="first_exhausted",
    )

    # ------------------------------------------------------------------
    # 3. Tokenise and pack into fixed-length chunks
    # ------------------------------------------------------------------
    chunk_len = max_len + 1 if shift_labels else max_len

    def process_fn(examples: dict) -> Dict[str, Tensor]:
        texts: List[str] = examples["text"]
        encodings = tokenizer(
            texts, max_length=10 ** 6, truncation=True, return_tensors="np"
        )

        orig_ids: np.ndarray = encodings["input_ids"]
        batch_ids = [np.append(ids, eos_token_id) for ids in orig_ids]
        concat_ids = np.concatenate(batch_ids)

        total_len = (concat_ids.shape[0] // chunk_len) * chunk_len
        n_chunks = total_len // chunk_len

        if shift_labels:
            input_ids_np = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
            labels_np    = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
        else:
            input_ids_np = np.zeros((n_chunks, chunk_len), dtype=np.int64)
            labels_np    = np.zeros((n_chunks, chunk_len), dtype=np.int64)

        for i in range(n_chunks):
            chunk = concat_ids[i * chunk_len: (i + 1) * chunk_len]
            if shift_labels:
                input_ids_np[i] = chunk[:-1]
                labels_np[i]    = chunk[1:]
            else:
                input_ids_np[i] = chunk
                labels_np[i]    = chunk

        input_ids_t:  Tensor = torch.from_numpy(input_ids_np)
        labels_t:     Tensor = torch.from_numpy(labels_np)
        position_ids: Tensor = torch.arange(input_ids_t.shape[1]).repeat(n_chunks, 1)

        if return_list:
            return {
                "input_ids":    input_ids_t.tolist(),
                "labels":       labels_t.tolist(),
                "position_ids": position_ids.tolist(),
            }
        return {
            "input_ids":    input_ids_t,
            "labels":       labels_t,
            "position_ids": position_ids,
        }

    tokenize_batch_size = kwargs.get("tokenize_batch_size", 1000)

    tokenized = mixed.map(
        process_fn,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=["text"] if token_ids_only else [],
    )
    return tokenized
