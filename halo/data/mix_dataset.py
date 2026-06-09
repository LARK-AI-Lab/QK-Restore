from pathlib import Path
from typing import Union, Optional, Dict, List

import numpy as np
import torch
from torch import Tensor
from datasets import load_dataset, load_from_disk, interleave_datasets, IterableDataset
from transformers import AutoTokenizer


def estimate_avg_tokens(
    dataset: IterableDataset,
    tokenizer,
    n_samples: int = 1000,
    text_column: str = "text",
    seed: int = 42,
) -> float:
    """
    Estimate the average token length of documents in *dataset* by sampling
    up to `n_samples` examples and tokenizing them.

    Works with both streaming and non-streaming IterableDatasets by simply
    taking the first `n_samples` items after an optional shuffle buffer.
    """
    try:
        sample_ds = dataset.shuffle(seed=seed, buffer_size=min(n_samples * 10, 10_000))
    except Exception:
        sample_ds = dataset

    total_tokens = 0
    count = 0
    for example in sample_ds:
        if count >= n_samples:
            break
        text = example.get(text_column, "")
        if not text:
            continue
        ids = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        total_tokens += len(ids)
        count += 1

    if count == 0:
        raise RuntimeError("No samples found while estimating average token length.")

    avg = total_tokens / count
    print(f"  Sampled {count} docs → avg tokens = {avg:.1f}")
    return avg


def compute_sampling_probabilities(
    web_ratio: float,
    web_avg_tokens: float,
    math_avg_tokens: float,
):
    """
    Convert a desired token-level ratio into sample-level probabilities,
    accounting for the fact that web and math samples have different
    average token lengths.

    Solves:
        P_web * web_avg_tokens
        ----------------------  = web_ratio
        P_math * math_avg_tokens

    then normalises so P_web + P_math = 1.
    """
    web_weight  = web_ratio       / web_avg_tokens
    math_weight = (1 - web_ratio) / math_avg_tokens
    total = web_weight + math_weight
    return web_weight / total, math_weight / total


def build_dataset(
    tokenizer_name: str,
    # --- data sources ---
    fineweb_dir:   Union[str, Path] = "HuggingFaceFW/fineweb-edu",
    fineweb_name:  str              = "sample-100BT",
    math_dir:      Union[str, Path] = "/path/to/data/ultramath",
    math_text_column: str           = "content",
    # --- mixing ---
    web_token_ratio:  float = 0.8,
    web_avg_tokens:   Optional[float] = None,   # if None, estimated by sampling
    math_avg_tokens:  Optional[float] = None,   # if None, estimated by sampling
    avg_token_estimate_samples: int   = 1000,   # docs to sample for estimation
    seed: int = 42,
    # --- tokenisation / packing ---
    streaming:       bool          = True,
    overwrite_cache: bool          = False,
    token_ids_only:  bool          = True,
    max_len:         int           = 512,
    eos_token_id:    Optional[int] = None,
    shift_labels:    bool          = False,
    return_list:     bool          = False,
    **kwargs,
) -> IterableDataset:
    """
    Build a streaming mixed dataset of FineWeb-Edu and UltraMath.

    Sampling probabilities are computed from `web_token_ratio` and
    the estimated average token lengths of each source, so that the
    resulting token mix matches the requested ratio even though the
    two datasets have very different document lengths.

    Parameters
    ----------
    fineweb_dir : HuggingFace dataset ID or local path for FineWeb-Edu.
    fineweb_name : config name passed to load_dataset (e.g. "sample-100BT").
    math_dir : local path to the UltraMath dataset (load_from_disk).
    math_text_column : column containing raw text in the math dataset.
    web_token_ratio : desired fraction of *tokens* coming from FineWeb (0–1).
    web_avg_tokens / math_avg_tokens : average token length estimates used
        to convert the token ratio to a sample-level probability.  When
        ``None`` (default), they are estimated automatically by tokenising
        ``avg_token_estimate_samples`` randomly sampled documents from each
        source before mixing begins.
    avg_token_estimate_samples : number of documents to sample for the
        automatic estimation (ignored when explicit values are provided).
    """

    print(f"Building tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None

    # ------------------------------------------------------------------
    # 1. Load raw datasets
    # ------------------------------------------------------------------
    print(f"Loading FineWeb-Edu from {fineweb_dir} (name={fineweb_name})...")
    raw_fineweb: IterableDataset = load_dataset(
        str(fineweb_dir),
        name=fineweb_name,
        split="train",
        streaming=streaming,
    )

    print(f"Loading UltraMath from {math_dir}...")
    math_ds = load_from_disk(str(math_dir))
    if streaming:
        math_ds = math_ds.to_iterable_dataset()

    # ------------------------------------------------------------------
    # 2. Normalise both to a single "text" column
    # ------------------------------------------------------------------
    if math_text_column != "text":
        math_ds = math_ds.rename_column(math_text_column, "text")

    # Keep only the text column so interleave sees identical schemas
    raw_fineweb = raw_fineweb.select_columns(["text"])
    math_ds     = math_ds.select_columns(["text"])

    # ------------------------------------------------------------------
    # 3. Estimate average token lengths if not provided
    # ------------------------------------------------------------------
    if web_avg_tokens is None:
        print(f"Estimating FineWeb-Edu avg token length from {avg_token_estimate_samples} samples...")
        web_avg_tokens = estimate_avg_tokens(
            raw_fineweb, tokenizer,
            n_samples=avg_token_estimate_samples, seed=seed,
        )
    if math_avg_tokens is None:
        print(f"Estimating UltraMath avg token length from {avg_token_estimate_samples} samples...")
        math_avg_tokens = estimate_avg_tokens(
            math_ds, tokenizer,
            n_samples=avg_token_estimate_samples, seed=seed,
        )

    # ------------------------------------------------------------------
    # 4. Compute sample-level mixing probabilities
    # ------------------------------------------------------------------
    p_web, p_math = compute_sampling_probabilities(
        web_token_ratio, web_avg_tokens, math_avg_tokens
    )
    print(
        f"Token ratio  : FineWeb {web_token_ratio:.2f} / Math {1 - web_token_ratio:.2f}\n"
        f"Sample probs : FineWeb {p_web:.4f} / Math {p_math:.4f}  "
        f"(web_avg={web_avg_tokens:.0f} tok, math_avg={math_avg_tokens:.0f} tok)"
    )

    # ------------------------------------------------------------------
    # 5. Interleave
    # ------------------------------------------------------------------
    mixed: IterableDataset = interleave_datasets(
        [raw_fineweb, math_ds],
        probabilities=[p_web, p_math],
        seed=seed,
        stopping_strategy="all_exhausted",
    )

    # ------------------------------------------------------------------
    # 6. Tokenise and pack into fixed-length chunks
    # ------------------------------------------------------------------
    chunk_len = max_len + 1 if shift_labels else max_len

    def process_fn(examples: dict) -> Dict[str, Tensor | List]:
        texts: List[str] = examples["text"]

        encodings = tokenizer(
            texts, max_length=10**6, truncation=True, return_tensors="np"
        )

        orig_input_ids: np.ndarray = encodings["input_ids"]
        batch_ids  = [np.append(ids, eos_token_id) for ids in orig_input_ids]
        concat_ids = np.concatenate(batch_ids)

        total_len = (concat_ids.shape[0] // chunk_len) * chunk_len
        n_chunks  = total_len // chunk_len

        if shift_labels:
            input_ids = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
            labels    = np.zeros((n_chunks, chunk_len - 1), dtype=np.int64)
        else:
            input_ids = np.zeros((n_chunks, chunk_len), dtype=np.int64)
            labels    = np.zeros((n_chunks, chunk_len), dtype=np.int64)

        for i in range(n_chunks):
            chunk = concat_ids[i * chunk_len : (i + 1) * chunk_len]
            if shift_labels:
                input_ids[i] = chunk[:-1]
                labels[i]    = chunk[1:]
            else:
                input_ids[i] = chunk
                labels[i]    = chunk

        input_ids_t:  Tensor = torch.from_numpy(input_ids)
        labels_t:     Tensor = torch.from_numpy(labels)
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

    batch_size = kwargs.get("tokenize_batch_size", 1000)

    if streaming:
        tokenized_dataset = mixed.map(
            process_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"] if token_ids_only else [],
        )
    else:
        tokenized_dataset = mixed.map(
            process_fn,
            batched=True,
            batch_size=batch_size,
            remove_columns=["text"] if token_ids_only else [],
            load_from_cache_file=not overwrite_cache,
            desc="Tokenising mixed dataset",
        )

    return tokenized_dataset
