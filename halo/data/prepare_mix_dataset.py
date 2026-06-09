#!/usr/bin/env python3
"""
prepare_mixed_dataset.py

This script prepares a mixed pretraining dataset by combining:

    1. FineWeb-Edu (general web corpus)
    2. A math / reasoning dataset

IMPORTANT DESIGN CHOICE
-----------------------
This script intentionally DOES NOT tokenize the data.

Tokenization and sequence packing are performed later inside
the training pipeline (the `build_dataset()` function).

Reasons:
    • Avoid duplicating tokenization work
    • Keep the saved dataset small
    • Allow changing tokenizers without rebuilding the dataset
    • Simplify preprocessing

Therefore this script only mixes RAW TEXT samples.

Because tokenization is not performed here, we cannot know the
exact token counts of each dataset. Instead we estimate the
expected token ratio using the average tokens per sample.

This produces a good approximation of token-balanced mixing.

Pipeline overview:

    FineWeb dataset
           │
           ├─ normalize column → "text"
           │
    Math dataset
           │
           ├─ rename "content" → "text"
           │
           ▼
     Estimate average tokens/sample
           ▼
     Compute sampling probabilities
           ▼
     Interleave datasets
           ▼
     Save mixed dataset

The resulting dataset contains only:

    {"text": "...raw text..."}

Later, the training pipeline will perform:

    tokenize → concatenate → pack into sequences

Usage example:

    python prepare_mixed_dataset.py \
        --fineweb_dir data/fineweb_edu \
        --math_dir data/math_dataset \
        --output_dir data/pretrain_mixed \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --web_token_ratio 0.8
"""

import argparse
from pathlib import Path
import numpy as np

from datasets import load_from_disk, interleave_datasets
from transformers import AutoTokenizer


def estimate_avg_tokens(dataset, tokenizer, text_key="text", n_samples=2000):
    """
    Estimate average tokens per sample for a dataset.

    Only a small subset of samples is tokenized to avoid
    expensive full-dataset tokenization.

    Parameters
    ----------
    dataset : HuggingFace Dataset
    tokenizer : tokenizer used for estimation
    text_key : column containing raw text
    n_samples : number of samples used for estimation

    Returns
    -------
    float
        estimated average tokens per sample
    """

    n = min(len(dataset), n_samples)

    subset = dataset.select(range(n))
    texts = subset[text_key]

    # sanitize inputs
    clean_texts = []
    for t in texts:
        if t is None:
            continue
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if len(t) == 0:
            continue
        clean_texts.append(t)

    if len(clean_texts) == 0:
        raise ValueError("No valid text samples found for token estimation")

    enc = tokenizer(
        clean_texts,
        truncation=False,
        return_attention_mask=False,
    )

    lengths = [len(x) for x in enc["input_ids"]]

    return float(np.mean(lengths))


def compute_sampling_probabilities(
    web_ratio,
    web_avg_tokens,
    math_avg_tokens,
):
    """
    Convert desired token ratio → sample probabilities.

    Suppose:

        desired_token_ratio = 0.8 web / 0.2 math

    But each dataset has different average lengths:

        web_avg_tokens  ≈ 900
        math_avg_tokens ≈ 200

    If we sampled with probabilities [0.8, 0.2], the true
    token ratio would become highly skewed.

    Instead we solve:

        P_web  * web_avg_tokens
        -----------------------  = web_ratio
        P_math * math_avg_tokens

    Then normalize probabilities.
    """

    web_weight = web_ratio / web_avg_tokens
    math_weight = (1 - web_ratio) / math_avg_tokens

    total = web_weight + math_weight

    p_web = web_weight / total
    p_math = math_weight / total

    return p_web, p_math


def prepare_mixed_dataset(
    fineweb_dir,
    math_dir,
    output_dir,
    tokenizer_name,
    web_token_ratio=0.8,
):
    """
    Main dataset preparation function.
    """

    print("Loading datasets...")

    fineweb = load_from_disk(fineweb_dir)
    math = load_from_disk(math_dir)

    print("Normalizing dataset schema...")

    # FineWeb already contains "text"
    if "text" not in fineweb.column_names:
        raise ValueError("FineWeb dataset must contain 'text' column")

    # math dataset uses "content"
    if "content" not in math.column_names:
        raise ValueError("Math dataset must contain 'content' column")

    math = math.rename_column("content", "text")

    # keep only raw text
    fineweb = fineweb.select_columns(["text"])
    math = math.select_columns(["text"])

    print("Building tokenizer for token-length estimation...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Estimating average token length...")

    web_avg_tokens = estimate_avg_tokens(fineweb, tokenizer)
    math_avg_tokens = estimate_avg_tokens(math, tokenizer)

    print(f"Estimated tokens/sample:")
    print(f"  FineWeb : {web_avg_tokens:.2f}")
    print(f"  Math    : {math_avg_tokens:.2f}")

    p_web, p_math = compute_sampling_probabilities(
        web_token_ratio,
        web_avg_tokens,
        math_avg_tokens,
    )

    print("Sampling probabilities (approx token-balanced):")
    print(f"  FineWeb : {p_web:.4f}")
    print(f"  Math    : {p_math:.4f}")

    print("Mixing datasets using probabilistic interleaving...")

    mixed = interleave_datasets(
        [fineweb, math],
        probabilities=[p_web, p_math],
        seed=42,
        stopping_strategy="first_exhausted",
    )

    print("Saving mixed dataset...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(output_dir)

    print("Finished.")
    print(f"Total samples: {len(mixed)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--fineweb_dir", required=True)
    parser.add_argument("--math_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer used for estimating token lengths",
    )

    parser.add_argument(
        "--web_token_ratio",
        type=float,
        default=0.8,
        help="Desired fraction of tokens from web corpus",
    )

    args = parser.parse_args()

    prepare_mixed_dataset(
        fineweb_dir=args.fineweb_dir,
        math_dir=args.math_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        web_token_ratio=args.web_token_ratio,
    )