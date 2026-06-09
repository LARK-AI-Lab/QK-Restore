#!/usr/bin/env python3
"""
eval_math_avg16.py

Evaluate a CoT-SFT (or any causal-LM) model on the MATH test set using AVG@16:
  - Generate 16 independent samples per problem
  - Extract the boxed answer from each sample
  - Score each sample as correct/incorrect against ground truth
  - Report AVG@16 = mean pass rate per problem, averaged over all problems

Also reports:
  - pass@1  (accuracy of the first sample only)
  - maj@16  (accuracy of the majority-voted answer)
  - Accuracy breakdowns by level, type, and dataset source
  - Optional: per-problem CoT length stats (--record_lengths)

Data format expected:
  Parquet with columns:
    prompt        : list of message dicts [{role, content}, ...]
    reward_model  : dict with key 'ground_truth'
    level, type, data_source, ability, extra_info  (for breakdowns)

Usage:
  # vLLM backend (recommended for speed)
  python eval_math_avg16.py \\
      --model /path/to/model \\
      --data /path/to/data/math/test.parquet \\
      --n_samples 16 \\
      --backend vllm \\
      --output results/math_avg16.json

  # HuggingFace backend (fallback, slower)
  python eval_math_avg16.py \\
      --model /path/to/model \\
      --data /path/to/data/math/test.parquet \\
      --n_samples 16 \\
      --backend hf \\
      --output results/math_avg16.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Import scoring logic from math_reward.py (the ground-truth reference)
# ─────────────────────────────────────────────────────────────────────────────
# We delegate all extraction and equivalence checking to math_reward.py to
# guarantee identical scoring to the RL training reward.  Specifically:
#   - last_boxed_only_string():  handles \boxed{...}, \boxed ..., and \fbox{...}
#   - remove_boxed():            strips the \boxed wrapper, returning inner content
#   - is_equiv() / strip_string(): full LaTeX normalization (dfrac→frac, \left/\right,
#                                   sqrt3→sqrt{3}, 0.5↔\frac{1}{2}, spaces, units, …)
#
# Using our own simpler normalize_answer() would silently undercount correct answers
# for non-numeric MATH answers (fractions, roots, expressions).

_MATH_REWARD_PATH = "/path/to/reward_func"
if _MATH_REWARD_PATH not in sys.path:
    sys.path.insert(0, _MATH_REWARD_PATH)

from math_reward import (
    compute_score,
    last_boxed_only_string,
    remove_boxed,
    is_equiv,
    strip_string,
)


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the inner content of the last \\boxed{} (or \\boxed  or \\fbox{})
    in a generated response, using math_reward.last_boxed_only_string + remove_boxed.
    Handles nested braces, space-variant, and \\fbox fallback.
    """
    s = last_boxed_only_string(text)
    if s is None:
        return None
    try:
        return remove_boxed(s)
    except Exception:
        return None


def is_correct(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Check correctness using math_reward.is_equiv, which applies strip_string
    normalization (dfrac→frac, spaces, sqrt, fractions, etc.).
    Matches the scoring used during RL training exactly.
    """
    if predicted is None:
        return False
    return is_equiv(predicted, ground_truth)


# ─────────────────────────────────────────────────────────────────────────────
# Majority voting
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(answers: list[Optional[str]]) -> Optional[str]:
    """
    Return the answer that appears most frequently after strip_string normalization.
    Uses the same normalization as is_correct so votes aggregate correctly.
    """
    normalized = []
    for a in answers:
        if a is None:
            normalized.append(None)
        else:
            try:
                normalized.append(strip_string(a))
            except Exception:
                normalized.append(a)
    non_empty = [n for n in normalized if n]
    if not non_empty:
        return None
    counter = Counter(non_empty)
    return counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Generation backends
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_texts(df: pd.DataFrame, tokenizer) -> list[str]:
    """Apply chat template to each row's prompt messages."""
    texts = []
    for _, row in df.iterrows():
        messages = row["prompt"]
        # prompt may be a numpy array of dicts — convert to plain list of dicts
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        # Each element should be a dict with 'role' and 'content'
        messages = [dict(m) for m in messages]
        # Apply chat template — add_generation_prompt=True to elicit response
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)
    return texts


def generate_vllm(
    prompt_texts: list[str],
    model_path: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    dtype: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
) -> list[list[str]]:
    """
    Generate n_samples completions per prompt using vLLM.
    Returns a list of lists: outputs[i][j] = j-th completion for prompt i.
    """
    import torch
    from vllm import LLM, SamplingParams

    available_gpus = torch.cuda.device_count()
    if tensor_parallel_size > available_gpus:
        print(f"[vLLM] WARNING: tensor_parallel_size={tensor_parallel_size} > available GPUs={available_gpus}; clamping to {available_gpus}", flush=True)
        tensor_parallel_size = available_gpus

    print(f"\n[vLLM] Loading model from {model_path} (tensor_parallel_size={tensor_parallel_size}) …", flush=True)
    llm = LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print(f"[vLLM] Generating {n_samples} samples × {len(prompt_texts):,} prompts …", flush=True)
    results = llm.generate(prompt_texts, sampling_params)

    outputs = []
    for result in results:
        completions = [output.text for output in result.outputs]
        outputs.append(completions)
    return outputs


def generate_hf(
    prompt_texts: list[str],
    model_path: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    dtype: str,
    batch_size: int,
) -> list[list[str]]:
    """
    Generate n_samples completions per prompt using HuggingFace transformers.
    Returns a list of lists: outputs[i][j] = j-th completion for prompt i.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"\n[HF] Loading model from {model_path} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="balanced",
        trust_remote_code=True,
    )
    model.eval()

    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = [[] for _ in range(len(prompt_texts))]
    n_total = len(prompt_texts)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_texts = prompt_texts[start:end]
        batch_size_actual = end - start

        print(f"  [HF] Generating batch {start // batch_size + 1} / {(n_total + batch_size - 1) // batch_size} …", flush=True)

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**enc, generation_config=gen_config)

        input_len = enc["input_ids"].shape[1]
        # gen_ids shape: (batch_size_actual * n_samples, seq_len)
        for i in range(batch_size_actual):
            completions = []
            for j in range(n_samples):
                idx = i * n_samples + j
                new_tokens = gen_ids[idx, input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                completions.append(text)
            outputs[start + i] = completions

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_all(
    df: pd.DataFrame,
    all_completions: list[list[str]],
    record_lengths: bool = False,
    start_idx: int = 0,
) -> dict:
    """
    Compute AVG@16, pass@1, maj@16, and breakdowns.
    Returns a results dict.
    """
    n = len(df)
    assert len(all_completions) == n

    per_problem = []
    for i, (_, row) in enumerate(df.iterrows()):
        gt = row["reward_model"]["ground_truth"]
        completions = all_completions[i]
        answers = [extract_boxed_answer(c) for c in completions]

        correct_flags = [is_correct(a, gt) for a in answers]
        pass_rate = float(np.mean(correct_flags))

        maj = majority_vote(answers)
        maj_correct = is_correct(maj, gt)

        record = {
            "idx": start_idx + i,
            "level": row.get("level", ""),
            "type": row.get("type", ""),
            "data_source": row.get("data_source", ""),
            "ground_truth": gt,
            "pass_rate": pass_rate,        # AVG@n
            "pass@1": correct_flags[0],    # first sample only
            "maj@n": maj_correct,          # majority vote
            "extracted_answers": answers,
            "correct_flags": correct_flags,
        }

        if record_lengths:
            lengths = [len(c.split()) for c in completions]  # approx word count
            record["mean_cot_length"] = float(np.mean(lengths))
            record["completions"] = completions  # full text, for downstream analysis

        per_problem.append(record)

    # Aggregate
    avg16     = float(np.mean([r["pass_rate"] for r in per_problem]))
    pass1     = float(np.mean([r["pass@1"]    for r in per_problem]))
    maj_acc   = float(np.mean([r["maj@n"]     for r in per_problem]))
    no_answer = sum(1 for r in per_problem if all(a is None for a in r["extracted_answers"]))

    # Breakdowns
    def breakdown(key):
        groups = defaultdict(list)
        for r in per_problem:
            groups[r[key]].append(r["pass_rate"])
        return {k: float(np.mean(v)) for k, v in sorted(groups.items())}

    results = {
        "summary": {
            "n_problems": n,
            "n_samples_per_problem": len(all_completions[0]),
            "avg@n":   avg16,
            "pass@1":  pass1,
            "maj@n":   maj_acc,
            "no_answer_rate": no_answer / n,
        },
        "by_level":       breakdown("level"),
        "by_type":        breakdown("type"),
        "by_data_source": breakdown("data_source"),
        "per_problem":    per_problem,
    }
    return results


def print_summary(results: dict) -> None:
    s = results["summary"]
    n = s["n_samples_per_problem"]
    print(f"\n{'═' * 60}")
    print(f"  MATH Evaluation Results (AVG@{n})")
    print(f"{'═' * 60}")
    print(f"  {'Problems evaluated':30s}: {s['n_problems']:>8,}")
    print(f"  {'Samples per problem':30s}: {n:>8,}")
    print(f"  {'AVG@' + str(n) + ' (mean pass rate)':30s}: {s['avg@n']:>8.3f}  ({s['avg@n']*100:.1f}%)")
    print(f"  {'pass@1 (greedy equiv.)':30s}: {s['pass@1']:>8.3f}  ({s['pass@1']*100:.1f}%)")
    print(f"  {'maj@' + str(n) + ' (majority vote)':30s}: {s['maj@n']:>8.3f}  ({s['maj@n']*100:.1f}%)")
    print(f"  {'No-answer rate':30s}: {s['no_answer_rate']:>8.3f}  ({s['no_answer_rate']*100:.1f}%)")

    print(f"\n  By level:")
    for lvl, acc in results["by_level"].items():
        bar = "█" * int(acc * 30)
        print(f"    {str(lvl):<20s} | {bar:<30s}  {acc*100:5.1f}%")

    print(f"\n  By type (top 8):")
    by_type = sorted(results["by_type"].items(), key=lambda x: -x[1])
    for typ, acc in by_type[:8]:
        bar = "█" * int(acc * 30)
        print(f"    {str(typ):<20s} | {bar:<30s}  {acc*100:5.1f}%")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a CoT-SFT model on MATH using AVG@N metric."
    )
    p.add_argument("--model",       required=True,  help="Path to model (HF format)")
    p.add_argument("--data",        default="/path/to/data/math/test.parquet",
                   help="Path to MATH test parquet")
    p.add_argument("--output",      default=None,
                   help="Path to save JSON results (default: <model_dir>/math_avg16_results.json)")
    p.add_argument("--n_samples",   type=int, default=16,
                   help="Number of samples to generate per problem (default: 16)")
    p.add_argument("--n_problems",  type=int, default=0,
                   help="Evaluate only first N problems (default: 0 = all)")
    p.add_argument("--start_idx",   type=int, default=0,
                   help="Start index for data-parallel chunk (inclusive, 0-based)")
    p.add_argument("--end_idx",     type=int, default=-1,
                   help="End index for data-parallel chunk (exclusive; -1 = all)")
    p.add_argument("--backend",     default="vllm", choices=["vllm", "hf"],
                   help="Generation backend (default: vllm)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--top_p",       type=float, default=0.95,
                   help="Top-p nucleus sampling (default: 0.95)")
    p.add_argument("--max_new_tokens", type=int, default=8192,
                   help="Max new tokens per sample (default: 2048)")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype (default: bfloat16)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                   help="vLLM GPU memory utilization fraction (default: 0.90)")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="vLLM tensor parallel size (number of GPUs, default: 1)")
    p.add_argument("--batch_size",  type=int, default=8,
                   help="HF backend: batch size (default: 8)")
    p.add_argument("--record_lengths", action="store_true",
                   help="Record per-problem CoT lengths (adds generated text to output)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──
    print(f"\nLoading data from {args.data} …", flush=True)
    df = pd.read_parquet(args.data)
    print(f"  Loaded {len(df):,} problems")

    if args.n_problems > 0:
        df = df.head(args.n_problems)
        print(f"  Using first {len(df):,} problems (--n_problems)")

    if args.start_idx > 0 or args.end_idx >= 0:
        end = args.end_idx if args.end_idx >= 0 else len(df)
        df = df.iloc[args.start_idx:end].reset_index(drop=True)
        print(f"  Using problems [{args.start_idx}, {end}) — {len(df):,} problems (data-parallel chunk)")

    # ── Load tokenizer (for chat template) ──
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer from {args.model} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Build prompt texts ──
    print("Applying chat template …", flush=True)
    prompt_texts = build_prompt_texts(df, tokenizer)
    print(f"  Example prompt (first 300 chars):\n  {prompt_texts[0][:300]!r}")

    # ── Generate ──
    if args.backend == "vllm":
        all_completions = generate_vllm(
            prompt_texts=prompt_texts,
            model_path=args.model,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:  # hf
        all_completions = generate_hf(
            prompt_texts=prompt_texts,
            model_path=args.model,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            dtype=args.dtype,
            batch_size=args.batch_size,
        )

    # ── Score ──
    print("\nScoring …", flush=True)
    results = score_all(df, all_completions, record_lengths=args.record_lengths, start_idx=args.start_idx)
    results["config"] = vars(args)

    print_summary(results)

    # ── Save ──
    if args.output is None:
        model_dir = Path(args.model)
        args.output = str(model_dir / f"math_avg{args.n_samples}_results.json")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Don't serialize full completions by default (large) unless record_lengths requested
    if not args.record_lengths:
        for r in results["per_problem"]:
            r.pop("completions", None)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
