"""
layer_analysis.py — Layer importance scoring for hybrid-attention distillation.

Implements the HALO metric: for each layer l, score = (rt_max - rt_l) / (cs_max - cs_l + ε),
capped at 50.  Higher score = more important to keep as full attention.

Usage
-----
python layer_analysis.py --root <root> --model <model>

Expected directory layout
-------------------------
<root>/
  <model>/
    layer0/
      commonsense_eval.log    # lm_eval output: hellaswag, arc_easy, arc_challenge
      retrieval_eval.log      # lm_eval output: ruler_qa_squad, fda, swde
    layer1/ ...
"""

import os
import re
import json
import argparse
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

PATTERN = re.compile(
    r"\|(?P<task>[^|]+)\s*\|"
    r"\s*(?P<version>\d*)\s*\|"
    r"[^|]*\|\s*\d+\s*\|"
    r"(?P<context>[\w\d]*)\s*\|"
    r"↑\s*\|\s*(?P<value>[-+]?\d+(?:\.\d+)?)"
)


def extract_scores(path, skip_first=False):
    """Parse an lm-evaluation-harness log and return {task: [(ctx, value)]}."""
    results = defaultdict(list)
    current_task = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            m = PATTERN.search(line)
            if not m:
                continue

            task = m.group("task").strip()
            if task:
                current_task = task
            elif current_task is None:
                continue

            try:
                ctx = int(m.group("context"))
            except ValueError:
                ctx = 0

            val = float(m.group("value"))
            if val < 0:
                continue

            results[current_task].append((ctx, val))

    for t in results:
        results[t].sort(key=lambda x: x[0])
        if skip_first:
            results[t] = results[t][1:]
        if len(results[t]) == 0:
            raise RuntimeError(f"Empty task '{t}' in {path}")

    if not results:
        raise RuntimeError(f"No valid results in {path}")

    return results


def average_tasks(task_scores):
    """Flatten all (ctx, val) pairs and return the grand mean."""
    total, count = 0.0, 0
    for scores in task_scores.values():
        vals = [v for _, v in scores]
        total += sum(vals)
        count += len(vals)
    if count == 0:
        raise RuntimeError("No values to average")
    return total / count


# ---------------------------------------------------------------------------
# Layer collection
# ---------------------------------------------------------------------------

LAYER_RE = re.compile(r"^layer(\d+)$")


def _layer_dirs(model_dir):
    """Return sorted list of (layer_index, layer_dir_path)."""
    entries = []
    for d in os.listdir(model_dir):
        m = LAYER_RE.match(d)
        if m:
            entries.append((int(m.group(1)), os.path.join(model_dir, d)))
    if not entries:
        raise RuntimeError(f"No layer* directories found in {model_dir}")
    return sorted(entries)


def collect_layers(model_dir, filename, skip_first=False):
    """
    Collect a scalar score per layer from <model_dir>/layer<N>/<filename>.
    Returns {"contexts": [layer_idx, ...], "values": [score, ...]}.
    """
    values = {}
    for layer_idx, layer_path in _layer_dirs(model_dir):
        path = os.path.join(layer_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        scores = extract_scores(path, skip_first)
        values[layer_idx] = average_tasks(scores)

    layers = sorted(values)
    return {
        "contexts": layers,
        "values": [values[l] for l in layers],
    }


# ---------------------------------------------------------------------------
# HALO importance metric
# ---------------------------------------------------------------------------

def compute_halo(cs, rt):
    """
    HALO metric:  score(l) = (rt_max - rt_l) / (cs_max - cs_l + ε),  capped at 50

    cs_max / rt_max are the observed maxima across the linearized layer sweep.
    Higher score = this layer matters more for retrieval relative to commonsense,
    i.e. more important to keep as full attention.
    """
    cs = np.array(cs, dtype=float)
    rt = np.array(rt, dtype=float)
    eps = 1e-10
    out = (rt.max() - rt) / (cs.max() - cs + eps)
    return np.minimum(50.0, out)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_importance(layers, scores, model, out_path):
    plt.figure(figsize=(10, 4))
    plt.bar(layers, scores, color="steelblue", edgecolor="black", linewidth=0.5)
    plt.xlabel("Layer index")
    plt.ylabel("HALO importance score")
    plt.title(f"{model} — HALO layer importance")
    plt.xticks(layers, [str(l) for l in layers], fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    model_dir = os.path.join(args.root, args.model)

    cs_data = collect_layers(model_dir, "commonsense_eval.log", skip_first=True)
    rt_data = collect_layers(model_dir, "retrieval_eval.log",   skip_first=False)

    layers = cs_data["contexts"]
    cs_vals = np.array(cs_data["values"], dtype=float)
    rt_vals = np.array(rt_data["values"], dtype=float)

    scores = compute_halo(cs_vals, rt_vals)

    # Save per-layer averages
    cs_json = os.path.join(args.root, f"{args.model}_commonsense_avg.json")
    rt_json = os.path.join(args.root, f"{args.model}_retrieval_avg.json")
    with open(cs_json, "w") as f:
        json.dump(cs_data, f, indent=4)
    with open(rt_json, "w") as f:
        json.dump(rt_data, f, indent=4)

    # Rank and print
    order = np.argsort(scores)[::-1]

    print(f"\n{'='*55}")
    print(f"  HALO Layer Importance — {args.model}")
    print(f"{'='*55}")
    print(f"{'Rank':>4} | {'Layer':>5} | {'CS avg':>7} | {'RT avg':>7} | Score")
    print(f"{'----':>4}-+-{'-----':>5}-+-{'-------':>7}-+-{'-------':>7}-+--------")
    for rank, idx in enumerate(order):
        print(
            f"{rank+1:4d} | {layers[idx]:5d} | {cs_vals[idx]:7.4f} | "
            f"{rt_vals[idx]:7.4f} | {scores[idx]:.4f}"
        )

    os.makedirs("figures", exist_ok=True)
    fig_path = f"figures/{args.model}_halo_importance.png"
    plot_importance(layers, scores, args.model, fig_path)

    print(f"\nLayer averages → {cs_json}, {rt_json}")
    print(f"Importance plot → {fig_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="HALO layer importance scoring for hybrid-attention distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root",  required=True,
                    help="Root directory containing <model>/layer<N>/ subdirs")
    ap.add_argument("--model", required=True,
                    help="Model subdirectory name (e.g. hypenet-2b_lightning-attn)")
    main(ap.parse_args())
