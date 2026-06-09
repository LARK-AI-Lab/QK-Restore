#!/bin/bash
# eval_gsm_avg16-multigpus.sh
#
# Evaluate a model on GSM8K using AVG@16 with optional multi-GPU data parallelism.
# Each GPU processes a disjoint subset of problems; results are merged at the end.
#
# Usage:
#   bash eval_gsm_avg16-multigpus.sh model=/path/to/model
#   bash eval_gsm_avg16-multigpus.sh model=/path/to/model n_gpus=4
#   bash eval_gsm_avg16-multigpus.sh model=/path/to/model n_gpus=1   # single GPU

set -euo pipefail
for arg in "$@"; do eval "$arg"; done

export TORCHDYNAMO_VERBOSE=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ── Defaults ──────────────────────────────────────────────────────────────────
model=${model:-"/path/to/results/hypenet_CoT_SFT/ckpt_100"}
model_root=${model_root:-""}
model_variants=${model_variants:-"qkrr_all konly_all vonly_all qkrr_procrustes"}

data=${data:-"/path/to/data/gsm8k/test.parquet"}
n_samples=${n_samples:-16}
n_problems=${n_problems:-0}           # 0 = all problems
n_gpus=${n_gpus:-0}                   # 0 = auto-detect from CUDA_VISIBLE_DEVICES
backend=${backend:-"vllm"}
temperature=${temperature:-0.6}
top_p=${top_p:-1}
max_new_tokens=${max_new_tokens:-2048}
dtype=${dtype:-"bfloat16"}
gpu_memory_utilization=${gpu_memory_utilization:-0.60}
tensor_parallel_size=${tensor_parallel_size:-1}
batch_size=${batch_size:-1}
record_lengths=${record_lengths:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_math_avg16.py"

# ── Decide models to evaluate ─────────────────────────────────────────────────
models_to_eval=()
if [ -n "${model}" ]; then
    models_to_eval+=("${model}")
else
    for variant in ${model_variants}; do
        models_to_eval+=("${model_root}/${variant}")
    done
fi

# ── GPU setup ─────────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
n_available_gpus=${#GPU_LIST[@]}
[ "${n_gpus}" -le 0 ] && n_gpus=${n_available_gpus}
if [ "${n_gpus}" -gt "${n_available_gpus}" ]; then
    echo "[WARN] n_gpus=${n_gpus} > available=${n_available_gpus}; capping"
    n_gpus=${n_available_gpus}
fi

# ── Evaluate each model ───────────────────────────────────────────────────────
for current_model in "${models_to_eval[@]}"; do

    output="${current_model}/gsm_avg${n_samples}_res.json"

    echo "=============================="
    echo "  GSM AVG@${n_samples} Evaluation"
    echo "=============================="
    echo "  model    : ${current_model}"
    echo "  data     : ${data}"
    echo "  output   : ${output}"
    echo "  n_samples: ${n_samples}, n_problems: ${n_problems} (0=all)"
    echo "  backend  : ${backend}, n_gpus: ${n_gpus}"
    echo "=============================="

    cmd="python ${EVAL_SCRIPT}"
    cmd+=" --model ${current_model}"
    cmd+=" --data ${data}"
    cmd+=" --n_samples ${n_samples}"
    cmd+=" --n_problems ${n_problems}"
    cmd+=" --backend ${backend}"
    cmd+=" --temperature ${temperature}"
    cmd+=" --top_p ${top_p}"
    cmd+=" --max_new_tokens ${max_new_tokens}"
    cmd+=" --dtype ${dtype}"
    cmd+=" --gpu_memory_utilization ${gpu_memory_utilization}"
    cmd+=" --tensor_parallel_size ${tensor_parallel_size}"
    [ "${backend}" = "hf" ] && cmd+=" --batch_size ${batch_size}"
    [ "${record_lengths}" = "yes" ] && cmd+=" --record_lengths"

    if [ "${n_gpus}" -le 1 ]; then
        eval "${cmd} --output ${output}"
    else
        echo "[multi-gpu] Splitting across ${n_gpus} GPUs: ${GPU_LIST[*]:0:${n_gpus}}"

        if [ "${n_problems}" -gt 0 ]; then
            total_problems=${n_problems}
        else
            total_problems=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${data}')))")
        fi
        echo "[multi-gpu] ${total_problems} problems total"

        chunk_size=$(( (total_problems + n_gpus - 1) / n_gpus ))
        output_base="${output%.json}"
        pids=(); chunk_files=()

        for i in $(seq 0 $((n_gpus - 1))); do
            start_idx=$((i * chunk_size))
            end_idx=$(( (i + 1) * chunk_size ))
            [ "${end_idx}" -gt "${total_problems}" ] && end_idx=${total_problems}
            [ "${start_idx}" -ge "${total_problems}" ] && break

            chunk_file="${output_base}_chunk${i}.json"
            chunk_files+=("${chunk_file}")
            gpu_id="${GPU_LIST[$i]}"

            chunk_cmd="${cmd} --start_idx ${start_idx} --end_idx ${end_idx} --output ${chunk_file}"
            chunk_cmd="${chunk_cmd/--n_problems ${n_problems}/--n_problems 0}"

            echo "[multi-gpu] GPU ${gpu_id}: [${start_idx}, ${end_idx}) → ${chunk_file}"
            (
                export CUDA_VISIBLE_DEVICES="${gpu_id}"
                export CUDA_LAUNCH_BLOCKING=1
                eval "${chunk_cmd}"
            ) > "${output_base}_chunk${i}.log" 2>&1 &
            pids+=($!)
        done

        failed=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || { echo "[ERROR] PID ${pid} failed"; failed=1; }
        done
        [ "${failed}" -ne 0 ] && { echo "[ERROR] Some eval processes failed"; exit 1; }

        echo "[multi-gpu] Merging ${#chunk_files[@]} chunks → ${output}"
        python3 - "${output}" "${chunk_files[@]}" <<'PYEOF'
import json, sys, numpy as np
from collections import defaultdict
output_path = sys.argv[1]
input_files = sys.argv[2:]
all_per_problem = []; cfg = {}
for f in input_files:
    with open(f) as fp: d = json.load(fp)
    all_per_problem.extend(d["per_problem"])
    if not cfg: cfg = d.get("config", {})
all_per_problem.sort(key=lambda r: r["idx"])
n = len(all_per_problem)
n_s = len(all_per_problem[0]["correct_flags"]) if all_per_problem else 0
def breakdown(key):
    g = defaultdict(list)
    for r in all_per_problem: g[r.get(key,"")].append(r["pass_rate"])
    return {k: float(np.mean(v)) for k, v in sorted(g.items())}
result = {"summary": {"n_problems": n, "n_samples_per_problem": n_s,
    "avg@n": float(np.mean([r["pass_rate"] for r in all_per_problem])),
    "pass@1": float(np.mean([r["pass@1"] for r in all_per_problem])),
    "maj@n": float(np.mean([r["maj@n"] for r in all_per_problem])),
    "no_answer_rate": sum(1 for r in all_per_problem if all(a is None for a in r.get("extracted_answers",[None]))) / max(n,1)},
    "by_level": breakdown("level"), "by_type": breakdown("type"),
    "by_data_source": breakdown("data_source"), "per_problem": all_per_problem, "config": cfg}
with open(output_path, "w") as fp: json.dump(result, fp, indent=2, default=str)
s = result["summary"]
print(f"\nMerged: {n} problems, AVG@{n_s}={s['avg@n']*100:.1f}%, maj@{n_s}={s['maj@n']*100:.1f}%")
print(f"Saved → {output_path}")
PYEOF
    fi

    echo "GSM AVG@${n_samples} complete: ${current_model}"
done

echo "All GSM AVG@${n_samples} evaluations complete."
