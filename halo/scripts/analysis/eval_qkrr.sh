#!/bin/bash
# e7_eval_qkrr.sh
#
# E7: Post-SFT Recall Recovery via QK-Restoration (QKRR) — Multi-GPU
#
# Step 1 — Generate all 6 QKRR variant checkpoints from the post-SFT model
#           (Baseline, QKRR-1/2/3/all, SALR-all) using qk_restore.py
# Step 2 — Evaluate every variant + pre-SFT and post-SFT baselines on:
#           - NIAH-S3 (niah_single_3) at ctx [4096, 8192, 16384]
#           - NIAH-S1 (niah_single_1) at ctx [4096, 8192, 16384]
#           Models are distributed across available GPUs and run in parallel.
# Step 3 — (optional) MATH AVG@16 for the key models
# Step 4 — Print a side-by-side comparison table
#
# Usage:
#   bash e7_eval_qkrr.sh                        # all 8 GPUs, 8 models in parallel
#   bash e7_eval_qkrr.sh gpus=0,1,2,3           # 4 GPUs
#   bash e7_eval_qkrr.sh gpus=0                 # single GPU, sequential
#   bash e7_eval_qkrr.sh gpus=0,1,2,3 tp=2      # tensor-parallel: 2 GPUs per model
#   bash e7_eval_qkrr.sh skip_gen=1             # skip variant generation
#   bash e7_eval_qkrr.sh run_math=1             # also run MATH eval
#   bash e7_eval_qkrr.sh job_timeout=14400      # per-job timeout in seconds (default 4h)
#   bash e7_eval_qkrr.sh batch_size=8           # NIAH batch size (default: 8)
#   bash e7_eval_qkrr.sh math_tasks=mathqa_local,aqua_rat   # lm_eval math tasks
#   bash e7_eval_qkrr.sh math_batch_size=32     # math lm_eval batch size (default: 32)

set -euo pipefail

# ── Parse key=value args ──────────────────────────────────────────────────────
for arg in "$@"; do eval "$arg"; done

# ── Configuration ─────────────────────────────────────────────────────────────
PRE_SFT=${pre_sft:-"/path/to/results/hypenet-Stage3/MixedData_stage3_2b_OldLayers/hf/ckpt_500"}
POST_SFT=${post_sft:-"/path/to/results/MixedData_hypenet_CoT_SFT/2b_cot_sft_math_OldSelection/ckpt_100"}
VARIANT_DIR=${variant_dir:-"/path/to/results/QKRR/hypenet-2b-old-selection"}
RESULTS_DIR=${results_dir:-"results/QKRR/hypenet-2b-old-selection"}
RECALL_LAYERS=${recall_layers:-"2,4,8,9,14,25,26"}
NIAH_LENS=${niah_lens:-"4096,8192,16384,32768,65536,131072,262144"}
# max_length: context window passed to lm_eval for NIAH.
# 65536 allows proper 32K evaluation (default); set to 131072 for full range.
MAX_LENGTH=${max_length:-268174}

# gpus: comma-separated GPU IDs (e.g. "0,1,2,3")
# tp:   GPUs per model for tensor parallelism; tp=1 = one GPU per model
# job_timeout: seconds before a single eval job is killed (default 4h)
GPUS=${gpus:-"0,1,2,3,4,5,6,7"}
TP=${tp:-1}
JOB_TIMEOUT=${job_timeout:-14400}
# "auto" lets lm_eval binary-search for the largest batch that fits in VRAM.
# Override with a fixed integer (e.g. batch_size=4) if you want deterministic memory use.
BATCH_SIZE=${batch_size:-1}
# lm_eval math tasks (same as attn-layer-selection/scripts/evaluation/eval_math.sh)
MATH_TASKS=${math_tasks:-"mathqa_local,aqua_rat"}
MATH_BATCH_SIZE=${math_batch_size:-32}

skip_gen=${skip_gen:-0}
run_math=${run_math:-0}
MATH_DATA=${math_data:-"/path/to/data/math/test.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Kill all child processes on exit / Ctrl-C ─────────────────────────────────
_CHILD_PIDS=()
cleanup() {
    if (( ${#_CHILD_PIDS[@]} > 0 )); then
        echo ""
        echo "[CLEANUP] Killing ${#_CHILD_PIDS[@]} background job(s)..."
        kill "${_CHILD_PIDS[@]}" 2>/dev/null || true
        wait "${_CHILD_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

# ── Parse GPU list and build worker groups ────────────────────────────────────
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}
N_WORKERS=$(( N_GPUS / TP ))

if (( N_WORKERS < 1 )); then
    echo "[ERROR] gpus=${GPUS} has ${N_GPUS} GPUs but tp=${TP} requires at least ${TP} per job"
    exit 1
fi

GPU_GROUPS=()
for (( w = 0; w < N_WORKERS; w++ )); do
    grp=""
    for (( t = 0; t < TP; t++ )); do
        idx=$(( w * TP + t ))
        [[ -n "$grp" ]] && grp+=","
        grp+="${GPU_ARR[$idx]}"
    done
    GPU_GROUPS+=("$grp")
done

MODEL_EXTRA=""
(( TP > 1 )) && MODEL_EXTRA=",device_map=auto"

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_PROFILE="/path/to/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate hybridrl
else
    echo "[WARN] Conda profile not found — assuming env is active"
fi

mkdir -p "${RESULTS_DIR}"

echo "============================================================"
echo "  E7: QK-Restoration (QKRR) — Generate + Evaluate"
echo "============================================================"
echo "  pre_sft      : ${PRE_SFT}"
echo "  post_sft     : ${POST_SFT}"
echo "  variant_dir  : ${VARIANT_DIR}"
echo "  results_dir  : ${RESULTS_DIR}"
echo "  recall_layers: ${RECALL_LAYERS}"
echo "  niah_lens    : ${NIAH_LENS}"
echo "  max_length   : ${MAX_LENGTH}"
echo "  gpus         : ${GPUS}  (${N_GPUS} GPUs, tp=${TP}, ${N_WORKERS} parallel workers)"
echo "  job_timeout  : ${JOB_TIMEOUT}s"
echo "  batch_size   : ${BATCH_SIZE}"
echo "  math_tasks   : ${MATH_TASKS}"
echo "  math_batch   : ${MATH_BATCH_SIZE}"
echo "  skip_gen     : ${skip_gen}"
echo "  run_math     : ${run_math}"
echo "============================================================"
echo ""

# ── Reusable: run a batch of jobs in parallel, wait, report ──────────────────
# Usage: run_batch TAG CMD_PREFIX name1 name2 ...
#   TAG        = label for log messages (e.g. "NIAH" or "MATH")
#   CMD_PREFIX = everything before the model path (as a single string); the
#                function appends per-job args and launches with timeout
#
# Internally this function uses global arrays EVAL_MODELS / RESULTS_DIR.
# Each job's stdout+stderr goes to RESULTS_DIR/${TAG,,}_${name}.log
# Returns 0 if all succeeded, exits the whole script on any failure.
run_parallel_batch() {
    local tag="$1"
    local batch_num=0
    shift
    local names=("$@")
    local n=${#names[@]}

    if (( n == 0 )); then
        echo "  [${tag}] Nothing to run."
        return 0
    fi

    local i=0
    while (( i < n )); do
        batch_num=$(( batch_num + 1 ))
        local batch_pids=()
        local batch_names=()
        local batch_logs=()

        for (( w = 0; w < N_WORKERS && i < n; w++, i++ )); do
            local name="${names[$i]}"
            local ckpt="${EVAL_MODELS[$name]}"
            local out_log="${RESULTS_DIR}/${tag,,}_${name}.log"
            local gpu_grp="${GPU_GROUPS[$w]}"

            echo "  [${tag} launch batch=${batch_num} worker=${w}] ${name}  →  GPU(s) ${gpu_grp}"

            # Build task-specific command
            if [[ "$tag" == "NIAH" ]]; then
                local out_json="${RESULTS_DIR}/niah_${name}.json"
                CUDA_VISIBLE_DEVICES="${gpu_grp}" \
                python -m lm_eval \
                    --model hf \
                    --model_args "pretrained=${ckpt},dtype=bfloat16,trust_remote_code=True,max_length=${MAX_LENGTH}${MODEL_EXTRA}" \
                    --tasks niah_single_1,niah_single_2,niah_single_3 \
                    --num_fewshot 0 \
                    --batch_size "${BATCH_SIZE}" \
                    --metadata "{\"max_seq_lengths\":[${NIAH_LENS}]}" \
                    --output_path "${out_json}" \
                    > "${out_log}" 2>&1 &
            elif [[ "$tag" == "MATH" ]]; then
                local out_json="${RESULTS_DIR}/math_${name}.json"
                CUDA_VISIBLE_DEVICES="${gpu_grp}" \
                bash "${SCRIPT_DIR}/eval_math_avg16.sh" \
                    model="${ckpt}" \
                    output="${out_json}" \
                    n_samples=16 \
                    > "${out_log}" 2>&1 &
            elif [[ "$tag" == "MATH_LM" ]]; then
                local out_json="${RESULTS_DIR}/math_lm_${name}.json"
                CUDA_VISIBLE_DEVICES="${gpu_grp}" \
                timeout "${JOB_TIMEOUT}" \
                python -m lm_eval \
                    --model hf \
                    --model_args "pretrained=${ckpt},dtype=bfloat16,trust_remote_code=True${MODEL_EXTRA}" \
                    --tasks "${MATH_TASKS}" \
                    --num_fewshot 0 \
                    --batch_size "${MATH_BATCH_SIZE}" \
                    --output_path "${out_json}" \
                    > "${out_log}" 2>&1 &
            fi

            local pid=$!
            _CHILD_PIDS+=($pid)
            batch_pids+=($pid)
            batch_names+=("$name")
            batch_logs+=("$out_log")
        done

        # Heartbeat: print elapsed time every 60s while this batch runs
        local batch_start=$SECONDS
        (
            while true; do
                sleep 60
                elapsed=$(( SECONDS - batch_start ))
                echo "  [${tag}] Batch ${batch_num} still running — ${elapsed}s elapsed (jobs: ${batch_names[*]})"
            done
        ) &
        local hb_pid=$!
        _CHILD_PIDS+=($hb_pid)

        # Collect results
        local all_ok=1
        for (( b = 0; b < ${#batch_pids[@]}; b++ )); do
            local pid="${batch_pids[$b]}"
            local bname="${batch_names[$b]}"
            local blog="${batch_logs[$b]}"
            local elapsed

            if wait "$pid"; then
                elapsed=$(( SECONDS - batch_start ))
                echo "  [${tag} DONE] ${bname}  (${elapsed}s)"
            else
                local rc=$?
                elapsed=$(( SECONDS - batch_start ))
                if (( rc == 124 )); then
                    echo "  [${tag} TIMEOUT] ${bname} exceeded ${JOB_TIMEOUT}s — see ${blog}"
                else
                    echo "  [${tag} FAIL] ${bname} exit=${rc} — see ${blog}"
                fi
                echo "  ── last 20 lines of log ──────────────────────────────"
                tail -n 20 "${blog}" | sed 's/^/  /'
                echo "  ──────────────────────────────────────────────────────"
                all_ok=0
            fi
        done

        # Stop heartbeat
        kill "$hb_pid" 2>/dev/null || true
        wait "$hb_pid" 2>/dev/null || true
        # Remove heartbeat from _CHILD_PIDS (best-effort; harmless to leave)
        _CHILD_PIDS=("${_CHILD_PIDS[@]/$hb_pid/}")

        echo ""

        if (( !all_ok )); then
            echo "[ERROR] One or more ${tag} jobs in batch ${batch_num} failed. Aborting."
            exit 1
        fi
    done
}

# ── Generate QKRR variant checkpoints ─────────────────────────────────
if [ "${skip_gen}" = "0" ]; then
    echo "[Step 1] Generating QKRR variants (including ablation Q/K/V-only)..."
    CUDA_VISIBLE_DEVICES="${GPU_GROUPS[0]}" python "${SCRIPT_DIR}/qk_restore.py" \
        --pre_sft  "${PRE_SFT}" \
        --post_sft "${POST_SFT}" \
        --out_dir  "${VARIANT_DIR}" \
        --recall_layers "${RECALL_LAYERS}" \
        --configs  "SALR-all,QONLY-all,KONLY-all,VONLY-all" \
        --dtype    bfloat16
    echo "[Step 1] Done."
else
    echo "[Step 1] Skipped (skip_gen=1)."
fi
echo ""
# SALR-all,QONLY-all,KONLY-all,VONLY-all
