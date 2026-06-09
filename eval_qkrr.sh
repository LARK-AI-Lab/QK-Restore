#!/bin/bash
# eval_qkrr.sh
#
# QK-RESTORE Evaluation Script
#
# Generates all QK-RESTORE variant checkpoints from a post-SFT model, then
# evaluates each variant on:
#   - NIAH (needle-in-a-haystack) at multiple context lengths
#   - Math reasoning (optional: avg@16 sampling + lm_eval tasks)
#
# Usage
# -----
#   bash eval_qkrr.sh                          # defaults, all 8 GPUs
#   bash eval_qkrr.sh gpus=0,1,2,3             # 4 GPUs
#   bash eval_qkrr.sh skip_gen=1               # skip checkpoint generation
#   bash eval_qkrr.sh run_math=1               # also run math evaluation
#   bash eval_qkrr.sh tp=2 gpus=0,1,2,3        # tensor-parallel, 2 GPUs/model
#   bash eval_qkrr.sh max_length=131072         # extend context window
#
# Key-value args can be passed in any order.  All have sensible defaults.

set -euo pipefail

# ── Parse key=value args ──────────────────────────────────────────────────────
for arg in "$@"; do eval "$arg"; done

# ── Checkpoint paths ──────────────────────────────────────────────────────────
PRE_SFT=${pre_sft:-"/path/to/results/hypenet-Stage3/MixedData_stage3_2b_OldLayers/hf/ckpt_500"}
POST_SFT=${post_sft:-"/path/to/results/MixedData_hypenet_CoT_SFT/2b_cot_sft_math_OldSelection/ckpt_100"}
VARIANT_DIR=${variant_dir:-"/path/to/results/qk_restore_variants"}
RESULTS_DIR=${results_dir:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results/qkrr"}

# ── Layer configuration ───────────────────────────────────────────────────────
# Ordered from most to least recall-critical (from layer-sweep analysis).
RECALL_LAYERS=${recall_layers:-"2,4,8,9,14,25,26"}

# ── Evaluation settings ───────────────────────────────────────────────────────
# NIAH context lengths to evaluate (comma-separated, in tokens)
NIAH_LENS=${niah_lens:-"4096,8192,16384,32768,65536"}

# max_length passed to lm_eval for NIAH (should be ≥ max(NIAH_LENS))
MAX_LENGTH=${max_length:-131072}

# ── GPU / parallelism ─────────────────────────────────────────────────────────
GPUS=${gpus:-"0,1,2,3,4,5,6,7"}     # comma-separated GPU IDs
TP=${tp:-1}                           # GPUs per model for tensor parallelism
JOB_TIMEOUT=${job_timeout:-14400}     # per-job wall-clock limit in seconds
BATCH_SIZE=${batch_size:-1}           # lm_eval batch size for NIAH

# ── Math eval settings (used only when run_math=1) ───────────────────────────
MATH_TASKS=${math_tasks:-"mathqa_local,aqua_rat"}
MATH_BATCH_SIZE=${math_batch_size:-32}

# ── Control flags ─────────────────────────────────────────────────────────────
skip_gen=${skip_gen:-0}      # 1 = skip checkpoint generation, go straight to eval
run_math=${run_math:-0}      # 1 = also run math evaluation

# ── Derived ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Graceful shutdown ─────────────────────────────────────────────────────────
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

# ── Build GPU worker groups ───────────────────────────────────────────────────
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
        [[ -n "$grp" ]] && grp+=","
        grp+="${GPU_ARR[$(( w * TP + t ))]}"
    done
    GPU_GROUPS+=("$grp")
done

MODEL_EXTRA=""
(( TP > 1 )) && MODEL_EXTRA=",device_map=auto"

# ── Conda environment ─────────────────────────────────────────────────────────
CONDA_PROFILE="/path/to/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate hybridrl
else
    echo "[WARN] Conda profile not found — assuming the correct environment is active"
fi

mkdir -p "${RESULTS_DIR}"

# ── Print configuration ───────────────────────────────────────────────────────
echo "================================================================"
echo "  QK-RESTORE  —  Generate + Evaluate"
echo "================================================================"
echo "  pre_sft       : ${PRE_SFT}"
echo "  post_sft      : ${POST_SFT}"
echo "  variant_dir   : ${VARIANT_DIR}"
echo "  results_dir   : ${RESULTS_DIR}"
echo "  recall_layers : ${RECALL_LAYERS}"
echo "  niah_lens     : ${NIAH_LENS}"
echo "  max_length    : ${MAX_LENGTH}"
echo "  gpus          : ${GPUS}  (${N_GPUS} total, tp=${TP}, ${N_WORKERS} parallel workers)"
echo "  job_timeout   : ${JOB_TIMEOUT}s"
echo "  batch_size    : ${BATCH_SIZE}"
echo "  skip_gen      : ${skip_gen}"
echo "  run_math      : ${run_math}"
echo "================================================================"
echo ""

# ── Reusable parallel batch runner ───────────────────────────────────────────
# run_parallel_batch TAG name1 name2 ...
#
# Launches evaluation jobs in waves of N_WORKERS, each on its assigned GPU group.
# All stdout/stderr goes to RESULTS_DIR/${tag,,}_${name}.log.
# Prints a heartbeat every 60 s while a batch is in flight.
# Exits non-zero (and aborts the script) if any job in any batch fails.
run_parallel_batch() {
    local tag="$1"; shift
    local names=("$@")
    local n=${#names[@]}

    if (( n == 0 )); then
        echo "  [${tag}] Nothing to run."
        return 0
    fi

    local i=0 batch_num=0
    while (( i < n )); do
        batch_num=$(( batch_num + 1 ))
        local batch_pids=() batch_names=() batch_logs=()

        for (( w = 0; w < N_WORKERS && i < n; w++, i++ )); do
            local name="${names[$i]}"
            local ckpt="${EVAL_MODELS[$name]}"
            local gpu_grp="${GPU_GROUPS[$w]}"
            local out_log="${RESULTS_DIR}/${tag,,}_${name}.log"

            echo "  [${tag} launch batch=${batch_num}] ${name}  →  GPU(s) ${gpu_grp}"

            case "$tag" in
            NIAH)
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
                ;;
            MATH_LM)
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
                ;;
            esac

            local pid=$!
            _CHILD_PIDS+=($pid)
            batch_pids+=($pid)
            batch_names+=("$name")
            batch_logs+=("$out_log")
        done

        # Heartbeat
        local batch_start=$SECONDS
        ( while true; do
            sleep 60
            echo "  [${tag}] Batch ${batch_num} still running — $(( SECONDS - batch_start ))s  (${batch_names[*]})"
          done ) &
        local hb_pid=$!
        _CHILD_PIDS+=($hb_pid)

        # Collect results
        local all_ok=1
        for (( b = 0; b < ${#batch_pids[@]}; b++ )); do
            local pid="${batch_pids[$b]}"
            local bname="${batch_names[$b]}"
            local blog="${batch_logs[$b]}"

            if wait "$pid"; then
                echo "  [${tag} OK] ${bname}  ($(( SECONDS - batch_start ))s)"
            else
                local rc=$?
                echo "  [${tag} FAIL] ${bname} exit=${rc} — see ${blog}"
                echo "  ── last 20 lines ──────────────────────────────────"
                tail -n 20 "${blog}" | sed 's/^/  /'
                echo "  ───────────────────────────────────────────────────"
                all_ok=0
            fi
        done

        kill "$hb_pid" 2>/dev/null || true
        wait "$hb_pid" 2>/dev/null || true
        echo ""

        (( all_ok )) || { echo "[ERROR] ${tag} batch ${batch_num} had failures."; exit 1; }
    done
}

# ── Model registry ────────────────────────────────────────────────────────────
declare -A EVAL_MODELS
EVAL_MODELS["pre_sft"]="${PRE_SFT}"
EVAL_MODELS["post_sft"]="${POST_SFT}"
EVAL_MODELS["qkrr_all"]="${VARIANT_DIR}/qkrr_all"
EVAL_MODELS["konly_all"]="${VARIANT_DIR}/konly_all"
EVAL_MODELS["vonly_all"]="${VARIANT_DIR}/vonly_all"
EVAL_MODELS["qkrr_procrustes"]="${VARIANT_DIR}/qkrr_procrustes"

# Evaluation order
EVAL_ORDER=(
    pre_sft
    post_sft
    qkrr_all
    konly_all
    vonly_all
    qkrr_procrustes
)

# ── Step 1: Generate variant checkpoints ──────────────────────────────────────
if [ "${skip_gen}" = "0" ]; then
    echo "[Step 1] Generating QK-RESTORE variant checkpoints ..."
    CUDA_VISIBLE_DEVICES="${GPU_GROUPS[0]}" \
    python "${SCRIPT_DIR}/qk_restore.py" \
        --pre_sft  "${PRE_SFT}" \
        --post_sft "${POST_SFT}" \
        --out_dir  "${VARIANT_DIR}" \
        --recall_layers "${RECALL_LAYERS}" \
        --dtype bfloat16
    echo "[Step 1] Done."
else
    echo "[Step 1] Skipped (skip_gen=1)."
fi
echo ""

# ── Step 2: NIAH evaluation ───────────────────────────────────────────────────
echo "[Step 2] Running NIAH evaluations (${N_WORKERS} parallel workers) ..."

NIAH_PENDING=()
for name in "${EVAL_ORDER[@]}"; do
    ckpt="${EVAL_MODELS[$name]:-}"
    [ -z "${ckpt}" ]         && { echo "  [SKIP] ${name}: not in model registry"; continue; }
    [ ! -d "${ckpt}" ]       && { echo "  [SKIP] ${name}: checkpoint not found (${ckpt})"; continue; }
    [ -f "${RESULTS_DIR}/niah_${name}.json" ] && { echo "  [SKIP] ${name}: already evaluated"; continue; }
    NIAH_PENDING+=("$name")
done

run_parallel_batch "NIAH" "${NIAH_PENDING[@]+"${NIAH_PENDING[@]}"}"
echo "[Step 2] All NIAH evaluations complete."
echo ""

# ── Step 3: Math evaluation (optional) ───────────────────────────────────────
if [ "${run_math}" = "1" ]; then
    echo "[Step 3] Running math evaluations (lm_eval, ${N_WORKERS} parallel workers) ..."

    MATH_LM_PENDING=()
    for name in "${EVAL_ORDER[@]}"; do
        ckpt="${EVAL_MODELS[$name]:-}"
        [ -z "${ckpt}" ] || [ ! -d "${ckpt}" ] && continue
        [ -f "${RESULTS_DIR}/math_lm_${name}.json" ] && { echo "  [SKIP] ${name}: already evaluated"; continue; }
        MATH_LM_PENDING+=("$name")
    done

    run_parallel_batch "MATH_LM" "${MATH_LM_PENDING[@]+"${MATH_LM_PENDING[@]}"}"
    echo "[Step 3] Math evaluations complete."
    echo ""
else
    echo "[Step 3] Math eval skipped (run_math=0; re-run with run_math=1 to enable)."
    echo ""
fi

# ── Step 4: Results table ─────────────────────────────────────────────────────
echo "[Step 4] Results summary:"
echo ""

export RESULTS_DIR NIAH_LENS MATH_TASKS

python3 - <<'PYEOF'
import json, os
from pathlib import Path

RESULTS_DIR   = os.environ["RESULTS_DIR"]
NIAH_LENS     = [int(x) for x in os.environ.get("NIAH_LENS", "4096,8192,16384").split(",")]
MATH_TASKS    = os.environ.get("MATH_TASKS", "mathqa_local,aqua_rat").split(",")

EVAL_ORDER = [
    "pre_sft",
    "post_sft",
    "qkrr_all",
    "konly_all",
    "vonly_all",
    "qkrr_procrustes",
]

LABELS = {
    "pre_sft":         "pre-SFT (Stage 3 baseline)",
    "post_sft":        "post-SFT (CoT-SFT checkpoint)  [reference]",
    "qkrr_all":        "QKRR-all  (restore Q+K, all layers)   [main]",
    "konly_all":       "K-only    (restore K only)   [ablation]",
    "vonly_all":       "VO-only   (restore V+O only) [ablation]",
    "qkrr_procrustes": "QKRR-Procrustes (min-deviation constrained)",
}

def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def get_niah_score(data, task, length):
    if data is None:
        return None
    for key, val in data.get("results", {}).get(task, {}).items():
        if key.startswith(f"{length},"):
            return float(val) if val not in (-1, -1.0) else None
    return None

def get_lm_math(data, task):
    if data is None:
        return None
    v = data.get("results", {}).get(task, {}).get("acc,none")
    return float(v) if v is not None else None

def fmt(v, width=6):
    return f"{v*100:{width}.1f}%" if v is not None else f"{'n/a':>{width+1}}"

rdir = Path(RESULTS_DIR)
lens_hdr  = "  ".join(f"S3@{L//1024}K" for L in NIAH_LENS)
task_hdrs = "  ".join(f"{t[:8]:>8}" for t in MATH_TASKS)
W = 100

print("=" * W)
print(f"{'Model':<50}  {lens_hdr}    {task_hdrs}")
print("=" * W)
for name in EVAL_ORDER:
    label     = LABELS.get(name, name)
    niah_data = load_json(rdir / f"niah_{name}.json")
    lm_data   = load_json(rdir / f"math_lm_{name}.json")
    s3_str    = "  ".join(fmt(get_niah_score(niah_data, "niah_single_3", L)) for L in NIAH_LENS)
    lm_str    = "  ".join(f"{fmt(get_lm_math(lm_data, t), 7):>8}" for t in MATH_TASKS)
    print(f"{label:<50}  {s3_str}    {lm_str}")

print()
s1_hdr = "  ".join(f"S1@{L//1024}K" for L in NIAH_LENS)
print("─" * W)
print(f"{'Model':<50}  {s1_hdr}    (NIAH-S1: control)")
print("─" * W)
for name in EVAL_ORDER:
    label     = LABELS.get(name, name)
    niah_data = load_json(rdir / f"niah_{name}.json")
    s1_str    = "  ".join(fmt(get_niah_score(niah_data, "niah_single_1", L)) for L in NIAH_LENS)
    print(f"{label:<50}  {s1_str}")
print("=" * W)
print()
print("Key comparisons:")
print("  QKRR-all > pre-SFT       → super-recovery: pre-SFT routing × post-SFT extraction")
print("  QKRR-all > post-SFT      → confirms CoT-SFT degrades long-context recall")
print()
print("Ablation:")
print("  VO-only  ≈ post-SFT  → V/O do not carry routing information")
print("  K-only vs QKRR-all   → contribution of Q vs K to recall recovery")
PYEOF

echo ""
echo "================================================================"
echo "  QK-RESTORE evaluation complete.  Results: ${RESULTS_DIR}"
echo "================================================================"
