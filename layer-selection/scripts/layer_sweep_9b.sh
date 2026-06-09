#!/bin/bash
# layer_sweep_9b.sh — Layer importance sweep for HypeNet-9B
#
# Same structure as layer_sweep_5b.sh.  Key differences:
#   - Teacher: qwen3-8B (36-layer model → END_LAYER=35)
#   - Math eval is disabled by default (resource constraints); set run_math=1 to enable
#
# Usage: bash scripts/layer_sweep_9b.sh [KEY=VALUE ...]

set -euo pipefail
for arg in "$@"; do eval "$arg"; done

############################
# User-configurable paths
############################
ORIG_MODEL=${ORIG_MODEL:-"/path/to/models/qwen3-8B"}
RNN_BASE_PATH=${RNN_BASE_PATH:-"/path/to/results/hypenet-Stage1/MixedData_hypenet-9b-mix_old_selection_stage1_WebToken0.8"}
CKPT_NUM=${CKPT_NUM:-20000}
RNN_HF_MODEL=${RNN_HF_MODEL:-"${RNN_BASE_PATH}/hf/ckpt_${CKPT_NUM}"}
OUTPUT_NAME=${OUTPUT_NAME:-"hypenet_9b_mix_oldselection"}

MODEL_TYPE=${MODEL_TYPE:-hypenet-9b-mix_old_selection}
RNN_TYPE=${RNN_TYPE:-lightning-attn}

############################
# Layer sweep config
############################
START_LAYER=${START_LAYER:-0}
END_LAYER=${END_LAYER:-35}

############################
# Eval config
############################
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}
COMMONSENSE_BATCH_SIZE=${COMMONSENSE_BATCH_SIZE:-32}
RETRIEVAL_BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-32}

############################
# Results directory
############################
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-results/${OUTPUT_NAME}/layer_sweep}"
OUTPUT_MODELS_DIR=${OUTPUT_MODELS_DIR:-/tmp/attn_layer_selection/models}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_SEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONVERT_PY="${LAYER_SEL_DIR}/convert.py"

mkdir -p "${RESULTS_BASE_DIR}"
mkdir -p "${OUTPUT_MODELS_DIR}"

[[ ! -d "${ORIG_MODEL}" ]]  && { echo "ORIG_MODEL does not exist: ${ORIG_MODEL}"; exit 1; }
[[ ! -d "${RNN_BASE_PATH}" ]] && { echo "RNN_BASE_PATH does not exist: ${RNN_BASE_PATH}"; exit 1; }
[[ ! -f "${CONVERT_PY}" ]]  && { echo "convert.py not found: ${CONVERT_PY}"; exit 1; }
[[ -n "${RNN_HF_MODEL}" && ! -d "${RNN_HF_MODEL}" ]] && { echo "RNN_HF_MODEL does not exist: ${RNN_HF_MODEL}"; exit 1; }

echo "=============================================="
echo "Layer sweep started at $(date)"
echo "Model type     : ${MODEL_TYPE}"
echo "RNN type       : ${RNN_TYPE}"
echo "Teacher model  : ${ORIG_MODEL}"
echo "Stage1 dir     : ${RNN_BASE_PATH}"
echo "Stage1 HF      : ${RNN_HF_MODEL}"
echo "Layers         : ${START_LAYER} → ${END_LAYER}"
echo "Results dir    : ${RESULTS_BASE_DIR}"
echo "Clean outputs  : ${CLEAN_OUTPUT}"
echo "=============================================="


########################################
# Layer conversion + evaluation
########################################
run_layer_evaluation () {
    local layer=$1

    local model_name="${MODEL_TYPE}_${RNN_TYPE}/layer${layer}"
    local output_root="${OUTPUT_MODELS_DIR}/${model_name}"
    local layer_dir="${RESULTS_BASE_DIR}/${model_name}"

    mkdir -p "${layer_dir}"

    echo "[Layer ${layer}] Converting model..."

    local -a convert_cmd=(
        python "${CONVERT_PY}"
        --orig_model "${ORIG_MODEL}"
        --rnn_base_path "${RNN_BASE_PATH}"
        --ckpt_num "${CKPT_NUM}"
        --output_root "${output_root}"
        --layer_indices "${layer}"
        --rnn_type "${RNN_TYPE}"
    )
    [[ -n "${RNN_HF_MODEL}" ]] && convert_cmd+=(--rnn_hf_model "${RNN_HF_MODEL}")

    if ! "${convert_cmd[@]}" > "${layer_dir}/conversion.log" 2>&1; then
        echo "[Layer ${layer}] ❌ Conversion failed" | tee "${layer_dir}/error.log"
        return 1
    fi

    echo "[Layer ${layer}] Running commonsense eval..."
    if ! accelerate launch --config_file "${LAYER_SEL_DIR}/configs/infer.yaml" -m lm_eval \
        --model hf \
        --model_args "pretrained=${output_root},dtype=bfloat16,trust_remote_code=True" \
        --tasks hellaswag,arc_easy,arc_challenge \
        --num_fewshot 0 \
        --batch_size "${COMMONSENSE_BATCH_SIZE}" \
        > "${layer_dir}/commonsense_eval.log" 2>&1; then
        echo "[Layer ${layer}] ❌ Commonsense eval failed" >> "${layer_dir}/error.log"
    fi

    echo "[Layer ${layer}] Running retrieval eval..."
    if ! accelerate launch --config_file "${LAYER_SEL_DIR}/configs/infer.yaml" -m lm_eval \
        --model hf \
        --model_args "pretrained=${output_root},dtype=bfloat16,trust_remote_code=True" \
        --tasks ruler_qa_squad,fda,swde \
        --num_fewshot 0 \
        --batch_size "${RETRIEVAL_BATCH_SIZE}" \
        > "${layer_dir}/retrieval_eval.log" 2>&1; then
        echo "[Layer ${layer}] ❌ Retrieval eval failed" >> "${layer_dir}/error.log"
    fi

    if [[ "${CLEAN_OUTPUT}" == "1" && -d "${output_root}" ]]; then
        rm -rf "${output_root}"
        echo "[Layer ${layer}] Checkpoints cleaned."
    fi

    echo "[Layer ${layer}] ✅ Done"
}


########################################
# Main sweep
########################################
for (( layer=START_LAYER; layer<=END_LAYER; layer++ )); do
    echo "----------------------------------------------"
    echo "Starting layer ${layer}"
    echo "----------------------------------------------"
    run_layer_evaluation "${layer}" || echo "[Layer ${layer}] ⚠️  Failed, continuing..."
done


########################################
# Summary
########################################
SUMMARY_FILE="${RESULTS_BASE_DIR}/summary.txt"
{
    echo "Layer Sweep Summary"
    echo "Timestamp  : ${TIMESTAMP}"
    echo "Model type : ${MODEL_TYPE}"
    echo "RNN type   : ${RNN_TYPE}"
    echo "Teacher    : ${ORIG_MODEL}"
    echo "Stage1 HF  : ${RNN_HF_MODEL}"
    echo ""
    for (( layer=START_LAYER; layer<=END_LAYER; layer++ )); do
        layer_dir="${RESULTS_BASE_DIR}/${MODEL_TYPE}_${RNN_TYPE}/layer${layer}"
        if [[ -f "${layer_dir}/error.log" ]]; then
            echo "Layer ${layer}: FAILED"
        else
            echo "Layer ${layer}: OK"
        fi
    done
} > "${SUMMARY_FILE}"

echo "=============================================="
echo "Layer sweep completed at $(date)"
echo "Summary: ${SUMMARY_FILE}"
echo "=============================================="
