#!/bin/bash
# eval_niah.sh — NIAH (needle-in-a-haystack) evaluation for a single model.
#
# Evaluates niah_single_1/2/3 at context lengths up to 262K using
# lm-evaluation-harness + flash_attention_2.
#
# Usage: bash scripts/evaluation/eval_niah.sh [KEY=VALUE ...]
#   ORIG_MODEL=/path/to/model
#   model_config=my-model-name
#   RESULTS_BASE_DIR=/path/to/results

set -euo pipefail
for arg in "$@"; do eval "$arg"; done

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1

model_config=${model_config:-"hypenet-2b"}
ORIG_MODEL=${ORIG_MODEL:-"/path/to/model"}
NIAH_LENS=${NIAH_LENS:-"4096,8192,16384,32768,65536,131072,262144"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-"results/niah/${model_config}"}
mkdir -p "${RESULTS_BASE_DIR}"

echo "=============================================="
echo "NIAH eval started at $(date)"
echo "Model config   : ${model_config}"
echo "Model path     : ${ORIG_MODEL}"
echo "NIAH lengths   : ${NIAH_LENS}"
echo "Results dir    : ${RESULTS_BASE_DIR}"
echo "=============================================="

if ! accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=${ORIG_MODEL},dtype=bfloat16,trust_remote_code=True,max_length=262144,attn_implementation=flash_attention_2" \
    --tasks niah_single_1,niah_single_2,niah_single_3 \
    --num_fewshot 0 \
    --batch_size 1 \
    --gen_kwargs "max_new_tokens=128" \
    --metadata="{\"max_seq_lengths\":[${NIAH_LENS}]}" \
    > "${RESULTS_BASE_DIR}/niah_eval.log" 2>&1; then
    echo "❌ NIAH eval failed" | tee "${RESULTS_BASE_DIR}/error.log"
    exit 1
fi

echo "✅ NIAH eval complete → ${RESULTS_BASE_DIR}/niah_eval.log"

SUMMARY_FILE="${RESULTS_BASE_DIR}/summary.txt"
{
    echo "NIAH Eval Summary"
    echo "Timestamp   : ${TIMESTAMP}"
    echo "Model config: ${model_config}"
    echo "Model path  : ${ORIG_MODEL}"
    echo ""
    if [[ -f "${RESULTS_BASE_DIR}/error.log" ]]; then
        echo "NIAH eval: FAILED"
    else
        echo "NIAH eval: OK"
    fi
} > "${SUMMARY_FILE}"

echo "Summary: ${SUMMARY_FILE}"
