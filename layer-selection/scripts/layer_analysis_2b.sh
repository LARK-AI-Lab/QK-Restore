#!/bin/bash
# layer_analysis_2b.sh — Score HALO layer importance for HypeNet-2B after a sweep.
#
# Run this after layer_sweep_2b.sh completes.
# Usage: bash scripts/layer_analysis_2b.sh [RESULTS_BASE_DIR=<path>]

set -euo pipefail
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_SEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-results/hypenet_2b_mix/layer_sweep}"
MODEL="${MODEL:-hypenet-2b_lightning-attn}"

cd "${LAYER_SEL_DIR}"
python layer_analysis.py \
    --root  "${RESULTS_BASE_DIR}" \
    --model "${MODEL}"
