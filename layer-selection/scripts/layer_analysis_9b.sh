#!/bin/bash
# layer_analysis_9b.sh — Score HALO layer importance for HypeNet-9B after a sweep.
#
# Usage: bash scripts/layer_analysis_9b.sh [RESULTS_BASE_DIR=<path>]

set -euo pipefail
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_SEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-results/hypenet_9b_mix_oldselection/layer_sweep}"
MODEL="${MODEL:-hypenet-9b-mix_old_selection_lightning-attn}"

cd "${LAYER_SEL_DIR}"
python layer_analysis.py \
    --root  "${RESULTS_BASE_DIR}" \
    --model "${MODEL}"
