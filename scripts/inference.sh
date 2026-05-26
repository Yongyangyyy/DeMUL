#!/usr/bin/env bash
# Run DeMUL inference on a trained checkpoint.
# Usage: bash scripts/inference.sh <model_dir> [split] [gpu]

set -euo pipefail

MODEL_DIR=${1:?"Please provide model_dir as first argument"}
SPLIT=${2:-val}
GPU=${3:-0}

python -u inference.py \
    --model_dir       "${MODEL_DIR}" \
    --eval_split_name "${SPLIT}" \
    --tasks           VCMR SVMR VR \
    --device          "${GPU}" \
    --device_ids      "${GPU}" \
    --max_vcmr_video  100 \
    --nms_thd         0.7 \
    --min_pred_l      0 \
    --max_pred_l      24 \
    --max_before_nms  200
