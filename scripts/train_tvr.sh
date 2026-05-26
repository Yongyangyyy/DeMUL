#!/usr/bin/env bash
# Train DeMUL on TVR dataset.
# Usage: bash scripts/train_tvr.sh [exp_id] [seed] [gpu]

set -euo pipefail

EXP_ID=${1:-"demul_tvr"}
SEED=${2:-2018}
GPU=${3:-0}

python -u train.py \
    --exp_id          "${EXP_ID}" \
    --dset_name       tvr \
    --eval_split_name val \
    --dataset_config  config/tvr_data_config.json \
    --model_config    config/model_config.json \
    --ctx_mode        visual_sub \
    --visual_dim      4352 \
    --text_dim        768 \
    --query_dim       768 \
    --hidden_dim      384 \
    --n_epoch         50 \
    --bsz             32 \
    --lr              1e-4 \
    --neg_video_num   10 \
    --use_extend_pool 1000 \
    --max_ctx_len     100 \
    --max_desc_len    30 \
    --clip_length     1.5 \
    --max_es_cnt      5 \
    --stop_task       VCMR \
    --eval_tasks_at_training VCMR SVMR VR \
    --use_interal_vr_scores \
    --max_vcmr_video  10 \
    --min_pred_l      1 \
    --max_pred_l      24 \
    --max_before_nms  600 \
    --num_workers     8 \
    --nms_thd 0.7 \
    --seed            "${SEED}" \
    --device          "${GPU}" \
    --device_ids      "${GPU}"
