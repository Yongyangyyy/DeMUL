#!/usr/bin/env bash
# Train DeMUL on DiDeMo dataset.
# Usage: bash scripts/train_didemo.sh [exp_id] [seed] [gpu] [encoder_ckpt]

set -euo pipefail

EXP_ID=${1:-"demul_didemo"}
SEED=${2:-2018}
GPU=${3:-0}
ENCODER_CKPT=${4:-""}

EXTRA_ARGS=()
if [[ -n "${ENCODER_CKPT}" ]]; then
    EXTRA_ARGS+=(--encoder_pretrain_ckpt_filepath "${ENCODER_CKPT}")
fi

python -u train.py \
    --exp_id          "${EXP_ID}" \
    --dset_name       didemo \
    --eval_split_name val \
    --dataset_config  config/didemo_data_config.json \
    --model_config    config/model_config.json \
    --ctx_mode        visual_sub \
    --visual_dim      4352 \
    --text_dim        768 \
    --query_dim       768 \
    --hidden_dim      384 \
    --n_epoch         50 \
    --bsz             32 \
    --lr              1e-4 \
    --neg_video_num   3 \
    --use_extend_pool 1000 \
    --max_ctx_len     20 \
    --max_desc_len    30 \
    --clip_length     1.5 \
    --max_es_cnt      5 \
    --stop_task       VCMR \
    --eval_tasks_at_training VCMR SVMR VR \
    --use_interal_vr_scores \
    --max_vcmr_video  10 \
    --min_pred_l      3 \
    --max_pred_l      7 \
    --max_before_nms  200 \
    --num_workers     8 \
    --nms_thd 0.7 \
    --seed            "${SEED}" \
    --device          "${GPU}" \
    --device_ids      "${GPU}" \
    "${EXTRA_ARGS[@]}"
