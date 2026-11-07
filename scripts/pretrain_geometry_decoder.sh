#!/usr/bin/env bash
# Two-stage recipe; full training is opt-in by running this script.
# The validation report uses a separate 32/8/8-image smoke dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
export FASTPOSE_DEVICE="${FASTPOSE_DEVICE:-npu:0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH=.
PYTHON="${PYTHON:-.venv-npu/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets/SPEED_FIXED}"
RUN_DIR="${RUN_DIR:-experiments/geometry_dual_token}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
mkdir -p "$RUN_DIR"
phase=pretraining
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "failed phase=%s exit_code=%s\n" "$phase" "$code" > "$RUN_DIR/status.txt"; fi' EXIT
printf 'running phase=pretraining\n' > "$RUN_DIR/status.txt"
common=(--dataset SPEED --dataset_root_dir "$DATASET_ROOT"
    --vit_model vit_b_16_no_weights --pose_decoder geometry_dual_token
    --rotation_format matrix --merge_outputs --no_mlp --dropout_rate 0
    --decoder_dim 256 --decoder_heads 4 --decoder_layers 2
    --batch_size 4 --num_workers 2 --optimizer adam --scheduler cosineannealinglr
    --translation_loss relative_translation --rotation_loss 6d_relative_frobenius
    --skip_tensorboard --max_grad_norm 1.0)
"$PYTHON" -m src.train "${common[@]}" \
    --model_weights weights/vit_b_16.pth --init_backbone_only --freeze_backbone \
    --epochs "$PRETRAIN_EPOCHS" --max_lr 0.0003 --min_lr 0.00001 \
    --model_save_path "$RUN_DIR/pretrained.pth" --log_dir "$RUN_DIR/pretrain_logs" 2>&1 | tee "$RUN_DIR/pretrain.log"
phase=joint_training
printf 'running phase=joint_training\n' > "$RUN_DIR/status.txt"
"$PYTHON" -m src.train "${common[@]}" \
    --model_weights "$RUN_DIR/pretrained.pth" \
    --epochs "$FINETUNE_EPOCHS" --max_lr 0.00001 --min_lr 0.000001 \
    --model_save_path "$RUN_DIR/finetuned.pth" --log_dir "$RUN_DIR/finetune_logs" 2>&1 | tee "$RUN_DIR/finetune.log"
printf 'completed\n' > "$RUN_DIR/status.txt"
