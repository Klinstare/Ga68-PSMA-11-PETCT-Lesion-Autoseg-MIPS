#!/usr/bin/env bash
# train.sh — train the Stage-II cascade (5-fold 3d_fullres) with a chosen trainer.
#
# Usage:
#   PSMA_ROOT=/path/to/project \
#   PRIOR_CHANNELS=2,3 \                       # match the built dataset's prior channels
#   GPUS=0,1,2,3 \                             # GPUs for folds 0..3 (fold 4 on first GPU)
#   bash scripts/train.sh <DS_ID> <TRAINER>
#
# Example (Stage II + CT):
#   PSMA_ROOT=$PWD PRIOR_CHANNELS=2,3 GPUS=0,1,2,3 \
#   bash scripts/train.sh 560 nnUNetTrainerFTPD_PETCT_d50_b80
#
# The trainer classes live in nnunet_trainers/ and must be importable by
# nnU-Net (see README "Installing the trainers").
set -euo pipefail
DS_ID="${1:?usage: train.sh <DS_ID> <TRAINER>}"
TRAINER="${2:?usage: train.sh <DS_ID> <TRAINER>}"
: "${PSMA_ROOT:?set PSMA_ROOT to the project root}"
export nnUNet_raw="$PSMA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PSMA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PSMA_ROOT/nnUNet_results"
# PRIOR_CHANNELS / TVERSKY_* / PRIOR_DROPOUT_P are read by the trainer if set.

IFS=',' read -r -a G <<< "${GPUS:-0}"
NG=${#G[@]}
LOG="$PSMA_ROOT/logs"; mkdir -p "$LOG"

# folds 0..3 in parallel across available GPUs
pids=()
for f in 0 1 2 3; do
  g=${G[$(( f % NG ))]}
  echo "fold $f -> GPU $g"
  CUDA_VISIBLE_DEVICES="$g" nohup nnUNetv2_train "$DS_ID" 3d_fullres "$f" -tr "$TRAINER" \
      > "$LOG/train_ds${DS_ID}_${TRAINER}_fold${f}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
# fold 4 on the first GPU
echo "fold 4 -> GPU ${G[0]}"
CUDA_VISIBLE_DEVICES="${G[0]}" nnUNetv2_train "$DS_ID" 3d_fullres 4 -tr "$TRAINER" \
    > "$LOG/train_ds${DS_ID}_${TRAINER}_fold4.log" 2>&1
echo "training complete: dataset $DS_ID trainer $TRAINER"
