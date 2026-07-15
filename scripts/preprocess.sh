#!/usr/bin/env bash
# preprocess.sh — nnU-Net v2 planning + preprocessing for a built dataset.
# Usage: PSMA_ROOT=/path/to/project bash scripts/preprocess.sh <DS_ID>
set -euo pipefail
DS_ID="${1:?usage: preprocess.sh <DS_ID>}"
: "${PSMA_ROOT:?set PSMA_ROOT to the project root}"
export nnUNet_raw="$PSMA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PSMA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PSMA_ROOT/nnUNet_results"

nnUNetv2_plan_and_preprocess -d "$DS_ID" -c 3d_fullres --verify_dataset_integrity
echo "preprocessed dataset $DS_ID"
