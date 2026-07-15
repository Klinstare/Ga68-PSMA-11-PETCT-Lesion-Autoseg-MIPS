
# 68Ga-PSMA-11-PET-CT_lesion_Auto-segmentation

Training code for the manuscript **"Maximum-intensity-projection guided 3D
characterization of whole-body metastatic lesions on [68Ga]Ga-PSMA-11 PET
imaging."**
It implements the **Stage-II** multi-channel lesion-segmentation network of a
MIP-guided cascade for automated PSMA PET/CT lesion detection:

- **Stage I** (standard nnU-Net, not the focus here) predicts lesion foreground
  in multi-view **slab** projections (coronal / sagittal / axial) and in 12
  **multi-angle (MA)** rotational projections (30° increments, 0–330°).
- Those 2D probabilities are back-projected and fused into two continuous 3D
  **soft priors** (`soft_slab`, `soft_MA`).
- **Stage II** is a 3D `nnU-Net` that takes **PET (optionally + CT)** stacked
  with the soft priors as extra input channels, trained with a **Focal-Tversky
  + CE** loss and **prior-channel dropout** to keep lesion recall high while the
  priors suppress physiologic false positives.

> **Data sharing note.** This repository contains **code only**. It ships **no
> imaging data, no predictions, and no patient-identifying name maps.** You
> supply your own data and a non-identifying split file (see
> `config_template/`).

## What's included

```
nnunet_trainers/
  nnUNetTrainerFocalTverskyPriorDropout.py   # Focal-Tversky+CE loss + prior dropout
  ftpd_variants.py                           # named hyperparameter/channel variants
scripts/
  make_soft_priors.py                        # fuse Stage-I probs -> soft priors (avg | OR)
  build_dataset.py                           # assemble PET(+CT)+priors nnU-Net dataset
  preprocess.sh                              # nnU-Net plan + preprocess
  train.sh                                   # 5-fold 3d_fullres training driver
config_template/
  split_template.json                        # case-name -> id map (fill in your own)
  DATA_LAYOUT.md                             # expected directory layout
requirements.txt
```

## Install

```bash
pip install -r requirements.txt        # plus nnU-Net v2 per its own instructions
```

**Installing the trainers** so nnU-Net can find them by name: copy both files in
`nnunet_trainers/` onto nnU-Net's trainer discovery path, e.g.

```bash
NNUNET=$(python -c 'import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))')
cp nnunet_trainers/*.py "$NNUNET/training/nnUNetTrainer/variants/loss/"
```

Set the project root once per shell:

```bash
export PSMA_ROOT=/path/to/your/project    # holds Data_Origin/, nnUNet_raw/, ...
```

## Pipeline

```bash
# 1. Fuse Stage-I probabilities into continuous soft priors (soft-OR recommended)
python scripts/make_soft_priors.py --split-json config_template/split_template.json --fusion or

# 2a. Build a Stage-II dataset — PET + soft priors (3 channels)
python scripts/build_dataset.py --split-json config_template/split_template.json \
    --ds-id 555 --suffix PETSlabMA --modality pet --priors slabma

# 2b. Build a Stage-II dataset WITH CT — PET + CT + soft priors (4 channels)
python scripts/build_dataset.py --split-json config_template/split_template.json \
    --ds-id 560 --suffix PETCTSlabMA --modality petct --priors slabma

# 3. Preprocess
PSMA_ROOT=$PSMA_ROOT bash scripts/preprocess.sh 560

# 4. Train (PRIOR_CHANNELS must match the built dataset — build_dataset.py prints it)
PSMA_ROOT=$PSMA_ROOT PRIOR_CHANNELS=2,3 GPUS=0,1,2,3 \
    bash scripts/train.sh 560 nnUNetTrainerFTPD_PETCT_d50_b80
```

## Channel layouts and matching trainers

| `--modality` | `--priors` | input channels | `PRIOR_CHANNELS` | trainer |
|---|---|---|---|---|
| `pet` | `slabma` | PET, slab, MA | `1,2` | `nnUNetTrainerFTPD_d50_b80` |
| **`petct`** | **`slabma`** | **PET, CT, slab, MA** | **`2,3`** | **`nnUNetTrainerFTPD_PETCT_d50_b80`** |
| `pet` | `maonly` | PET, MA | `1` | `nnUNetTrainerFTPD_MAonly_d50_b80` |
| `petct` | `maonly` | PET, CT, MA | `2` | `nnUNetTrainerFTPD_MAonly_PETCT_d50_b80` |
| `pet`/`petct` | `none` | PET (+CT) | — | `nnUNetTrainer` (baseline) |

Loss/dropout are also overridable via env vars: `TVERSKY_ALPHA`, `TVERSKY_BETA`
(recall weight; β>α penalises misses), `TVERSKY_GAMMA`, `PRIOR_DROPOUT_P`.

## Soft-prior fusion

`make_soft_priors.py --fusion`:
- `avg` — slab = weighted sum (0.4/0.4/0.2), MA = mean over angles.
- `or`  — element-wise **max** across views/angles (soft-OR). Preserves lesions
  visible in only one projection and is recommended for recall.

## Citation

If you use this code, please cite the manuscript *(details to be added on
acceptance)*.

