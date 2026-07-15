# Expected data layout

This release ships **code only** — no imaging data and no patient-identifying
name maps. To run the pipeline on your own PSMA PET/CT data, arrange it as
follows under `$PSMA_ROOT`.

```
$PSMA_ROOT/
  Data_Origin/Data_PETSegm/
    <CASE_NAME>/
      PET/PP_<CASE_NAME>_PET.nii.gz            # SUV-normalised PET (channel 0)
      CT/<CASE_NAME>_CT.nii.gz                  # CT, only for --modality petct (channel 1)
      GroundTruth/PP_<CASE_NAME>_Segm.nii.gz    # binary lesion label
  nnUNet_raw/            # nnU-Net v2 raw datasets (created by build_dataset.py)
  nnUNet_preprocessed/   # created by preprocess.sh
  nnUNet_results/        # created by train.sh
  Predictions/           # Stage-I probabilities + soft priors
    <SlabCor|SlabSag|SlabAx>_<train|test>_prob/   # nnU-Net --save_probabilities .npz
    MA_prob_<train|test>_prob/                    # multi-angle .npz
    soft_priors_or/all/                           # created by make_soft_priors.py
  nnUNet_raw/Dataset*_SlabCor/slabs.json          # slab depth metadata
```

## Split file

`build_dataset.py` and `make_soft_priors.py` take `--split-json`, a JSON mapping
**your** case folder names to unique integer ids:

```json
{"train": {"CASE_0001": 1, ...}, "test": {"CASE_0301": 301, ...}}
```

See `split_template.json`. Generate it from your own cohort; do **not** embed
patient identifiers you cannot share.

## Stage-I probabilities

The soft priors are fused from Stage-I 2D nnU-Net models run with
`nnUNetv2_predict ... --save_probabilities`, which writes per-slice `.npz`
files. `slabs.json` records each slab's `case_id`, `split`, `file_key`,
`slab_start`, `slab_end`. Training those Stage-I slab/multi-angle models uses
standard nnU-Net; only the Stage-II fusion + trainer are the novel components
released here.
