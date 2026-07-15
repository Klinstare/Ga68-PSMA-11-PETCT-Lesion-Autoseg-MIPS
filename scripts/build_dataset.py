#!/usr/bin/env python3
"""
build_dataset.py — assemble a multi-channel nnU-Net v2 dataset for the Stage-II
lesion-segmentation network from PET (optionally + CT) and the continuous soft
priors produced by make_soft_priors.py.

Channel layouts (--modality x --priors):
  pet   + slabma  -> [PET, soft_slab, soft_MA]          (3ch)   priors (1,2)
  petct + slabma  -> [PET, CT, soft_slab, soft_MA]      (4ch)   priors (2,3)   <- Stage II + CT
  pet   + maonly  -> [PET, soft_MA]                      (2ch)   priors (1,)
  petct + maonly  -> [PET, CT, soft_MA]                  (3ch)   priors (2,)
  pet   + none    -> [PET]                               (1ch)   PET-only baseline
  petct + none    -> [PET, CT]                           (2ch)   PET/CT baseline

Expected data layout (per case NAME), under <PSMA_ROOT>/Data_Origin/Data_PETSegm:
  <name>/PET/PP_<name>_PET.nii.gz
  <name>/CT/<name>_CT.nii.gz               (only needed for --modality petct)
  <name>/GroundTruth/PP_<name>_Segm.nii.gz
Soft priors from make_soft_priors.py under <PSMA_ROOT>/Predictions/<priors-dir>.

The split json (--split-json) maps case NAME -> integer id; no patient-identifying
map is shipped (see config_template/split_template.json).

Released with the manuscript. No data or identifiers are included in this repo.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

ROOT   = Path(os.environ.get("PSMA_ROOT", ".")).resolve()
RAW    = ROOT/"Data_Origin/Data_PETSegm"
NNRAW  = ROOT/"nnUNet_raw"


def _resample_to(ref, img):
    if img.GetSize() != ref.GetSize():
        r = sitk.ResampleImageFilter(); r.SetReferenceImage(ref)
        r.SetInterpolator(sitk.sitkLinear); r.SetDefaultPixelValue(0.0)
        img = r.Execute(img)
    img.CopyInformation(ref)     # enforce identical header
    return img


def channel_plan(modality, priors, priors_dir):
    """Return (chan_specs, chan_names, prior_channel_indices).
    chan_specs: list of (channel_index, kind, path_template_key)."""
    ch = 0
    specs, names, prior_idx = [], {}, []
    # channel 0 = PET (always)
    specs.append((ch, "pet", None)); names[str(ch)] = "PET_SUV"; ch += 1
    if modality == "petct":
        specs.append((ch, "ct", None)); names[str(ch)] = "CT"; ch += 1
    if priors in ("slabma",):
        specs.append((ch, "slab", "soft_slab")); names[str(ch)] = "soft_slab_prob"; prior_idx.append(ch); ch += 1
        specs.append((ch, "ma", "soft_ma"));     names[str(ch)] = "soft_MA_prob";   prior_idx.append(ch); ch += 1
    elif priors in ("maonly",):
        specs.append((ch, "ma", "soft_ma"));     names[str(ch)] = "soft_MA_prob";   prior_idx.append(ch); ch += 1
    return specs, names, prior_idx


def main(split_json, ds_id, suffix, modality, priors, priors_dir):
    PRIORS = Path(priors_dir)
    if not PRIORS.is_absolute():
        PRIORS = ROOT/priors_dir
    NM = json.load(open(split_json))
    specs, chan_names, prior_idx = channel_plan(modality, priors, PRIORS)

    ds_name = f"Dataset{ds_id:03d}_{suffix}"
    ds = NNRAW/ds_name
    for sub in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        (ds/sub).mkdir(parents=True, exist_ok=True)

    n_train = n_test = skipped = 0
    for split in ["train", "test"]:
        img_sub = "imagesTr" if split == "train" else "imagesTs"
        lbl_sub = "labelsTr" if split == "train" else "labelsTs"
        for name in tqdm(sorted(NM[split]), desc=f"{suffix} {split}"):
            cid = int(NM[split][name]); cid3 = f"{cid:03d}"
            paths = {
                "pet":  RAW/name/"PET"/f"PP_{name}_PET.nii.gz",
                "ct":   RAW/name/"CT"/f"{name}_CT.nii.gz",
                "gt":   RAW/name/"GroundTruth"/f"PP_{name}_Segm.nii.gz",
                "slab": PRIORS/f"soft_slab_{name}.nii.gz",
                "ma":   PRIORS/f"soft_ma_{name}.nii.gz",
            }
            # required path keys = {gt} + one per input channel (kind is the paths[] key)
            need = list(dict.fromkeys(["gt"] + [kind for _, kind, _ in specs]))
            miss = [k for k in need if not paths[k].exists()]
            if miss:
                skipped += 1; print(f"  [SKIP] {name}: missing {miss}", flush=True); continue

            pet = sitk.ReadImage(str(paths["pet"]))
            prefix = f"{suffix}_{cid3}"
            for ci, kind, tmpl in specs:
                if kind == "pet":
                    img = pet
                elif kind == "ct":
                    img = _resample_to(pet, sitk.ReadImage(str(paths["ct"])))
                else:  # slab / ma prior
                    img = _resample_to(pet, sitk.ReadImage(str(paths[kind])))
                sitk.WriteImage(img, str(ds/img_sub/f"{prefix}_{ci:04d}.nii.gz"))

            gt_arr = (sitk.GetArrayFromImage(sitk.ReadImage(str(paths["gt"]))) != 0).astype(np.uint8)
            go = sitk.GetImageFromArray(gt_arr); go.CopyInformation(pet)
            sitk.WriteImage(go, str(ds/lbl_sub/f"{prefix}.nii.gz"))

            if split == "train": n_train += 1
            else: n_test += 1

    dj = {"channel_names": chan_names,
          "labels": {"background": 0, "lesion": 1},
          "numTraining": n_train, "file_ending": ".nii.gz", "name": ds_name,
          "description": f"Stage-II cascade — modality={modality} priors={priors}; prior_channels={tuple(prior_idx)}"}
    json.dump(dj, open(ds/"dataset.json", "w"), indent=2)
    print(f"\nDONE {ds_name}: modality={modality} priors={priors} "
          f"channels={list(chan_names.values())} prior_channels={tuple(prior_idx)} "
          f"train={n_train} test={n_test} skipped={skipped}")
    print(f"  -> set PRIOR_CHANNELS={','.join(map(str, prior_idx))} when training (if any priors).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--ds-id", type=int, required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--modality", choices=["pet", "petct"], default="pet")
    ap.add_argument("--priors", choices=["slabma", "maonly", "none"], default="slabma")
    ap.add_argument("--priors-dir", default="Predictions/soft_priors_or/all")
    a = ap.parse_args()
    main(a.split_json, a.ds_id, a.suffix, a.modality, a.priors, a.priors_dir)
