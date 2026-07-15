#!/usr/bin/env python3
"""
make_soft_priors.py — build continuous soft-prior 3D volumes for the Stage-II
cascade from Stage-I 2D model probabilities.

For each case it writes two float NIfTI volumes into <PSMA_ROOT>/Predictions/<out>/all:
  soft_slab_<name>.nii.gz  — fusion across the coronal/sagittal/axial slab models
                             of per-view max-pooled foreground probabilities,
                             back-projected over each slab's own depth range.
  soft_ma_<name>.nii.gz    — fusion over the multi-angle (MA) model of
                             back-projected foreground probability, each angle
                             rotated back into the reference frame.

Fusion (--fusion):
  avg   slab = weighted sum (0.4 cor / 0.4 sag / 0.2 ax); MA = mean over angles.
  or    slab = max over views; MA = max over angles (soft-OR). Preserves lesions
        visible in only one view/angle; recommended for recall.

Inputs (produced by nnU-Net Stage-I inference with --save_probabilities):
  <PSMA_ROOT>/Predictions/<SlabDir>_<split>_prob/<file_key>.npz   (per slab)
  <PSMA_ROOT>/Predictions/<MADir>_<split>_prob/<PREFIX>_<cid>a<ai>.npz (per angle)
  <PSMA_ROOT>/nnUNet_raw/<SlabDataset>/slabs.json                 (slab metadata)

The split json maps case NAME -> integer case id, e.g.
  {"train": {"CASE_0001": 1, ...}, "test": {"CASE_0301": 301, ...}}
See config_template/split_template.json. No patient-identifying map is shipped.

CPU only. Reference implementation released with the manuscript.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

ROOT = Path(os.environ.get("PSMA_ROOT", ".")).resolve()
RAW  = ROOT/"Data_Origin/Data_PETSegm"     # <ROOT>/Data_Origin/Data_PETSegm/<name>/PET/PP_<name>_PET.nii.gz
PRED = ROOT/"Predictions"

# --- Multi-angle geometry (inlined so this script is self-contained) ---------
# 12 rotational projection angles, 30-degree increments, spanning 0..330 deg.
ANGLES_12     = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
MA_PREFIX     = "MultiAngle"
MA_PROB_DIR   = "MA_prob"      # -> Predictions/MA_prob_<split>_prob

# --- Slab views: (prob-dir prefix, slab-dataset glob, numpy axis, avg weight) -
SLAB_VIEWS = [
    ("SlabCor", "Dataset*_SlabCor", 1, 0.4),
    ("SlabSag", "Dataset*_SlabSag", 2, 0.4),
    ("SlabAx",  "Dataset*_SlabAx",  0, 0.2),
]


def _rotate_sitk_z(sitk_img, angle_deg, interpolator=sitk.sitkLinear, default_value=0.0):
    """Rotate a 3D SimpleITK image around the Z axis by angle_deg degrees."""
    if angle_deg == 0:
        return sitk_img
    size = sitk_img.GetSize()
    center = sitk_img.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2.0 for s in size])
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0.0, 0.0, np.deg2rad(angle_deg))
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(sitk_img); r.SetTransform(transform)
    r.SetInterpolator(interpolator); r.SetDefaultPixelValue(default_value)
    return r.Execute(sitk_img)


def refp(name): return RAW/name/"PET"/f"PP_{name}_PET.nii.gz"


def load_fg(npz_path, target_hw):
    p = np.load(npz_path)["probabilities"][1, 0].astype(np.float32)  # foreground (H,W)
    if p.shape != target_hw:
        p = zoom(p, [target_hw[0]/p.shape[0], target_hw[1]/p.shape[1]], order=1)
    return p


def soft_ma(ref_img, cid3, split, fusion="avg"):
    Z, Y, X = sitk.GetArrayFromImage(ref_img).shape
    accum = np.zeros((Z, Y, X), np.float32); n = 0
    pdir = PRED/f"{MA_PROB_DIR}_{split}_prob"
    for ai, ang in enumerate(ANGLES_12):
        f = pdir/f"{MA_PREFIX}_{cid3}a{ai}.npz"
        if not f.exists():
            continue
        p2d = load_fg(f, (Z, X))
        vol = np.repeat(p2d[:, None, :], Y, axis=1)
        vimg = sitk.GetImageFromArray(vol); vimg.CopyInformation(ref_img)
        va = sitk.GetArrayFromImage(_rotate_sitk_z(vimg, -ang)) if ang != 0 else vol
        if fusion == "or":
            np.maximum(accum, va, out=accum)
        else:
            accum += va
        n += 1
    out = accum if fusion == "or" else accum/max(n, 1)
    return out, n


def soft_slab(ref_img, cid, split, slab_meta_cache, fusion="avg"):
    Z, Y, X = sitk.GetArrayFromImage(ref_img).shape
    fused = np.zeros((Z, Y, X), np.float32); used = 0
    for prob_pref, ds_glob, axis, w in SLAB_VIEWS:
        slabs = slab_meta_cache[ds_glob]
        pdir = PRED/f"{prob_pref}_{split}_prob"
        view = np.zeros((Z, Y, X), np.float32)
        hw = (Z, X) if axis == 1 else (Z, Y) if axis == 2 else (Y, X)
        for key, meta in slabs.items():
            if meta.get("split") != split or meta["case_id"] != cid:
                continue
            npz = pdir/f'{meta["file_key"]}.npz'
            if not npz.exists():
                continue
            s0, s1 = meta["slab_start"], meta["slab_end"]; depth = s1 - s0
            p2d = load_fg(npz, hw)
            if axis == 0:
                tiled = np.repeat(p2d[None, :, :], depth, axis=0); np.maximum(view[s0:s1, :, :], tiled, out=view[s0:s1, :, :])
            elif axis == 1:
                tiled = np.repeat(p2d[:, None, :], depth, axis=1); np.maximum(view[:, s0:s1, :], tiled, out=view[:, s0:s1, :])
            else:
                tiled = np.repeat(p2d[:, :, None], depth, axis=2); np.maximum(view[:, :, s0:s1], tiled, out=view[:, :, s0:s1])
            used += 1
        if fusion == "or":
            np.maximum(fused, view, out=fused)
        else:
            fused += w * view
    return fused, used


def main(split_json, fusion, nshards, shard):
    tag = "soft_priors_or" if fusion == "or" else "soft_priors"
    outdir = PRED/tag/"all"; outdir.mkdir(parents=True, exist_ok=True)
    slab_meta_cache = {g: json.load(open(next((ROOT/"nnUNet_raw").glob(g))/"slabs.json")) for _, g, _, _ in SLAB_VIEWS}
    COH = json.load(open(split_json))
    loc = {name: (int(cid), mem) for mem in ("train", "test") for name, cid in COH[mem].items()}
    names = sorted(loc)
    for k, name in enumerate(names):
        if k % nshards != shard:
            continue
        out_slab = outdir/f"soft_slab_{name}.nii.gz"; out_ma = outdir/f"soft_ma_{name}.nii.gz"
        if out_slab.exists() and out_ma.exists():
            print(f"[{k+1}/{len(names)}] {name} already done, skip", flush=True); continue
        cid, mem = loc[name]; cid3 = f"{cid:03d}"
        rp = refp(name)
        if not rp.exists():
            print(f"[{k+1}] skip {name}: no PET at {rp}", flush=True); continue
        ref = sitk.ReadImage(str(rp))
        ma, n_ang = soft_ma(ref, cid3, mem, fusion)
        sl, n_slab = soft_slab(ref, cid, mem, slab_meta_cache, fusion)
        for arr, op in [(sl, out_slab), (ma, out_ma)]:
            img = sitk.GetImageFromArray(arr.astype(np.float32)); img.CopyInformation(ref)
            sitk.WriteImage(img, str(op))
        print(f"[{k+1}/{len(names)}] {name} ({mem} cid={cid3}, angles={n_ang}, slabs={n_slab}) "
              f"slab[max={sl.max():.3f}] ma[max={ma.max():.3f}]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-json", required=True, help="case-name -> id map (see config_template/split_template.json)")
    ap.add_argument("--fusion", choices=["avg", "or"], default="or")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    main(a.split_json, a.fusion, a.nshards, a.shard)
