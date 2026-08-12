# -*- coding: utf-8 -*-
"""
FP-aware nnU-Net ensemble with orientation-safe export.
====================================================
Handles the axis-transpose issue: nnUNet .npz softmax is in nnUNet's internal
(C, X, Y, Z) order with a per-case crop/transpose recorded in the sibling
.pkl. Hand-saving the argmax array with the reference affine gave transposed
output (e.g. 155x240x240 vs reference 240x240x155).

This implementation uses nnU-Net's native export pipeline, which reads the .pkl
properties and writes a correctly-oriented NIfTI matching the reference.

Fusion (unchanged, FP-aware):
  - tumor (NETC, SNFH, ET): average softmax
  - RC: agreement (both models must exceed threshold) -> suppress FPs
  - then RC dust + hole fill, ET dust

USAGE:
  python -m src.postprocess.fp_aware_ensemble \
      --npz-dir-a /path/predictions_rcweighted_resencl_137 \
      --npz-dir-b /path/predictions_carvemix_standard_138 \
      --output-dir /path/ensemble_out \
      --dataset-json /path/Dataset137.../dataset.json \
      --plans-json   /path/Dataset137.../plans.json \
      --rc-dust 50 --et-dust 15
"""
import argparse
import pickle
from pathlib import Path
import numpy as np


def cc_remove_small(mask, min_voxels):
    if min_voxels <= 0:
        return mask
    try:
        import cc3d
        cc = cc3d.connected_components(mask.astype(np.uint8))
        out = np.zeros_like(mask, dtype=bool)
        for l in range(1, int(cc.max()) + 1):
            c = cc == l
            if c.sum() >= min_voxels:
                out[c] = True
        return out
    except ImportError:
        from scipy import ndimage
        cc, n = ndimage.label(mask)
        out = np.zeros_like(mask, dtype=bool)
        for i in range(1, n + 1):
            c = cc == i
            if c.sum() >= min_voxels:
                out[c] = True
        return out


def fill_holes(mask):
    from scipy import ndimage
    return ndimage.binary_fill_holes(mask)


def fuse_logits(pa, pb, rc_label=4, rc_thresh=0.5, rc_mode='agreement'):
    """
    Returns a fused SEGMENTATION (integer labels) in the SAME axis order
    as the input arrays (C, X, Y, Z) -> (X, Y, Z). Orientation correction
    happens later via nnUNet export.
    """
    avg = 0.5 * pa + 0.5 * pb
    seg = np.argmax(avg, axis=0).astype(np.uint8)

    rc_a = pa[rc_label] > rc_thresh
    rc_b = pb[rc_label] > rc_thresh
    if rc_mode == 'agreement':
        rc_mask = rc_a & rc_b
    elif rc_mode == 'union':
        rc_mask = rc_a | rc_b
    else:
        rc_mask = avg[rc_label] > rc_thresh

    return seg, rc_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz-dir-a', required=True)
    ap.add_argument('--npz-dir-b', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--dataset-json', required=True)
    ap.add_argument('--plans-json', required=True)
    ap.add_argument('--rc-mode', default='agreement',
                    choices=['agreement', 'union', 'average'])
    ap.add_argument('--rc-thresh', type=float, default=0.5)
    ap.add_argument('--rc-dust', type=int, default=50)
    ap.add_argument('--et-dust', type=int, default=15)
    ap.add_argument('--rc-label', type=int, default=4)
    ap.add_argument('--et-label', type=int, default=3)
    args = ap.parse_args()

    # nnUNet imports (run inside the nnunet env)
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    import json

    a_dir = Path(args.npz_dir_a)
    b_dir = Path(args.npz_dir_b)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_json = json.load(open(args.dataset_json))
    plans = json.load(open(args.plans_json))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')

    a_files = sorted(a_dir.glob("*.npz"))
    print(f"Ensembling {len(a_files)} cases | RC mode {args.rc_mode} | dust {args.rc_dust}")

    done = 0
    for npz_a in a_files:
        cid = npz_a.stem
        npz_b = b_dir / f"{cid}.npz"
        pkl_a = a_dir / f"{cid}.pkl"
        if not npz_b.exists() or not pkl_a.exists():
            print(f"  SKIP {cid}: missing pair or .pkl")
            continue

        da = np.load(npz_a)
        db = np.load(npz_b)
        keya = 'probabilities' if 'probabilities' in da else da.files[0]
        keyb = 'probabilities' if 'probabilities' in db else db.files[0]
        pa = da[keya]
        pb = db[keyb]
        if pa.shape != pb.shape:
            print(f"  SKIP {cid}: softmax shape mismatch {pa.shape} vs {pb.shape}")
            continue

        # fuse -> build a fused PROBABILITY volume so nnUNet can export it
        avg = 0.5 * pa + 0.5 * pb

        # apply agreement RC by zeroing RC prob where models disagree
        if args.rc_mode == 'agreement':
            rc_a = pa[args.rc_label] > args.rc_thresh
            rc_b = pb[args.rc_label] > args.rc_thresh
            disagree = ~(rc_a & rc_b)
            avg[args.rc_label][disagree] = 0.0

        # load properties (.pkl) -- has the original shape/crop/transpose
        with open(pkl_a, 'rb') as f:
            props = pickle.load(f)

        # Let nnUNet export the correctly-oriented segmentation.
        # export_prediction_from_logits applies argmax + inverse transforms.
        export_prediction_from_logits(
            avg.astype(np.float32), props, configuration_manager,
            plans_manager, dataset_json, str(out_dir / cid),
            save_probabilities=False
        )

        # post-hoc dust/holes on the exported nii
        import nibabel as nib
        nii_path = out_dir / f"{cid}.nii.gz"
        if nii_path.exists():
            nimg = nib.load(str(nii_path))
            seg = nimg.get_fdata().astype(np.uint8)
            # RC dust + holes
            rc = seg == args.rc_label
            rc = cc_remove_small(rc, args.rc_dust)
            if rc.any():
                rc = fill_holes(rc)
            # ET dust
            et = seg == args.et_label
            et_clean = cc_remove_small(et, args.et_dust)
            seg[(seg == args.et_label) & ~et_clean] = 0
            seg[seg == args.rc_label] = 0
            seg[rc] = args.rc_label
            nib.save(nib.Nifti1Image(seg, nimg.affine, nimg.header), str(nii_path))

        done += 1
        if done % 25 == 0:
            print(f"  {done} done...")

    print(f"\nDone: {done} cases -> {out_dir}")


if __name__ == '__main__':
    main()