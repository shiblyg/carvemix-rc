import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def remove_small_components(mask, min_voxels):
    if min_voxels <= 0:
        return mask
    try:
        import cc3d
        components = cc3d.connected_components(mask.astype(np.uint8))
    except ImportError:
        from scipy import ndimage
        components, _ = ndimage.label(mask)
    out = np.zeros_like(mask, dtype=bool)
    for index in range(1, int(components.max()) + 1):
        component = components == index
        if component.sum() >= min_voxels:
            out[component] = True
    return out


def fill_holes(mask):
    from scipy import ndimage
    return ndimage.binary_fill_holes(mask)


def postprocess_case(label_map, rc_prob, rc_min_voxels, et_min_voxels,
                     rc_prob_thresh, fill_rc_holes, rc_label, et_label):
    out = label_map.copy()
    rc_mask = out == rc_label
    if rc_prob is not None and rc_prob_thresh is not None:
        rc_mask = rc_mask | (rc_prob > rc_prob_thresh)
    rc_mask = remove_small_components(rc_mask, rc_min_voxels)
    if fill_rc_holes and rc_mask.any():
        rc_mask = fill_holes(rc_mask)

    et_mask = remove_small_components(out == et_label, et_min_voxels)

    out[out == rc_label] = 0
    out[(label_map == et_label) & ~et_mask] = 0
    out[rc_mask] = rc_label
    return out


def run(pred_dir, output_dir, npz_dir, rc_min_voxels, et_min_voxels,
        rc_prob_thresh, fill_rc_holes, rc_label, et_label):
    pred_dir, output_dir = Path(pred_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(npz_dir) if npz_dir else None

    for path in sorted(pred_dir.glob("*.nii.gz")):
        case_id = path.name.replace(".nii.gz", "")
        image = nib.load(str(path))
        label_map = image.get_fdata().astype(np.int16)

        rc_prob = None
        if npz_dir is not None and rc_prob_thresh is not None:
            npz_path = npz_dir / f"{case_id}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                key = "probabilities" if "probabilities" in data else data.files[0]
                rc_prob = data[key][rc_label]

        result = postprocess_case(label_map, rc_prob, rc_min_voxels,
                                  et_min_voxels, rc_prob_thresh, fill_rc_holes,
                                  rc_label, et_label)
        nib.save(nib.Nifti1Image(result.astype(np.int16), image.affine, image.header),
                 str(output_dir / path.name))


def main():
    parser = argparse.ArgumentParser(description="RC-focused dust post-processing.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--npz-dir", default=None)
    parser.add_argument("--rc-min-voxels", type=int, default=50)
    parser.add_argument("--et-min-voxels", type=int, default=15)
    parser.add_argument("--rc-prob-thresh", type=float, default=None)
    parser.add_argument("--fill-rc-holes", type=int, default=1)
    parser.add_argument("--rc-label", type=int, default=4)
    parser.add_argument("--et-label", type=int, default=3)
    args = parser.parse_args()
    run(args.pred_dir, args.output_dir, args.npz_dir, args.rc_min_voxels,
        args.et_min_voxels, args.rc_prob_thresh, bool(args.fill_rc_holes),
        args.rc_label, args.et_label)


if __name__ == "__main__":
    main()
