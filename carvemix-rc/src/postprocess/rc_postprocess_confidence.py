import argparse
import itertools
from pathlib import Path

import numpy as np
import nibabel as nib


def load_softmax_matched(npz_path, seg_shape):
    data = np.load(npz_path)
    key = "probabilities" if "probabilities" in data else data.files[0]
    softmax = data[key]
    spatial = softmax.shape[1:]
    if tuple(spatial) == tuple(seg_shape):
        return softmax
    for perm in itertools.permutations([1, 2, 3]):
        if tuple(softmax.shape[p] for p in perm) == tuple(seg_shape):
            return np.transpose(softmax, (0,) + perm)
    return None


def connected_components(mask):
    try:
        import cc3d
        return cc3d.connected_components(mask.astype(np.uint8))
    except ImportError:
        from scipy import ndimage
        components, _ = ndimage.label(mask)
        return components


def confidence_clean(seg, prob, label, size_thresh, conf_thresh, replace_with=0):
    mask = seg == label
    if not mask.any() or size_thresh <= 0:
        return seg
    components = connected_components(mask)
    out = seg.copy()
    for index in range(1, int(components.max()) + 1):
        component = components == index
        if component.sum() >= size_thresh:
            continue
        if prob is not None:
            if float(prob[component].mean()) < conf_thresh:
                out[component] = replace_with
        else:
            out[component] = replace_with
    return out


def fill_holes(seg, label):
    from scipy import ndimage
    mask = seg == label
    if not mask.any():
        return seg
    out = seg.copy()
    out[ndimage.binary_fill_holes(mask)] = label
    return out


def run(pred_dir, output_dir, rc_label, et_label,
        rc_size, rc_conf, et_size, et_conf, fill_rc):
    pred_dir, output_dir = Path(pred_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(pred_dir.glob("*.nii.gz")):
        case_id = path.name.replace(".nii.gz", "")
        image = nib.load(str(path))
        seg = image.get_fdata().astype(np.uint8)

        npz_path = pred_dir / f"{case_id}.npz"
        softmax = load_softmax_matched(npz_path, seg.shape) if npz_path.exists() else None
        prob_rc = softmax[rc_label] if softmax is not None else None
        prob_et = softmax[et_label] if softmax is not None else None

        seg = confidence_clean(seg, prob_et, et_label, et_size, et_conf)
        seg = confidence_clean(seg, prob_rc, rc_label, rc_size, rc_conf)
        if fill_rc:
            seg = fill_holes(seg, rc_label)

        nib.save(nib.Nifti1Image(seg, image.affine, image.header),
                 str(output_dir / path.name))


def main():
    parser = argparse.ArgumentParser(description="Confidence-aware post-processing.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rc-label", type=int, default=4)
    parser.add_argument("--et-label", type=int, default=3)
    parser.add_argument("--rc-size", type=int, default=75)
    parser.add_argument("--rc-conf", type=float, default=0.5)
    parser.add_argument("--et-size", type=int, default=15)
    parser.add_argument("--et-conf", type=float, default=0.5)
    parser.add_argument("--fill-rc", type=int, default=1)
    args = parser.parse_args()
    run(args.pred_dir, args.output_dir, args.rc_label, args.et_label,
        args.rc_size, args.rc_conf, args.et_size, args.et_conf, bool(args.fill_rc))


if __name__ == "__main__":
    main()
