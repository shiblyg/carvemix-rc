import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


def list_cases(labels_dir):
    return {p.name.replace(".nii.gz", ""): p
            for p in sorted(Path(labels_dir).glob("*.nii.gz"))}


def has_rc(label_path, rc_label=4, min_voxels=50):
    data = nib.load(str(label_path)).get_fdata()
    return int((data == rc_label).sum()) >= min_voxels


def carve_mask(rc_mask, lam):
    inside = ndimage.distance_transform_edt(rc_mask)
    outside = ndimage.distance_transform_edt(~rc_mask)
    return (outside - inside) < lam


def harmonize_intensity(source, target_stats):
    s_mean, s_std = source.mean(), source.std() + 1e-6
    t_mean, t_std = target_stats
    return (source - s_mean) / s_std * t_std + t_mean


def load_modalities(images_dir, case_id, n_mod):
    images, ref = [], None
    for m in range(n_mod):
        nii = nib.load(str(Path(images_dir) / f"{case_id}_{m:04d}.nii.gz"))
        ref = ref or nii
        images.append(nii.get_fdata().astype(np.float32))
    return np.stack(images, axis=0), ref


def generate_one(source_id, target_id, images_dir, labels_dir, rc_label, n_mod,
                 lam_range):
    src_images, _ = load_modalities(images_dir, source_id, n_mod)
    src_label = nib.load(str(Path(labels_dir) / f"{source_id}.nii.gz")).get_fdata()
    src_rc = src_label == rc_label

    tgt_images, tgt_ref = load_modalities(images_dir, target_id, n_mod)
    tgt_label = nib.load(str(Path(labels_dir) / f"{target_id}.nii.gz")).get_fdata()

    lam = np.random.uniform(*lam_range)
    carved = carve_mask(src_rc, lam)
    if carved.sum() == 0:
        return None

    tgt_shape = tgt_images.shape[1:]
    coords = np.argwhere(carved)
    lo, hi = coords.min(0), coords.max(0) + 1
    box = tuple(slice(lo[i], hi[i]) for i in range(3))
    box_shape = tuple(hi[i] - lo[i] for i in range(3))

    if any(box_shape[i] > tgt_shape[i] for i in range(3)):
        return None

    carved_box = carved[box]
    start = [np.random.randint(0, tgt_shape[i] - box_shape[i] + 1) for i in range(3)]
    dst = tuple(slice(start[i], start[i] + box_shape[i]) for i in range(3))

    mixed = tgt_images.copy()
    mixed_label = tgt_label.copy()
    for m in range(n_mod):
        region = mixed[m][dst]
        source = src_images[m][box]
        local = region[carved_box]
        stats = (local.mean(), local.std() + 1e-6) if local.size else \
                (mixed[m].mean(), mixed[m].std() + 1e-6)
        region[carved_box] = harmonize_intensity(source[carved_box], stats)
        mixed[m][dst] = region

    label_region = mixed_label[dst]
    label_region[carved_box] = rc_label
    mixed_label[dst] = label_region

    return mixed, mixed_label, tgt_ref


def run(images_dir, labels_dir, out_images, out_labels, n_generate,
        rc_label, n_mod, lam_range, min_donor_voxels,
        prefer_rc_negative_targets, seed):
    np.random.seed(seed)
    Path(out_images).mkdir(parents=True, exist_ok=True)
    Path(out_labels).mkdir(parents=True, exist_ok=True)

    cases = list_cases(labels_dir)
    donors = [c for c, p in cases.items() if has_rc(p, rc_label, min_donor_voxels)]
    targets_neg = [c for c in cases if c not in donors]
    if not donors:
        raise RuntimeError("No RC-positive donor cases found.")
    target_pool = targets_neg if (prefer_rc_negative_targets and targets_neg) \
        else list(cases)

    manifest, made, attempts = [], 0, 0
    while made < n_generate and attempts < n_generate * 5:
        attempts += 1
        source_id = np.random.choice(donors)
        target_id = np.random.choice(target_pool)
        if source_id == target_id:
            continue
        result = generate_one(source_id, target_id, images_dir, labels_dir,
                              rc_label, n_mod, lam_range)
        if result is None:
            continue
        mixed, mixed_label, ref = result
        case_id = f"CarveMixRC_{made:04d}"
        for m in range(n_mod):
            nib.save(nib.Nifti1Image(mixed[m].astype(np.float32), ref.affine, ref.header),
                     str(Path(out_images) / f"{case_id}_{m:04d}.nii.gz"))
        nib.save(nib.Nifti1Image(mixed_label.astype(np.uint8), ref.affine, ref.header),
                 str(Path(out_labels) / f"{case_id}.nii.gz"))
        manifest.append({"new_id": case_id, "source": source_id,
                         "target": target_id,
                         "rc_voxels": int((mixed_label == rc_label).sum())})
        made += 1

    manifest_path = Path(out_labels).parent / "CarveMixRC_manifest.csv"
    with open(manifest_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["new_id", "source", "target", "rc_voxels"])
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {made} cases ({attempts} attempts). Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="RC-aware CarveMix augmentation.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--n-generate", type=int, default=300)
    parser.add_argument("--rc-label", type=int, default=4)
    parser.add_argument("--n-modalities", type=int, default=4)
    parser.add_argument("--lambda-low", type=float, default=-3.0)
    parser.add_argument("--lambda-high", type=float, default=5.0)
    parser.add_argument("--min-donor-voxels", type=int, default=50)
    parser.add_argument("--prefer-rc-negative-targets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args.images_dir, args.labels_dir, args.out_images, args.out_labels,
        args.n_generate, args.rc_label, args.n_modalities,
        (args.lambda_low, args.lambda_high), args.min_donor_voxels,
        bool(args.prefer_rc_negative_targets), args.seed)


if __name__ == "__main__":
    main()
