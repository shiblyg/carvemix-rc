import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib


def label_components(mask):
    try:
        import cc3d
        return cc3d.connected_components(mask.astype(np.uint8))
    except ImportError:
        from scipy import ndimage
        components, _ = ndimage.label(mask)
        return components


def lesionwise_dice(pred_mask, gt_mask, min_gt_voxels=50):
    gt_cc = label_components(gt_mask)
    pred_cc = label_components(pred_mask)
    gt_labels = [l for l in range(1, int(gt_cc.max()) + 1)
                 if (gt_cc == l).sum() >= min_gt_voxels]
    pred_labels = list(range(1, int(pred_cc.max()) + 1))

    if not gt_labels and not pred_labels:
        return None
    if not gt_labels:
        return 0.0 if pred_labels else None

    dices, matched = [], set()
    for gl in gt_labels:
        g = gt_cc == gl
        overlap = pred_cc[g]
        overlap = overlap[overlap > 0]
        if len(overlap) == 0:
            dices.append(0.0)
            continue
        best = np.bincount(overlap).argmax()
        p = pred_cc == best
        matched.add(best)
        dices.append(2.0 * (g & p).sum() / (g.sum() + p.sum()))
    for pl in pred_labels:
        if pl not in matched:
            dices.append(0.0)
    return float(np.mean(dices))


def run(pred_dir, gt_dir, dust_thresholds, label):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    results = {}
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = gt_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            continue
        pred = nib.load(str(pred_path)).get_fdata().astype(np.int16)
        gt = nib.load(str(gt_path)).get_fdata().astype(np.int16)
        if pred.shape != gt.shape:
            continue
        pred_mask = pred == label
        gt_mask = gt == label
        for dust in dust_thresholds:
            cleaned = pred_mask
            if dust > 0:
                components = label_components(pred_mask)
                cleaned = np.zeros_like(pred_mask)
                for index in range(1, int(components.max()) + 1):
                    component = components == index
                    if component.sum() >= dust:
                        cleaned[component] = True
            score = lesionwise_dice(cleaned, gt_mask)
            if score is not None:
                results.setdefault(dust, []).append(score)

    summary = [(dust, float(np.mean(scores)), len(scores))
               for dust, scores in sorted(results.items())]
    for dust, mean, count in summary:
        print(f"dust={dust:<5} lesion-wise Dice={mean:.4f}  (n={count})")
    best = max(summary, key=lambda row: row[1])
    print(f"Best dust threshold: {best[0]} (Dice {best[1]:.4f})")
    json.dump({"best_dust": best[0], "best_dice": best[1],
               "all": [{"dust": d, "dice": m, "n": n} for d, m, n in summary]},
              open("lesion_metrics.json", "w"), indent=2)


def main():
    parser = argparse.ArgumentParser(description="Lesion-wise Dice with dust sweep.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--dust-thresholds", nargs="+", type=int,
                        default=[0, 10, 25, 50, 75, 100])
    parser.add_argument("--label", type=int, default=4)
    args = parser.parse_args()
    run(args.pred_dir, args.gt_dir, args.dust_thresholds, args.label)


if __name__ == "__main__":
    main()
