# Method

## Problem

The resection-cavity (RC) class is severely under-represented in post-treatment
brain-metastasis MRI, causing standard segmentation models to under-detect it.

## CarveMix-RC

### Carving

An RC-positive donor is selected, and its cavity mask is extracted. The
signed-distance transform of the mask is thresholded at a sampled parameter to
produce a carved region whose extent varies around the annotated cavity.

### Insertion

The carved region is enclosed in an axis-aligned bounding box and inserted at a
valid random location within an RC-negative target. Donor-target pairs whose
cavity does not fit inside the target are rejected rather than resized, so
intensities and discrete labels are transferred without interpolation.

### Harmonization

Donor intensities within the carved region are matched to the target using a
per-modality z-score transfer, reducing boundary discontinuities.

### Training

The augmented dataset trains an RC-weighted nnU-Net. The cross-entropy weight
for the RC class is raised to 3.0 and the foreground-oversampling probability to
0.66. Predictions from complementary architectures (standard and Residual
Encoder) are ensembled and refined with dust removal and hole-filling.

## Registering the custom trainer

nnU-Net discovers trainers by class name. Copy the trainer into the nnU-Net
trainer directory so it can be found:

```bash
TRAINER_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer'))")
cp src/training/nnUNetTrainer_RCWeightedOversample.py "${TRAINER_DIR}/"
```

## Labels

| Value | Region |
|-------|--------|
| 0 | Background |
| 1 | NETC |
| 2 | SNFH |
| 3 | ET |
| 4 | RC |

## FP-aware ensemble

For RC-focused inference, `src/postprocess/fp_aware_ensemble.py` provides an
orientation-safe, false-positive-aware alternative to the standard nnU-Net
ensemble. Tumor-class probabilities are averaged across two models. For RC,
`agreement` mode retains RC probability only where both models exceed the RC
confidence threshold. The fused probabilities are exported using nnU-Net's
native `export_prediction_from_logits`, which restores the case-specific crop,
transpose, geometry, and orientation recorded in the prediction properties.
The exported segmentation is then refined by RC connected-component removal,
RC hole filling, and ET connected-component removal.

Example:

```bash
python -m src.postprocess.fp_aware_ensemble \
  --npz-dir-a <PRED_STANDARD> \
  --npz-dir-b <PRED_RESENCL> \
  --output-dir <OUTPUT_DIR> \
  --dataset-json <DATASET_JSON> \
  --plans-json <PLANS_JSON> \
  --rc-mode agreement \
  --rc-thresh 0.5 \
  --rc-dust 50 \
  --et-dust 15
```
