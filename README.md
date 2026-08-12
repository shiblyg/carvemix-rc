# CarveMix-RC

Lesion-aware synthetic augmentation for resection-cavity (RC) segmentation in
post-treatment brain metastases, built on nnU-Net v2 for the BraTS-METS 2026
challenge.

Resection cavities appear in only a small fraction of post-treatment cases,
which causes standard models to under-segment them. CarveMix-RC transfers real
cavities from RC-positive donors into RC-negative targets, and combines this
with an RC-weighted trainer and lesion-aware post-processing.

## Repository layout

```
carvemix-rc/
├── configs/carvemix_rc.yaml          default parameters
├── docs/METHOD.md                    method description
├── docs/PUBLISHING.md                how to publish this repository
├── scripts/                          thin command-line wrappers
│   ├── generate.sh                   synthetic case generation
│   ├── train.sh                      training (nnU-Net)
│   ├── predict.sh                    inference
│   ├── ensemble.sh                   standard probability ensembling
│   ├── ensemble_fp_aware.sh          FP-aware RC agreement ensemble
│   └── postprocess.sh                dust removal and hole-filling
└── src/
    ├── carvemix_rc/carvemix_rc.py    signed-distance carving and generation
    ├── training/                     RC-weighted nnU-Net trainer
    ├── postprocess/                  FP-aware ensemble and lesion post-processing
    └── evaluation/                   lesion-wise metrics and bootstrap
```

## Installation

```bash
git clone https://github.com/<username>/carvemix-rc.git
cd carvemix-rc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.1+, and nnU-Net v2. Register the custom
trainer by placing `src/training/nnUNetTrainer_RCWeightedOversample.py` on the
nnU-Net trainer search path (see docs/METHOD.md).

## Workflow

### 1. Generate synthetic cases

```bash
python -m src.carvemix_rc.carvemix_rc \
    --images-dir  data/DatasetXXX/imagesTr \
    --labels-dir  data/DatasetXXX/labelsTr \
    --out-images  data/DatasetXXX/imagesTr \
    --out-labels  data/DatasetXXX/labelsTr \
    --n-generate  300
```

### 2. Train

```bash
bash scripts/train.sh <DATASET_ID> nnUNetPlans 0
bash scripts/train.sh <DATASET_ID> nnUNetResEncUNetLPlans 0
```

### 3. Predict

```bash
bash scripts/predict.sh <DATASET_ID> <INPUT_DIR> <OUTPUT_DIR> nnUNetPlans
```

### 4. Ensemble

Standard nnU-Net probability ensemble:

```bash
bash scripts/ensemble.sh <PRED_STANDARD> <PRED_RESENCL> <OUTPUT_DIR>
```

FP-aware RC ensemble (recommended when suppressing isolated RC false positives):

```bash
bash scripts/ensemble_fp_aware.sh \
    <PRED_STANDARD> <PRED_RESENCL> <OUTPUT_DIR> \
    <DATASET_JSON> <PLANS_JSON> \
    --rc-mode agreement --rc-thresh 0.5 --rc-dust 50 --et-dust 15
```

The FP-aware implementation averages tumor probabilities while requiring
agreement between both models for RC. It then exports through nnU-Net's native
export pipeline so the original crop, transpose, geometry, and orientation are
restored correctly before RC/ET connected-component post-processing.

### 5. Post-process

```bash
python -m src.postprocess.rc_postprocess \
    --pred-dir <ENSEMBLE_DIR> --output-dir <FINAL_DIR> \
    --rc-min-voxels 50 --et-min-voxels 15
```

Confidence-aware post-processing (keeps small, high-confidence lesions):

```bash
python -m src.postprocess.rc_postprocess_confidence \
    --pred-dir <PRED_WITH_NPZ> --output-dir <FINAL_DIR> \
    --rc-size 75 --rc-conf 0.5 --et-size 15 --et-conf 0.5
```

### 6. Evaluate

```bash
python -m src.evaluation.lesion_metrics --pred-dir <FINAL_DIR> --gt-dir <GT_DIR> --label 4
python -m src.evaluation.bootstrap --scores-a baseline.csv --scores-b carvemix.csv
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_range` | `[-3.0, 5.0]` | Signed-distance threshold range |
| `min_donor_voxels` | `50` | Minimum cavity size for a donor |
| `n_generate` | `300` | Number of synthetic cases |
| `rc_class_weight` | `3.0` | Cross-entropy weight for the RC class |
| `oversample_foreground` | `0.66` | Foreground oversampling probability |
| `rc_min_voxels` | `50` | RC component threshold (post-processing) |
| `et_min_voxels` | `15` | ET component threshold (post-processing) |

## Data

Data are provided by the BraTS 2026 Brain Metastases Challenge
(https://www.synapse.org/brats2026) and are not redistributed here.

Label convention: `0` background, `1` NETC, `2` SNFH, `3` ET, `4` RC.

## Citation

```bibtex
@inproceedings{carvemixrc2026,
  title     = {CarveMix-RC: Lesion-Aware Synthetic Augmentation for Resection-Cavity Segmentation in Brain Metastases},
  author    = {<Authors>},
  booktitle = {BraTS-METS Challenge, MICCAI},
  year      = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
