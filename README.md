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
├── configs/carvemix\_rc.yaml          default parameters
├── docs/METHOD.md                    method description
├── docs/PUBLISHING.md                how to publish this repository
├── scripts/                          thin command-line wrappers
│   ├── generate.sh                   synthetic case generation
│   ├── train.sh                      training (nnU-Net)
│   ├── predict.sh                    inference
│   ├── ensemble.sh                   standard probability ensembling
│   ├── ensemble\_fp\_aware.sh          FP-aware RC agreement ensemble
│   └── postprocess.sh                dust removal and hole-filling
└── src/
    ├── carvemix\_rc/carvemix\_rc.py    signed-distance carving and generation
    ├── training/                     RC-weighted nnU-Net trainer
    ├── postprocess/                  FP-aware ensemble and lesion post-processing
    └── evaluation/                   lesion-wise metrics and bootstrap
```

## Installation

```bash
git clone https://github.com/<username>/carvemix-rc.git
cd carvemix-rc
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.1+, and nnU-Net v2. Register the custom
trainer by placing `src/training/nnUNetTrainer\_RCWeightedOversample.py` on the
nnU-Net trainer search path (see docs/METHOD.md).

## Workflow

### 1\. Generate synthetic cases

```bash
python -m src.carvemix\_rc.carvemix\_rc \\
    --images-dir  data/DatasetXXX/imagesTr \\
    --labels-dir  data/DatasetXXX/labelsTr \\
    --out-images  data/DatasetXXX/imagesTr \\
    --out-labels  data/DatasetXXX/labelsTr \\
    --n-generate  300
```

### 2\. Train

```bash
bash scripts/train.sh <DATASET\_ID> nnUNetPlans 0
bash scripts/train.sh <DATASET\_ID> nnUNetResEncUNetLPlans 0
```

### 3\. Predict

```bash
bash scripts/predict.sh <DATASET\_ID> <INPUT\_DIR> <OUTPUT\_DIR> nnUNetPlans
```

### 4\. Ensemble

Standard nnU-Net probability ensemble:

```bash
bash scripts/ensemble.sh <PRED\_STANDARD> <PRED\_RESENCL> <OUTPUT\_DIR>
```

FP-aware RC ensemble (recommended when suppressing isolated RC false positives):

```bash
bash scripts/ensemble\_fp\_aware.sh \\
    <PRED\_STANDARD> <PRED\_RESENCL> <OUTPUT\_DIR> \\
    <DATASET\_JSON> <PLANS\_JSON> \\
    --rc-mode agreement --rc-thresh 0.5 --rc-dust 50 --et-dust 15
```

The FP-aware implementation averages tumor probabilities while requiring
agreement between both models for RC. It then exports through nnU-Net's native
export pipeline so the original crop, transpose, geometry, and orientation are
restored correctly before RC/ET connected-component post-processing.

### 5\. Post-process

```bash
python -m src.postprocess.rc\_postprocess \\
    --pred-dir <ENSEMBLE\_DIR> --output-dir <FINAL\_DIR> \\
    --rc-min-voxels 50 --et-min-voxels 15
```

Confidence-aware post-processing (keeps small, high-confidence lesions):

```bash
python -m src.postprocess.rc\_postprocess\_confidence \\
    --pred-dir <PRED\_WITH\_NPZ> --output-dir <FINAL\_DIR> \\
    --rc-size 75 --rc-conf 0.5 --et-size 15 --et-conf 0.5
```

### 6\. Evaluate

```bash
python -m src.evaluation.lesion\_metrics --pred-dir <FINAL\_DIR> --gt-dir <GT\_DIR> --label 4
python -m src.evaluation.bootstrap --scores-a baseline.csv --scores-b carvemix.csv
```

## Configuration

|Parameter|Value|Description|
|-|-|-|
|`lambda\_range`|`\[-3.0, 5.0]`|Signed-distance threshold range|
|`min\_donor\_voxels`|`50`|Minimum cavity size for a donor|
|`n\_generate`|`300`|Number of synthetic cases|
|`rc\_class\_weight`|`3.0`|Cross-entropy weight for the RC class|
|`oversample\_foreground`|`0.66`|Foreground oversampling probability|
|`rc\_min\_voxels`|`50`|RC component threshold (post-processing)|
|`et\_min\_voxels`|`15`|ET component threshold (post-processing)|

## Data

Data are provided through the \[BraTS 2026 Challenge](https://www.synapse.org/Synapse:syn74274097/wiki/639579) and are not redistributed in this repository.



Label convention: `0` background, `1` NETC, `2` SNFH, `3` ET, `4` RC.



## Citation

If you use this repository, please cite the corresponding manuscript. Final citation metadata will be updated upon publication.

```bibtex
@inproceedings{carvemixrc2026,

&#x20; title     = {CarveMix-RC: Lesion-Aware Synthetic Augmentation for Resection-Cavity Segmentation in Brain Metastases},

&#x20; author    = {To be updated},

&#x20; booktitle = {To be updated},

&#x20; year      = {2026}

}```

## License

MIT License. See [LICENSE](LICENSE).

