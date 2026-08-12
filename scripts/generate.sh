#!/bin/bash
set -e
IMAGES_DIR=$1
LABELS_DIR=$2
OUT_IMAGES=$3
OUT_LABELS=$4
N_GENERATE=${5:-300}
python -m src.carvemix_rc.carvemix_rc \
    --images-dir "${IMAGES_DIR}" \
    --labels-dir "${LABELS_DIR}" \
    --out-images "${OUT_IMAGES}" \
    --out-labels "${OUT_LABELS}" \
    --n-generate "${N_GENERATE}"
