#!/bin/bash
set -e
PRED_DIR=$1
OUTPUT_DIR=$2
RC_MIN=${3:-50}
ET_MIN=${4:-15}
python -m src.postprocess.rc_postprocess \
    --pred-dir "${PRED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --rc-min-voxels "${RC_MIN}" \
    --et-min-voxels "${ET_MIN}"
