#!/bin/bash
set -e
DATASET_ID=$1
INPUT_DIR=$2
OUTPUT_DIR=$3
PLANS=${4:-nnUNetPlans}
nnUNetv2_predict \
    -d "${DATASET_ID}" \
    -i "${INPUT_DIR}" \
    -o "${OUTPUT_DIR}" \
    -c 3d_fullres \
    -p "${PLANS}" \
    -tr nnUNetTrainer_RCWeightedOversample \
    -f all \
    --save_probabilities
