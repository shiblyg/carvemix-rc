#!/bin/bash
set -e
DATASET_ID=$1
PLANS=${2:-nnUNetPlans}
FOLD=${3:-all}
nnUNetv2_train "${DATASET_ID}" 3d_fullres "${FOLD}" \
    -p "${PLANS}" \
    -tr nnUNetTrainer_RCWeightedOversample \
    --npz
