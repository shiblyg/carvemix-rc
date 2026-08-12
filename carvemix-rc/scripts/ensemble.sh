#!/bin/bash
set -e
PRED_A=$1
PRED_B=$2
OUTPUT_DIR=$3
nnUNetv2_ensemble -i "${PRED_A}" "${PRED_B}" -o "${OUTPUT_DIR}" -np 8
