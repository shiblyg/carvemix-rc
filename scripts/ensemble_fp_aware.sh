#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <PRED_A> <PRED_B> <OUTPUT_DIR> <DATASET_JSON> <PLANS_JSON> [extra options]"
  exit 1
fi

PRED_A=$1
PRED_B=$2
OUTPUT_DIR=$3
DATASET_JSON=$4
PLANS_JSON=$5
shift 5

python -m src.postprocess.fp_aware_ensemble \
  --npz-dir-a "${PRED_A}" \
  --npz-dir-b "${PRED_B}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-json "${DATASET_JSON}" \
  --plans-json "${PLANS_JSON}" \
  "$@"
