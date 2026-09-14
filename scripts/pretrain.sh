#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TRAIN_MANIFEST VALIDATION_MANIFEST OUTPUT_DIR" >&2
  exit 2
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m src.training.pretrain \
  --config configs/pretrain.yaml \
  --train-manifest "$1" \
  --validation-manifest "$2" \
  --output-dir "$3"

