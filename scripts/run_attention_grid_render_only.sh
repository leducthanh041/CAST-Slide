#!/bin/bash
set -euo pipefail
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${ATTENTION_CONFIG:-configs/attention_maps/ood_fold4_naive_6task.yaml}"

echo "[INFO] mode=render_only input=existing_original_and_attention_jpg"
echo "[INFO] config=$CONFIG"
exec "$PYTHON_BIN" -u create_continual_attention_grid.py \
  --config "$CONFIG" \
  --render-only
