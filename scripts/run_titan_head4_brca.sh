#!/bin/bash
set -euo pipefail
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export MERGESLIDE_DATA_ROOT="${MERGESLIDE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
export MERGESLIDE_RAW_ROOT_PRIMARY="${MERGESLIDE_RAW_ROOT_PRIMARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_BRCA_LUSC_RCC}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="${HEAD4_CONFIG:-configs/attention_heads/ood_fold5_brca_head4.yaml}"
PHASE="${HEAD4_PHASE:-all}"
LAYER_INDEX="${TITAN_LAYER_INDEX:-5}"
HEAD_INDEX="${TITAN_HEAD_INDEX:-4}"

echo "[INFO] attention=Transformer_CLS_to_patch_selected_head"
echo "[INFO] phase=$PHASE config=$CONFIG"
echo "[INFO] selected_paper_head4 layer_index=$LAYER_INDEX head_index=$HEAD_INDEX"

exec "$PYTHON_BIN" -u create_titan_head4_heatmaps.py \
  --config "$CONFIG" \
  --phase "$PHASE" \
  --layer-index "$LAYER_INDEX" \
  --head-index "$HEAD_INDEX"
