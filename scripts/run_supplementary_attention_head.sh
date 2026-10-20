#!/bin/bash
set -euo pipefail
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HEAD_INDEX="${HEAD_INDEX:?Set HEAD_INDEX=10 or HEAD_INDEX=11}"
if [[ "$HEAD_INDEX" != "10" && "$HEAD_INDEX" != "11" ]]; then
    echo "[ERROR] supplementary HEAD_INDEX must be 10 or 11" >&2
    exit 2
fi

export MERGESLIDE_DATA_ROOT="${MERGESLIDE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
export MERGESLIDE_RAW_ROOT_PRIMARY="${MERGESLIDE_RAW_ROOT_PRIMARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_BRCA_LUSC_RCC}"
export MERGESLIDE_RAW_ROOT_AUXILIARY="${MERGESLIDE_RAW_ROOT_AUXILIARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_CESC_ESCA_TGCT}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${ATTENTION_CONFIG:-configs/attention_maps/ood_fold4_naive_6task_head4.yaml}"
OUTPUT_DIR="logs/attention_maps/fold4_head${HEAD_INDEX}_last_layer_supplementary"
FIGURE="region_level_attention_maps_6tasks_fold4_naive_head${HEAD_INDEX}_last_layer.jpg"
CESC_SLIDE="TCGA-C5-A1BJ-01A-01-TSA.f24c19a0-b8af-44e5-a88f-a5b8e55aa55b"

echo "[INFO] supplementary final-layer attention head_index=$HEAD_INDEX"
echo "[INFO] output_dir=$OUTPUT_DIR"
exec "$PYTHON_BIN" -u create_continual_attention_grid.py \
    --config "$CONFIG" \
    --head-index "$HEAD_INDEX" \
    --output-dir "$OUTPUT_DIR" \
    --figure-filename "$FIGURE" \
    --run-scope full \
    --cesc-slide-id "$CESC_SLIDE" \
    --cesc-left-specimen \
    --no-pre-wsi-attention \
    "$@"
