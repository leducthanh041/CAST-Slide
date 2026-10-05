#!/bin/bash
set -euo pipefail
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export MERGESLIDE_DATA_ROOT="${MERGESLIDE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
export MERGESLIDE_RAW_ROOT_PRIMARY="${MERGESLIDE_RAW_ROOT_PRIMARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_BRCA_LUSC_RCC}"
export MERGESLIDE_RAW_ROOT_AUXILIARY="${MERGESLIDE_RAW_ROOT_AUXILIARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_CESC_ESCA_TGCT}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${ATTENTION_CONFIG:-configs/attention_maps/ood_fold4_naive_6task.yaml}"
OUTPUT_DIR="${ATTENTION_OUTPUT_DIR:-logs/attention_maps/fold4_train_dominant_specimen_jet}"
MAIN_FIGURE="$OUTPUT_DIR/region_level_attention_maps_6tasks_fold4_naive_restored.jpg"
EPISODIC_BACKUP="$OUTPUT_DIR/region_level_attention_maps_6tasks_fold4_naive_restored_episodic_backup.jpg"

if [ -f "$MAIN_FIGURE" ] && [ ! -f "$EPISODIC_BACKUP" ]; then
    cp "$MAIN_FIGURE" "$EPISODIC_BACKUP"
    echo "[INFO] preserved_previous_figure=$EPISODIC_BACKUP"
fi

echo "[INFO] pipeline=original_titan_pool_attention_jet"
echo "[INFO] protocol=OOD fold=4 online_causal_six_WSI mode=naive adapted_student"
echo "[INFO] config=$CONFIG"
echo "[INFO] output=logs/attention_maps/fold4_train_dominant_specimen_jet"
echo "[INFO] main_figure=region_level_attention_maps_6tasks_fold4_naive_restored.jpg"

exec "$PYTHON_BIN" -u create_continual_attention_grid.py --config "$CONFIG"
