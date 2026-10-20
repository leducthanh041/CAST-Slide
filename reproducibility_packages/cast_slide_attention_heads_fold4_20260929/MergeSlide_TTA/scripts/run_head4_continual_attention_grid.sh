#!/bin/bash
set -euo pipefail
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export MERGESLIDE_DATA_ROOT="${MERGESLIDE_DATA_ROOT:-$PROJECT_ROOT/../dataset}"
export MERGESLIDE_RAW_ROOT_PRIMARY="${MERGESLIDE_RAW_ROOT_PRIMARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_BRCA_LUSC_RCC}"
export MERGESLIDE_RAW_ROOT_AUXILIARY="${MERGESLIDE_RAW_ROOT_AUXILIARY:-/datastore/uittogether/LuuTru/Thuchd/Research/data_raw_CESC_ESCA_TGCT}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${ATTENTION_CONFIG:-configs/attention_maps/ood_fold4_naive_6task_head4.yaml}"

echo "[INFO] pipeline=final_layer_tensor_head_index_4_online_causal"
echo "[INFO] config=$CONFIG"
exec "$PYTHON_BIN" -u create_continual_attention_grid.py --config "$CONFIG" "$@"
