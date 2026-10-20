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
CESC_RANKS="${CESC_RANKS:-3,4,5,6,7,8}"

echo "[INFO] pipeline=CESC_unreviewed_Head4_from_pre_CESC_snapshots"
echo "[INFO] ranks=$CESC_RANKS config=$CONFIG"
exec "$PYTHON_BIN" -u visualize_cesc_head4_alternatives.py \
  --config "$CONFIG" --ranks "$CESC_RANKS"
