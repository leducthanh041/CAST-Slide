#!/bin/bash
#SBATCH --job-name=cast_ood_classil
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm/cast_ood_classil_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm/cast_ood_classil_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=mps:l40:2
#SBATCH --time=72:00:00

set -euo pipefail

PROJECT_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA"

echo "[INFO] start at $(date)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"

module clear -f
module load slurm/slurm/24.11

source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/mergePre
set -u

source /datastore/uittogether3/LuuTru/Thanhld/WSI/models/hf_cache_env.sh

export PROJECT_ROOT
export SETTING="ood"
export ORDER="forward"
export MODE="tcp"
export LOG_DIR="$PROJECT_ROOT/logs/classil_tta/ood_tcp"
export MERGESLIDE_DATA_ROOT="/datastore/uittogether3/LuuTru/Thanhld/WSI/dataset"
export MERGESLIDE_CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints_ood"
export MERGESLIDE_LOCAL_ROOT="$PROJECT_ROOT"
export PYTHON_BIN="$(command -v python)"
export WSI_NUM_WORKERS="${WSI_NUM_WORKERS:-8}"
export WSI_PIN_MEMORY="${WSI_PIN_MEMORY:-1}"
export WSI_PERSISTENT_WORKERS="${WSI_PERSISTENT_WORKERS:-1}"
export WSI_PREFETCH_FACTOR="${WSI_PREFETCH_FACTOR:-2}"
export HDF5_USE_FILE_LOCKING="FALSE"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<assigned by Slurm>}"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] WSI_NUM_WORKERS=$WSI_NUM_WORKERS"

exec bash scripts/test_classIL_tta.sh
