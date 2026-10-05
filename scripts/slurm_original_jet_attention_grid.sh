#!/bin/bash
#SBATCH --job-name=attn_online_jet
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_online/attn_online_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_online/attn_online_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=mps:l40:2
#SBATCH --time=72:00:00

set -euo pipefail
umask 000

METHOD_DIR="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    fi
}
trap cleanup EXIT

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/mergePre
set -u

source /datastore/uittogether3/LuuTru/Thanhld/WSI/models/hf_cache_env.sh
export PYTHON_BIN="$(command -v python)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENCV_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="/tmp/matplotlib-original-attention-${SLURM_JOB_ID}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-original-attention-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-original-attention-log-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

echo "[INFO] start=$(date) host=$(hostname) job=${SLURM_JOB_ID}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
cd "$METHOD_DIR"
bash scripts/run_original_jet_attention_grid.sh
echo "[INFO] finished=$(date)"
