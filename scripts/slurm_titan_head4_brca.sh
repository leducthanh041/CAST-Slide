#!/bin/bash
#SBATCH --job-name=titan_head4_brca
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_heads/head4_brca_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_heads/head4_brca_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=mps:a100:2
#SBATCH --time=24:00:00

set -euo pipefail
umask 000

METHOD_DIR="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA"
REQUIRED_VRAM="${REQUIRED_VRAM:-60000}"

cleanup() {
    local rc=$?
    echo "[INFO] cleanup rc=$rc at $(date)"
    if declare -F release_gpu_reservation >/dev/null 2>&1; then
        release_gpu_reservation
    fi
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]; then
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
export MPLCONFIGDIR="/tmp/matplotlib-titan-head4-${SLURM_JOB_ID}"
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-titan-head4-${SLURM_JOB_ID}"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-titan-head4-log-${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# MPS jobs can see multiple physical GPUs. Select and reserve one with enough
# currently free VRAM before starting the whole-slide TITAN forward.
unset CUDA_VISIBLE_DEVICES
source /datastore/uittogether3/LuuTru/Thanhld/WSI/models/slurm_gpu_reserve.sh
BEST_GPU="$(reserve_best_gpu "$REQUIRED_VRAM")"
export CUDA_VISIBLE_DEVICES="$BEST_GPU"

mkdir -p "$MPLCONFIGDIR" "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "[INFO] start=$(date) host=$(hostname) job=${SLURM_JOB_ID}"
echo "[INFO] REQUIRED_VRAM=${REQUIRED_VRAM}MiB BEST_GPU=$BEST_GPU"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable after GPU selection"; print(f"[INFO] torch CUDA ready device={torch.cuda.get_device_name(0)}", flush=True)'
cd "$METHOD_DIR"
bash scripts/run_titan_head4_brca.sh
echo "[INFO] finished=$(date)"
