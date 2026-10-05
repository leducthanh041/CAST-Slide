#!/bin/bash
#SBATCH --job-name=attn_grid_render
#SBATCH --output=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_render/attn_render_%j.out
#SBATCH --error=/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA/logs/slurm_attention_render/attn_render_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
umask 000

METHOD_DIR="/datastore/uittogether3/LuuTru/Thanhld/WSI/MergeSlide_TTA"

module clear -f
module load slurm/slurm/24.11
source /datastore/uittogether3/tools/miniconda3/etc/profile.d/conda.sh
set +u
conda activate /datastore/uittogether3/tools/miniconda3/envs/mergePre
set -u

export PYTHON_BIN="$(command -v python)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/matplotlib-attention-render-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] start=$(date) host=$(hostname) job=${SLURM_JOB_ID}"
cd "$METHOD_DIR"
bash scripts/run_attention_grid_render_only.sh
echo "[INFO] finished=$(date)"
