#!/bin/bash
#
# Merge runner for merge.py (OOD config).
# Mirrors train.sh exactly -- PT-first wrapper + run_to_logs + docker checkpoints.

#SBATCH --job-name=merge_ood
#SBATCH --output=logs/merge_ood_%j.out
#SBATCH --error=logs/merge_ood_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime_paths.sh"
LOG_DIR="${LOG_DIR:-logs}"

# PT-first wrapper -- fixes NFS HDF5 hang
CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
MERGE_ENTRYPOINT="${MERGE_ENTRYPOINT:-merge.py}"

# Config
CONFIG="${CONFIG:-configs/default_ood_eval_num_workers0.yaml}"

# checkpoints_ood: same level as checkpoints, not nested
FINETUNED_DIR="${FINETUNED_DIR:-$MERGESLIDE_CHECKPOINT_ROOT/finetuned}"
MERGED_DIR="${MERGED_DIR:-$MERGESLIDE_CHECKPOINT_ROOT/merged}"

# ---------------------------------------------------------------------------
# Python binary
# ---------------------------------------------------------------------------
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Directory + symlink setup (same pattern as all other scripts)
# ---------------------------------------------------------------------------
mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints/finetuned" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints/merged" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood/finetuned" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood/merged" \
         "$MERGESLIDE_LOCAL_ROOT/sqlite" \
         "$MERGESLIDE_LOCAL_ROOT/tmp"

for name in logs checkpoints checkpoints_ood; do
    repo_path="$PROJECT_ROOT/$name"
    local_path="$MERGESLIDE_LOCAL_ROOT/$name"
    if [ -L "$repo_path" ]; then
        :
    elif [ -e "$repo_path" ]; then
        echo "[WARN] $repo_path is not a symlink; hot writes should use $local_path"
    else
        ln -s "$local_path" "$repo_path"
    fi
done

mkdir -p "$LOG_DIR"
export TMPDIR="${TMPDIR:-$MERGESLIDE_LOCAL_ROOT/tmp}"
export SQLITE_TMPDIR="${SQLITE_TMPDIR:-$MERGESLIDE_LOCAL_ROOT/sqlite}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export MERGESLIDE_DISABLE_CHECKPOINT_MIRROR="${MERGESLIDE_DISABLE_CHECKPOINT_MIRROR:-1}"

# ---------------------------------------------------------------------------
# Logging info
# ---------------------------------------------------------------------------
echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] config=$CONFIG"
echo "[INFO] finetuned_dir=$FINETUNED_DIR"
echo "[INFO] merged_dir=$MERGED_DIR  (merged checkpoints saved to docker)"
echo "[INFO] checkpoint_mirror_disabled=$MERGESLIDE_DISABLE_CHECKPOINT_MIRROR"
echo "[INFO] merge_entrypoint=$MERGE_ENTRYPOINT"

# ---------------------------------------------------------------------------
# check_log_not_held -- identical to all other scripts
# ---------------------------------------------------------------------------
check_log_not_held() {
    local log_path="$1"
    local resolved_log
    resolved_log="$(readlink -f "$log_path" 2>/dev/null || true)"
    if [ -z "$resolved_log" ]; then return 0; fi
    local fd target pid state cmdline
    for fd in /proc/[0-9]*/fd/1 /proc/[0-9]*/fd/2; do
        [ -e "$fd" ] || continue
        target="$(readlink -f "$fd" 2>/dev/null || true)"
        [ "$target" = "$resolved_log" ] || continue
        pid="${fd#/proc/}"; pid="${pid%%/*}"
        [ "$pid" = "$$" ] && continue
        state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
        case "$cmdline" in torch_shm_manager*) continue ;; esac
        echo "[ERROR] $log_path is already held by PID $pid state=$state cmd=$cmdline" >&2
        echo "[ERROR] Refusing to reuse this log." >&2
        return 1
    done
}

run_to_logs() {
    local result_log="$1"
    local error_log="$2"
    shift 2
    echo "[INFO] running: $*"
    echo "[INFO] result_log=$result_log"
    echo "[INFO] error_log=$error_log"
    check_log_not_held "$result_log"
    check_log_not_held "$error_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$result_log"
    { echo "[INFO] start at $(date)"; echo "[INFO] command=$*"; } > "$error_log"
    "$@" >> "$result_log" 2>> "$error_log"
}

export CUDA_VISIBLE_DEVICES=4

# ---------------------------------------------------------------------------
# Run merge via PT-first wrapper
# Merged checkpoints go to $MERGED_DIR (docker), not NFS root
# ---------------------------------------------------------------------------
run_to_logs \
    "$LOG_DIR/result_merge_ood.log" \
    "$LOG_DIR/error_merge_ood.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint                    "$MERGE_ENTRYPOINT" \
        --config                        "$CONFIG" \
        --finetuned_checkpoints         "$FINETUNED_DIR" \
        --merged_checkpoints            "$MERGED_DIR"

echo "[INFO] Merge finished at $(date)"
echo "[INFO] Merged checkpoints saved to: $MERGED_DIR"
