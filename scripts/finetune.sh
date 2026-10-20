#!/bin/bash
#
# Per-task finetuning runner for train.py.
# Fixes NFS HDF5 hang via PT-first wrapper.
# Checkpoints saved to local /docker (hot storage), not NFS root.

#SBATCH --job-name=train_mergeslide
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/runtime_paths.sh"
LOG_DIR="${LOG_DIR:-logs}"

# PT-first wrapper -- fixes NFS HDF5 hang (same fix as eval scripts)
CLASSIL_WRAPPER="${CLASSIL_WRAPPER:-tools/run_classil_with_pt_features.py}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train.py}"

# Config: use OOD config as requested
CONFIG="${CONFIG:-configs/default_ood_eval_num_workers0.yaml}"

# Fold range -- change to run specific folds
FOLD_START="${FOLD_START:-0}"
FOLD_END="${FOLD_END:-10}"

# Checkpoint save dir: docker hot storage (not NFS root)
# checkpoints/    -- regular training
# checkpoints_ood/ -- OOD training  (same level, not nested)
SAVE_DIR="${SAVE_DIR:-$MERGESLIDE_CHECKPOINT_ROOT/finetuned}"

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

# ---------------------------------------------------------------------------
# Logging info
# ---------------------------------------------------------------------------
echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] config=$CONFIG"
echo "[INFO] fold_start=$FOLD_START | fold_end=$FOLD_END"
echo "[INFO] save_dir=$SAVE_DIR  (checkpoints saved to docker)"
echo "[INFO] train_entrypoint=$TRAIN_ENTRYPOINT"

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
# Run training via PT-first wrapper
# Checkpoints go to $SAVE_DIR (docker), not NFS root
# ---------------------------------------------------------------------------
run_to_logs \
    "$LOG_DIR/result_train.log" \
    "$LOG_DIR/error_train.log" \
    "$PYTHON_BIN" -u "$CLASSIL_WRAPPER" \
        --entrypoint  "$TRAIN_ENTRYPOINT" \
        --config      "$CONFIG" \
        --save_dir    "$SAVE_DIR" \
        --fold_start  "$FOLD_START" \
        --fold_end    "$FOLD_END"

echo "[INFO] Training finished at $(date)"
echo "[INFO] Checkpoints saved to: $SAVE_DIR"
