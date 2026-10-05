#!/bin/bash
#
# CLASS-IL TTA evaluation runner.
# Keeps hot writes under /docker via repo-local logs/checkpoints symlinks and
# uses *_num_workers0 configs to avoid DataLoader multiprocessing.

#SBATCH --job-name=test_classIL_tta
#SBATCH --output=logs/test_classIL_tta_%j.out
#SBATCH --error=logs/test_classIL_tta_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
USER_NAME="${USER:-thanhld}"
PROJECT_NAME="$(basename "$PROJECT_ROOT")"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-/docker/data/$USER_NAME/$PROJECT_NAME}"
SETTING="${SETTING:-ind}"
ORDER="${ORDER:-forward}"
MODE="${MODE:-${TTA_MODE:-tcp}}"
TTA_PARAM_FILE="${TTA_PARAM_FILE:-}"
if [ -z "$TTA_PARAM_FILE" ]; then
    if [ "$SETTING" = "ood" ]; then
        TTA_PARAM_FILE="configs/ood/tta_ood.env"
    else
        TTA_PARAM_FILE="configs/ind/tta_ind.env"
    fi
fi
if [ -n "$TTA_PARAM_FILE" ]; then
    if [[ "$TTA_PARAM_FILE" != /* ]]; then
        TTA_PARAM_FILE="$PROJECT_ROOT/$TTA_PARAM_FILE"
    fi
    if [ ! -f "$TTA_PARAM_FILE" ]; then
        echo "[ERROR] TTA parameter file not found: $TTA_PARAM_FILE" >&2
        exit 1
    fi
    # Presets use default-only assignments, so explicit environment values win.
    # shellcheck source=/dev/null
    source "$TTA_PARAM_FILE"
fi
LOG_DIR="${LOG_DIR:-}"
if [ -n "$LOG_DIR" ] && [[ "$LOG_DIR" != /* && "$LOG_DIR" != logs && "$LOG_DIR" != logs/* ]]; then
    LOG_DIR="logs/$LOG_DIR"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs/classil_tta/${SETTING}_${ORDER}_${MODE}"
fi
case "$MODE" in
    tcp|naive|all) ;;
    *) echo "[ERROR] Unsupported MODE=$MODE (expected tcp|naive|all)" >&2; exit 1 ;;
esac
case "${SETTING}_${ORDER}" in
    ood_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_ood_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints_ood/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints_ood/merged}"
        ;;
    ind_forward)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged}"
        ;;
    ind_reverse)
        CONFIG_FORWARD="${CONFIG_FORWARD:-configs/default_reverse_eval_num_workers0.yaml}"
        SAVE_DIR_FORWARD="${SAVE_DIR_FORWARD:-./checkpoints/finetuned_reverse}"
        MERGE_MODEL_PATH_FORWARD="${MERGE_MODEL_PATH_FORWARD:-./checkpoints/merged_reverse}"
        ;;
    ood_reverse)
        echo "[ERROR] OOD reverse is not configured. Use SETTING=ind ORDER=reverse." >&2
        exit 1
        ;;
    *)
        echo "[ERROR] Unsupported SETTING/ORDER: SETTING=$SETTING ORDER=$ORDER (expected ind|ood with forward, or ind with reverse)" >&2
        exit 1
        ;;
esac

CLASSIL_ENTRYPOINT="${CLASSIL_ENTRYPOINT:-tools/run_classil_with_pt_features.py}"
TTA_ENTRYPOINT="${TTA_ENTRYPOINT:-test_classIL_tta.py}"
TTA_VARIANTS="${TTA_VARIANTS:-$MODE}"

TTA_M="${TTA_M:-8}"                         # sub-bags/slide
TTA_K_SUB="${TTA_K_SUB:-300}"               # patches/sub-bag
TTA_TOP_RATIO="${TTA_TOP_RATIO:-0.5}"       # confident sub-bag ratio
TTA_ALPHA="${TTA_ALPHA:-0.5}"               # task loss weight
TTA_L2_ANCHOR_BETA="${TTA_L2_ANCHOR_BETA:-1.0}"
TTA_LR="${TTA_LR:-1e-4}"                    # LN optimizer lr
TTA_N_STEPS="${TTA_N_STEPS:-}"
TTA_TCP_N_STEPS="${TTA_TCP_N_STEPS:-5}"
TTA_NAIVE_N_STEPS="${TTA_NAIVE_N_STEPS:-5}"
TTA_PARAM_SCOPE="${TTA_PARAM_SCOPE:-ln_only}"  # ln_only | full
TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.4}"  # WSI-level filter
TTA_GAMMA="${TTA_GAMMA:-0.5}"                  # JSD task-agreement weight
TTA_SELECT_MODE="${TTA_SELECT_MODE:-intersection}"  # union | intersection | class_only | task_only
TTA_USE_TASK_DIVERSITY="${TTA_USE_TASK_DIVERSITY:-0}"
TTA_NO_TASK_AGREEMENT="${TTA_NO_TASK_AGREEMENT:-0}"   # 1 disables JSD agreement

# Prompt embedding-space adaptation.
# TCP default follows the prompt-adapt baseline:
#   teacher EMA on, task-prompt EMA on, source-anchor enabled.
# Naive inference uses the student by default. DaPC may still
# instantiate a teacher internally to build detached reliability targets.
TTA_NO_TEACHER="${TTA_NO_TEACHER:-0}"
TTA_TCP_INFERENCE_MODEL="${TTA_TCP_INFERENCE_MODEL:-teacher}"
TTA_NAIVE_INFERENCE_MODEL="${TTA_NAIVE_INFERENCE_MODEL:-student}"
TTA_EMA_ALPHA="${TTA_EMA_ALPHA:-0.999}"
TTA_NO_ADAPT_PROMPTS="${TTA_NO_ADAPT_PROMPTS:-0}"
TTA_EMA_ALPHA_PROMPT="${TTA_EMA_ALPHA_PROMPT:-0.999}"
TTA_DELTA_MARGIN="${TTA_DELTA_MARGIN:-0.10}"
TTA_TP_ANCHOR_BETA="${TTA_TP_ANCHOR_BETA:-0.3}"
TTA_GAMMA_MARGIN="${TTA_GAMMA_MARGIN:-0.0}"
TTA_NO_RESET_PROMPT_PER_TASK="${TTA_NO_RESET_PROMPT_PER_TASK:-0}"
TTA_TAU_TASK="${TTA_TAU_TASK:-0.70}"        # TCP confidence fallback gate
TTA_NAIVE_USE_TASK_ENTROPY="${TTA_NAIVE_USE_TASK_ENTROPY:-1}"
TTA_USE_DAPC="${TTA_USE_DAPC:-1}"
TTA_DAPC_LOSS_WEIGHT="${TTA_DAPC_LOSS_WEIGHT:-1.0}"
TTA_CLASS_LOSS_WEIGHT="${TTA_CLASS_LOSS_WEIGHT:-1.0}"
TTA_ENTROPY_LOSS_WEIGHT="${TTA_ENTROPY_LOSS_WEIGHT:-1.0}"
if [ "$SETTING" = "ood" ]; then
    TTA_DAPC_TAU_ANCHOR="${TTA_DAPC_TAU_ANCHOR:-0.70}"
    TTA_DAPC_BETA="${TTA_DAPC_BETA:-1.5}"
else
    TTA_DAPC_TAU_ANCHOR="${TTA_DAPC_TAU_ANCHOR:-0.92}"
    TTA_DAPC_BETA="${TTA_DAPC_BETA:-1.2}"
fi
TTA_VERBOSE_LOSS="${TTA_VERBOSE_LOSS:-1}"
TTA_DIAG_DIR="${TTA_DIAG_DIR:-}"
TTA_RESULT_CSV="${TTA_RESULT_CSV:-}"
TTA_EFFICIENCY_JSON="${TTA_EFFICIENCY_JSON:-}"

# Python runtime
if [ -z "${PYTHON_BIN:-}" ]; then
    DEFAULT_PYTHON="/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/mergePre/bin/python3.10"
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

cd "$PROJECT_ROOT"

# Runtime directories
mkdir -p "$MERGESLIDE_LOCAL_ROOT/logs" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints" \
         "$MERGESLIDE_LOCAL_ROOT/checkpoints_ood" \
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

# Run summary
TTA_EPISODIC="${TTA_EPISODIC:-0}"
case "$TTA_EPISODIC" in
    0) RESET_LABEL="continual" ;;
    1) RESET_LABEL="episodic_per_slide" ;;
    *)
        echo "[ERROR] TTA_EPISODIC must be 0 or 1, got: $TTA_EPISODIC" >&2
        exit 1
        ;;
esac

echo "[INFO] start at $(date)"
echo "[INFO] project_root=$PROJECT_ROOT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] local_hot_root=$MERGESLIDE_LOCAL_ROOT"
echo "[INFO] tta_entrypoint=$TTA_ENTRYPOINT"
echo "[INFO] setting=$SETTING"
echo "[INFO] order=$ORDER"
echo "[INFO] mode=$MODE"
echo "[INFO] tta_param_file=${TTA_PARAM_FILE:-<defaults>}"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] config=$CONFIG_FORWARD"
echo "[INFO] save_dir=$SAVE_DIR_FORWARD"
echo "[INFO] merge_model_path=$MERGE_MODEL_PATH_FORWARD"
echo "[INFO] tta_result_csv=$TTA_RESULT_CSV"
echo "[INFO] tta_efficiency_json=$TTA_EFFICIENCY_JSON"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] tta_variants=$TTA_VARIANTS"
echo "[INFO] TTA M=$TTA_M | K_sub=$TTA_K_SUB | top_ratio=$TTA_TOP_RATIO | alpha=$TTA_ALPHA | regularizer=l2_anchor | l2_anchor_beta=$TTA_L2_ANCHOR_BETA | lr=$TTA_LR | n_steps_override=${TTA_N_STEPS:-<none>} | tcp_n_steps=$TTA_TCP_N_STEPS | naive_n_steps=$TTA_NAIVE_N_STEPS | param_scope=$TTA_PARAM_SCOPE | entropy_threshold=$TTA_ENTROPY_THRESHOLD | tau_task=$TTA_TAU_TASK | naive_use_task_entropy=$TTA_NAIVE_USE_TASK_ENTROPY | reset=$RESET_LABEL | verbose_loss=$TTA_VERBOSE_LOSS"
echo "[INFO] task_objective gamma=$TTA_GAMMA | select_mode=$TTA_SELECT_MODE | use_task_diversity=$TTA_USE_TASK_DIVERSITY | no_task_agreement=$TTA_NO_TASK_AGREEMENT"
echo "[INFO] prompt_adapt no_teacher=$TTA_NO_TEACHER | ema_alpha=$TTA_EMA_ALPHA | no_adapt_prompts=$TTA_NO_ADAPT_PROMPTS | ema_alpha_prompt=$TTA_EMA_ALPHA_PROMPT | delta_margin=$TTA_DELTA_MARGIN | tp_anchor_beta=$TTA_TP_ANCHOR_BETA | gamma_margin=$TTA_GAMMA_MARGIN | no_reset_prompt_per_task=$TTA_NO_RESET_PROMPT_PER_TASK"
echo "[INFO] dapc enabled=$TTA_USE_DAPC | weight=$TTA_DAPC_LOSS_WEIGHT | entropy_weight=$TTA_ENTROPY_LOSS_WEIGHT | tau_anchor=$TTA_DAPC_TAU_ANCHOR | beta=$TTA_DAPC_BETA"

entrypoint_path="$TTA_ENTRYPOINT"
if [[ "$entrypoint_path" != /* ]]; then
    entrypoint_path="$PROJECT_ROOT/$entrypoint_path"
fi

supports_arg() {
    local arg_name="$1"
    grep -q -- "$arg_name" "$entrypoint_path"
}

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
    "$@" \
        > >(tee -a "$result_log") \
        2> >(tee -a "$error_log" >&2)
}

variant_enabled() {
    local variant="$1"
    if [ "$TTA_VARIANTS" = "all" ]; then
        return 0
    fi
    case ",$TTA_VARIANTS," in
        *,"$variant",*) return 0 ;;
        *) return 1 ;;
    esac
}

build_variant_args() {
    local variant="$1"
    local n_steps
    local alpha="$TTA_ALPHA"
    local gamma="$TTA_GAMMA"
    local ema_alpha="$TTA_EMA_ALPHA"
    local ema_alpha_prompt="$TTA_EMA_ALPHA_PROMPT"
    local delta_margin="$TTA_DELTA_MARGIN"
    local tp_anchor_beta="$TTA_TP_ANCHOR_BETA"
    local gamma_margin="$TTA_GAMMA_MARGIN"
    local tau_task="$TTA_TAU_TASK"
    local use_dapc="$TTA_USE_DAPC"
    local no_teacher="$TTA_NO_TEACHER"

    if [ -n "$TTA_N_STEPS" ]; then
        n_steps="$TTA_N_STEPS"
    elif [ "$variant" = "tcp" ]; then
        n_steps="$TTA_TCP_N_STEPS"
    else
        n_steps="$TTA_NAIVE_N_STEPS"
    fi

    if [ "$variant" = "naive" ]; then
        alpha="${TTA_NAIVE_ALPHA:-$alpha}"
        gamma="${TTA_NAIVE_GAMMA:-$gamma}"
        ema_alpha="${TTA_NAIVE_EMA_ALPHA:-$ema_alpha}"
        ema_alpha_prompt="${TTA_NAIVE_EMA_ALPHA_PROMPT:-$ema_alpha_prompt}"
        delta_margin="${TTA_NAIVE_DELTA_MARGIN:-$delta_margin}"
        tp_anchor_beta="${TTA_NAIVE_TP_ANCHOR_BETA:-$tp_anchor_beta}"
        gamma_margin="${TTA_NAIVE_GAMMA_MARGIN:-$gamma_margin}"
        tau_task="${TTA_NAIVE_TAU_TASK:-$tau_task}"
        use_dapc="${TTA_NAIVE_USE_DAPC:-$use_dapc}"
        no_teacher="${TTA_NAIVE_NO_TEACHER:-$no_teacher}"
    fi

    VARIANT_ARGS=(
    --entrypoint        "$TTA_ENTRYPOINT"
    --M                 "$TTA_M"
    --K_sub             "$TTA_K_SUB"
    --top_ratio         "$TTA_TOP_RATIO"
    --alpha             "$alpha"
    --l2_anchor_beta    "$TTA_L2_ANCHOR_BETA"
    --lr                "$TTA_LR"
    --n_steps           "$n_steps"
    --tta_param_scope   "$TTA_PARAM_SCOPE"
    --entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --gamma             "$gamma"
    --select_mode       "$TTA_SELECT_MODE"
    --ema_alpha         "$ema_alpha"
    --tcp_inference_model "$TTA_TCP_INFERENCE_MODEL"
    --naive_inference_model "$TTA_NAIVE_INFERENCE_MODEL"
    --ema_alpha_prompt  "$ema_alpha_prompt"
    --delta_margin      "$delta_margin"
    --tp_anchor_beta    "$tp_anchor_beta"
    --gamma_margin      "$gamma_margin"
    --tau_task          "$tau_task"
    --dapc_loss_weight  "$TTA_DAPC_LOSS_WEIGHT"
    --class_loss_weight "$TTA_CLASS_LOSS_WEIGHT"
    --entropy_loss_weight "$TTA_ENTROPY_LOSS_WEIGHT"
    --dapc_tau_anchor   "$TTA_DAPC_TAU_ANCHOR"
    --dapc_beta         "$TTA_DAPC_BETA"
    )
    if [ "$use_dapc" = "1" ]; then
        VARIANT_ARGS+=(--use_dapc)
    fi
    if [ "$TTA_VERBOSE_LOSS" = "1" ]; then
        VARIANT_ARGS+=(--verbose_loss)
    fi
    if [ "$TTA_EPISODIC" = "1" ]; then
        VARIANT_ARGS+=(--episodic)
    fi
    if [ "$TTA_NAIVE_USE_TASK_ENTROPY" = "1" ]; then
        VARIANT_ARGS+=(--naive_use_task_entropy)
    else
        VARIANT_ARGS+=(--no_naive_task_entropy)
    fi
    if [ "$TTA_USE_TASK_DIVERSITY" = "1" ]; then
        VARIANT_ARGS+=(--use_task_diversity)
    fi
    if [ "$TTA_NO_TASK_AGREEMENT" = "1" ]; then
        VARIANT_ARGS+=(--no_task_agreement)
    fi
    if [ "$no_teacher" = "1" ]; then
        VARIANT_ARGS+=(--no_teacher)
    fi
    if [ "$TTA_NO_ADAPT_PROMPTS" = "1" ]; then
        VARIANT_ARGS+=(--no_adapt_prompts)
    fi
    if [ "$TTA_NO_RESET_PROMPT_PER_TASK" = "1" ]; then
        VARIANT_ARGS+=(--no_reset_prompt_per_task)
    fi
    if [ -n "$TTA_DIAG_DIR" ]; then
        if supports_arg "--diag_dir"; then
            VARIANT_ARGS+=(--diag_dir "$TTA_DIAG_DIR")
        else
            echo "[WARN] $TTA_ENTRYPOINT does not support --diag_dir; skipping TTA_DIAG_DIR=$TTA_DIAG_DIR" >&2
        fi
    fi
}
result_csv_for() {
    local variant="$1"
    if [ -n "$TTA_RESULT_CSV" ]; then
        if [ "$TTA_VARIANTS" = "all" ]; then
            local base="${TTA_RESULT_CSV%.csv}"
            echo "${base}_${variant}.csv"
        else
            echo "$TTA_RESULT_CSV"
        fi
    else
        echo "$LOG_DIR/tta_${variant}_routing_results.csv"
    fi
}

append_result_csv_arg() {
    local variant="$1"
    if supports_arg "--result_csv"; then
        echo "--result_csv"
        echo "$(result_csv_for "$variant")"
    fi
}

append_efficiency_json_arg() {
    if [ -n "$TTA_EFFICIENCY_JSON" ] && supports_arg "--efficiency_json"; then
        echo "--efficiency_json"
        echo "$TTA_EFFICIENCY_JSON"
    fi
}

# Run selected Class-IL TTA mode on selected protocol.
# Protocols:
#   SETTING=ind ORDER=forward  -> B->R->N->E->T->C
#   SETTING=ood ORDER=forward  -> cross-site/OOD forward
#   SETTING=ind ORDER=reverse  -> C->T->E->N->R->B
# Modes:
#   MODE=tcp | MODE=naive | MODE=all

if variant_enabled tcp; then
    build_variant_args tcp
    run_to_logs \
        "$LOG_DIR/result_tta_tcp.log" \
        "$LOG_DIR/error_tta_tcp.log" \
        "$PYTHON_BIN" -u "$CLASSIL_ENTRYPOINT" \
            --config           "$CONFIG_FORWARD" \
            --save_dir         "$SAVE_DIR_FORWARD" \
            --merge_model_path "$MERGE_MODEL_PATH_FORWARD" \
            --mode tcp \
            "${VARIANT_ARGS[@]}" \
            $(append_result_csv_arg tcp) \
            $(append_efficiency_json_arg)
fi

if variant_enabled naive; then
   build_variant_args naive
   run_to_logs \
       "$LOG_DIR/result_tta_naive.log" \
       "$LOG_DIR/error_tta_naive.log" \
       "$PYTHON_BIN" -u "$CLASSIL_ENTRYPOINT" \
           --config           "$CONFIG_FORWARD" \
           --save_dir         "$SAVE_DIR_FORWARD" \
           --merge_model_path "$MERGE_MODEL_PATH_FORWARD" \
           --mode naive \
           "${VARIANT_ARGS[@]}" \
           $(append_result_csv_arg naive) \
           $(append_efficiency_json_arg)
fi

echo "[INFO] finished at $(date)"
