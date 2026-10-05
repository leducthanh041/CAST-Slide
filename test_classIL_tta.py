"""CLASS-IL evaluation for CAST-Slide.

Modes:
  tcp   (default): Task-to-Class Prompt-Aligned inference + TTA
  naive          : Direct class inference + TTA

TTA hyperparams:
  --M                  : sub-bags per slide (TTA batch size), default=8
  --K_sub              : patches per sub-bag, default=300
  --top_ratio          : confident sub-bag keep ratio, default=0.5
  --alpha              : task loss weight, default=0.5
  --l2_anchor_beta     : L2 regularizer weight toward merged source, default=1.0
  --tau_task           : TCP confidence threshold, default=0.70
  --lr                 : LN optimizer learning rate, default=1e-4
  --n_steps            : adapt steps per slide, default=1
  --episodic           : reset adaptation state before every slide
  --entropy_threshold  : only TTA when entropy >= threshold, default=0.4
                         Set 0.0 to TTA all slides.

Usage:
    python test_classIL_tta.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged \\
        --mode tcp
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from cast_slide.constants import K_PATCHES, NUM_TASKS
from cast_slide.datasets import Sequential_Generic_MIL_Dataset
from cast_slide.metrics import pad_numpy_arrays
from cast_slide.task_prompt_io import load_task_prompts_for_tasks
from cast_slide.prompts_zeroshot import (
    brca_prompts, rcc_prompts, nsclc_prompts,
    esca_prompts, tgct_prompts, cesc_prompts,
)
from cast_slide.utils import get_eval_metrics, seed_torch
from cast_slide.tta_adapter import CASTSlide, load_task_weights

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}

_PROMPT_FN_MAP = {
    "BRCA":  brca_prompts,
    "RCC":   rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA":  esca_prompts,
    "TGCT":  tgct_prompts,
    "CESC":  cesc_prompts,
}


def class_metric_labels(task_names, num_classes):
    return [
        f"{task_name}/class_{class_id}"
        for task_name, count in zip(task_names, num_classes)
        for class_id in range(count)
    ]


def safe_ovr_auc(targets, probabilities, class_id):
    binary_targets = (targets == class_id).astype(int)
    if np.unique(binary_targets).size < 2:
        return float("nan")
    return roc_auc_score(binary_targets, probabilities[:, class_id])

# Path handling

def get_local_hot_root() -> Path:
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", PROJECT_ROOT)).expanduser()


def ensure_local_hot_storage() -> Path:
    local_root = get_local_hot_root()
    local_root.mkdir(parents=True, exist_ok=True)
    for name in HOT_DIR_NAMES:
        (local_root / name).mkdir(parents=True, exist_ok=True)
    (local_root / "tmp").mkdir(parents=True, exist_ok=True)
    for name in ("logs", "checkpoints"):
        repo_path  = PROJECT_ROOT / name
        local_path = local_root / name
        if not repo_path.exists() and not repo_path.is_symlink():
            repo_path.symlink_to(local_path, target_is_directory=True)
    os.environ.setdefault("TMPDIR",                str(local_root / "tmp"))
    os.environ.setdefault("SQLITE_TMPDIR",         str(local_root / "sqlite"))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    return local_root


def resolve_hot_path(path: str, local_root: Path) -> Path:
    def resolve_hot_parts(parts: tuple[str, ...]) -> Path:
        repo_hot_root = PROJECT_ROOT / parts[0]
        if repo_hot_root.is_symlink():
            return repo_hot_root.resolve().joinpath(*parts[1:])
        return local_root.joinpath(*parts)

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        parts = raw.parts
        if parts and parts[0] in HOT_DIR_NAMES:
            return resolve_hot_parts(parts)
        return raw
    try:
        relative = raw.relative_to(PROJECT_ROOT)
    except ValueError:
        return raw
    parts = relative.parts
    if parts and parts[0] in HOT_DIR_NAMES:
        return resolve_hot_parts(parts)
    return raw


# Global classifier construction

def build_class_embeddings(device, task_names: list) -> torch.Tensor:
    """[768, C_total] -- identical logic to test_classIL_task_prompt.py."""
    unknown_tasks = [name for name in task_names if name not in _PROMPT_FN_MAP]
    if unknown_tasks:
        raise ValueError(
            f"Missing zero-shot prompt functions for tasks: {unknown_tasks}"
        )

    print("Building global class embeddings for naive/TCP fallback ...")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)
    _, templates = brca_prompts()
    all_prompts = []
    for name in task_names:
        class_prompts, _ = _PROMPT_FN_MAP[name]()
        all_prompts.extend(class_prompts)
    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = titan.zero_shot_classifier(
            all_prompts, templates, device=str(device)
        )
    del titan
    torch.cuda.empty_cache()
    return classifier.to(device)


# Per-task inference

def eval_task_tta(
    test_loader,
    task_id:              int,
    tta_model:            CASTSlide,
    task_to_global_class: dict,
    device,
    mode:                 str  = "tcp",
    num_classes_per_task: list = None,
    verbose_loss:         bool = False,
) -> tuple:
    """
    TTA inference for 1 task. Each slide: adapt_and_predict().

    TCP mode: pred_class is a task-local index.
    Naive mode: pred_class is a global classifier column index.
    """
    preds_all           = []
    probs_all           = []
    targets_all         = []
    convert_preds_all   = []
    convert_targets_all = []
    times               = []
    loss_logs           = []
    routing_correct     = 0

    # Map naive classifier columns to global class identifiers.
    if mode == "naive":
        column_to_global = np.array([
            task_to_global_class[t][local]
            for t in range(len(task_to_global_class))
            for local in sorted(task_to_global_class[t].keys())
        ])
        total_classes = len(column_to_global)
    else:
        column_to_global = None
        total_classes    = None

    for features, coords, label in tqdm(test_loader, leave=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        features = features.to(device)
        coords   = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        pred_class, probs, pred_task, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

        if verbose_loss:
            loss_logs.append(adapt_log)

        probs_np = probs.numpy()  # [1, C_task] for tcp | [1, C_total] for naive

        if mode == "tcp":
            # TCP predictions and labels are task-local.
            routing_correct += int(pred_task == task_id)
            preds_all.append(np.array([pred_class]))
            probs_all.append(probs_np)
            targets_all.append(label.numpy())

            g_label = task_to_global_class[task_id].get(int(label), -1)
            g_pred  = task_to_global_class[task_id].get(pred_class, -1)
            convert_targets_all.append(np.array([g_label]))
            convert_preds_all.append(np.array([g_pred]))

        else:
            # Naive predictions use global classifier columns.
            pred_global = int(column_to_global[pred_class])
            true_global = task_to_global_class[task_id][int(label)]

            # Remap probs from column order to global class order
            probs_out = np.zeros((1, total_classes), dtype=np.float32)
            for col_idx, g_idx in enumerate(column_to_global):
                probs_out[0, g_idx] = probs_np[0, col_idx]

            preds_all.append(np.array([pred_global]))
            probs_all.append(probs_out)
            targets_all.append(np.array([true_global]))

            convert_preds_all.append(np.array([pred_global]))
            convert_targets_all.append(np.array([true_global]))

    preds_arr   = np.concatenate(preds_all)
    targets_arr = np.concatenate(targets_all)
    try:
        probs_arr = np.concatenate(probs_all)
    except ValueError:
        probs_arr = pad_numpy_arrays(probs_all)

    metrics = get_eval_metrics(
        targets_arr, preds_arr, probs_arr,
        roc_kwargs={"multi_class": "ovo", "average": "macro"},
        prefix="",
    )

    loss_summary = {}
    if verbose_loss and loss_logs:
        adapted = [d for d in loss_logs if d.get("slide/adapted")]
        loss_keys = sorted({
            key
            for entry in adapted
            for key in entry
            if key.startswith(("loss/", "reliability/"))
        })
        loss_summary = {
            key: float(np.mean([entry[key] for entry in adapted if key in entry]))
            for key in loss_keys
        }
        loss_summary["adapted_count"] = len(adapted)
        loss_summary["total_count"] = len(loss_logs)
        tcp_confidences = [
            entry["slide/tcp_conf"]
            for entry in loss_logs
            if not np.isnan(entry.get("slide/tcp_conf", float("nan")))
        ]
        if tcp_confidences:
            loss_summary["tcp_conf_mean"] = float(np.mean(tcp_confidences))
            loss_summary["tcp_fallback_rate"] = float(np.mean([
                entry.get("slide/tcp_fallback", False) for entry in loss_logs
            ]))

    return (
        metrics, preds_arr, targets_arr, probs_arr,
        np.concatenate(convert_preds_all),
        np.concatenate(convert_targets_all),
        sum(times),
        routing_correct / max(1, len(test_loader)) if mode == "tcp" else float("nan"),
        loss_summary,
    )


def adapt_task_for_tcp_routing_tune(
    test_loader,
    task_id: int,
    tta_model: CASTSlide,
    device,
    measure_routing: bool,
) -> tuple[float, int, float]:
    """Run continual TTA while avoiding all class-level metric computation."""
    routing_correct = 0
    num_slides = 0
    elapsed_s = 0.0

    for features, coords, _label in tqdm(test_loader, leave=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        features = features.to(device)
        coords = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]
        _pred_class, _probs, pred_task, _adapt_log = (
            tta_model.adapt_and_predict(features, coords)
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_s += time.perf_counter() - started

        num_slides += 1
        if measure_routing:
            routing_correct += int(pred_task == task_id)

    routing_acc = (
        routing_correct / num_slides
        if measure_routing and num_slides > 0
        else float("nan")
    )
    return routing_acc, num_slides, elapsed_s


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Class-IL TTA evaluation")

    # Paths
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)

    # Inference mode
    parser.add_argument("--mode", type=str, default="tcp",
                        choices=["tcp", "naive"],
                        help="tcp (default): TCP inference | naive: direct class inference")

    # Adaptation parameters
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--alpha",             type=float, default=0.5)
    parser.add_argument(
        "--l2_anchor_beta",
        type=float,
        default=1.0,
                        help="L2 source-anchor regularization weight.",
    )
    parser.add_argument("--tau_task",           type=float, default=0.70,
                        help="TCP confidence threshold; fallback to global head below it.")
    parser.add_argument(
        "--naive_use_task_entropy",
        dest="naive_use_task_entropy",
        action="store_true",
        help="Use task-guided view selection and task-entropy diagnostics in naive mode (default).",
    )
    parser.add_argument(
        "--no_naive_task_entropy",
        dest="naive_use_task_entropy",
        action="store_false",
        help="Use class-only confidence selection in naive mode.",
    )
    parser.set_defaults(naive_use_task_entropy=True)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--tta_param_scope",    type=str,   default="ln_only",
                        choices=["ln_only", "full"],
                        help="Backbone parameter scope for TTA.")
    parser.add_argument("--entropy_threshold", type=float, default=0.4,
                        help="Only TTA when slide entropy >= threshold. "
                             "Set 0.0 to TTA all slides.")
    parser.add_argument("--episodic",          action="store_true",
                        help="Reset backbone, optimizer, teacher, and working "
                             "task prompts before every slide.")
    parser.add_argument("--use_task_diversity", action="store_true",
                        help="Enable task-distribution diversity.")
    parser.add_argument("--no_task_agreement",  action="store_true",
                        help="Disable JSD task agreement "
                             "term. Default: agreement term is ON.")
    parser.add_argument("--gamma",             type=float, default=0.5,
                        help="Weight of the JSD task-agreement term.")
    parser.add_argument("--select_mode",       type=str,   default="intersection",
                        choices=["union", "intersection", "class_only", "task_only"],
                        help="Confidence rule used to select sub-bags.")
    parser.add_argument("--no_teacher",        action="store_true",
                        help="Disable mean-teacher; route/infer with the "
                             "backbone being adapted directly.")
    parser.add_argument(
        "--tcp_inference_model",
        choices=["teacher", "student"],
        default="teacher",
        help="Backbone used by final TCP routing and class inference.",
    )
    parser.add_argument(
        "--naive_inference_model",
        choices=["student", "teacher"],
        default="student",
        help="Backbone used by final naive class inference.",
    )
    parser.add_argument("--ema_alpha",         type=float, default=0.999,
                        help="Teacher EMA momentum.")
    parser.add_argument("--no_adapt_prompts",  action="store_true",
                        help="Disable task-prompt EMA update; task_prompts "
                             "stay frozen.")
    parser.add_argument("--ema_alpha_prompt",  type=float, default=0.999,
                        help="Task-prompt EMA momentum.")
    parser.add_argument("--delta_margin",      type=float, default=0.10,
                        help="Confidence-gap gate for task-prompt update "
                             "(top1-top2 softmax score over task_prompts).")
    parser.add_argument("--tp_anchor_beta",    type=float, default=0.3,
                        help="Prompt pull-back toward the frozen source in [0,1].")
    parser.add_argument("--gamma_margin",      type=float, default=0.0,
                        help="Weight of task_margin_loss (0.0 = off).")
    parser.add_argument("--use_dapc", action="store_true",
                        help="Enable detached DaPC pseudo-label correction.")
    parser.add_argument("--dapc_loss_weight", type=float, default=1.0)
    parser.add_argument("--class_loss_weight", type=float, default=1.0)
    parser.add_argument("--entropy_loss_weight", type=float, default=1.0,
                        help="Weight of the entropy objective.")
    parser.add_argument("--dapc_tau_anchor", type=float, default=0.92)
    parser.add_argument("--dapc_beta", type=float, default=1.2)
    parser.add_argument("--no_reset_prompt_per_task", action="store_true",
                        help="Keep adapted task prompts across task boundaries.")
    parser.add_argument("--verbose_loss",      action="store_true")
    parser.add_argument(
        "--result_csv",
        type=str,
        default="",
        help="Optional CSV path to save per-fold/per-task TTA metrics, including TCP routing_acc.",
    )
    parser.add_argument(
        "--efficiency_json",
        type=str,
        default="",
        help="Optional JSON path to save updated params, TTA steps, throughput, and peak VRAM.",
    )
    parser.add_argument(
        "--tcp_tune_cesc_fold0",
        action="store_true",
        help="TCP tuning protocol: run fold 0 through tasks 0-5 continually, "
             "skip class metrics, and report only Task-5 CESC routing accuracy.",
    )
    parser.add_argument(
        "--tcp_tune_routing_all_folds",
        action="store_true",
        help="TCP tuning protocol: run all folds/tasks continually, skip "
             "class metrics, and report routing accuracy per fold/task.",
    )

    args = parser.parse_args()
    if (
        args.tcp_tune_cesc_fold0
        and args.tcp_tune_routing_all_folds
    ):
        parser.error(
            "--tcp_tune_cesc_fold0 and --tcp_tune_routing_all_folds "
            "are mutually exclusive"
        )
    routing_tune_only = (
        args.tcp_tune_cesc_fold0
        or args.tcp_tune_routing_all_folds
    )
    if routing_tune_only and args.mode != "tcp":
        parser.error("TCP routing-only tuning requires --mode tcp")
    if (
        args.mode == "naive"
        and args.naive_inference_model == "student"
        and not args.use_dapc
    ):
        # Naive Class-IL does not use TCP/task prompt routing. Keep the
        # teacher branch disabled by default even when users call this
        # entrypoint directly without the bash runner.
        args.no_teacher = True

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    if args.result_csv:
        args.result_csv = str(resolve_hot_path(args.result_csv, local_hot_root))
    if args.efficiency_json:
        args.efficiency_json = str(resolve_hot_path(args.efficiency_json, local_hot_root))
    else:
        args.efficiency_json = str(
            Path(args.merge_model_path) / f"efficiency_classil_tta_{args.mode}.json"
        )

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)
    if num_tasks != NUM_TASKS or len(seq_dataset.task_names) != NUM_TASKS:
        raise ValueError(
            "This TTA entrypoint requires the active six-task sequence: "
            f"config={num_tasks}, dataset={len(seq_dataset.task_names)}, "
            f"expected={NUM_TASKS}."
        )

    # Class embeddings
    task_prompts = load_task_prompts_for_tasks(
        PROJECT_ROOT / "task_prompts.pt",
        seq_dataset.task_names,
        device,
    )

    # Naive predicts globally; TCP also needs this classifier for its
    # low-confidence fallback.
    all_class_embeddings = build_class_embeddings(device, seq_dataset.task_names)
    expected_classes = sum(seq_dataset.num_classes)
    if all_class_embeddings.ndim != 2 or all_class_embeddings.shape[1] != expected_classes:
        raise ValueError(
            "Global class embedding shape does not match the configured task "
            f"sequence: expected [D, {expected_classes}], got "
            f"{tuple(all_class_embeddings.shape)}"
        )

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_accs         = []
    overall_baccs        = []
    overall_macro_f1s    = []
    overall_weighted_f1s = []
    overall_recalls      = []
    overall_precisions   = []
    overall_aucs         = []
    overall_times        = []
    all_acc_per_task     = []
    overall_routing_acc  = []
    overall_routing_acc_per_task = []
    all_results          = []
    efficiency_params    = None
    tune_num_slides      = 0
    tune_timed_elapsed_s = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_wall_start = time.perf_counter()

    fold_ids = (
        [0]
        if args.tcp_tune_cesc_fold0
        else range(cfg.training.num_folds)
    )
    for fold_id in tqdm(fold_ids, desc="Folds"):
        fold = f"fold_{fold_id}"

        merge_path = Path(args.merge_model_path) / fold / "merged_final.pth"
        print(f"\nLoading: {merge_path}")
        base_model.vision_encoder.load_state_dict(
            torch.load(str(merge_path), map_location="cpu")
        )

        task_model_paths = [
            str(Path(args.save_dir) / fold / f"task_{t}.pt")
            for t in range(num_tasks)
        ]
        task_weights = load_task_weights(task_model_paths, device)

        tta_model = CASTSlide(
            backbone             = base_model.vision_encoder,
            task_prompts         = task_prompts,
            task_weights         = task_weights,
            num_classes          = seq_dataset.num_classes,
            device               = device,
            mode                 = args.mode,
            all_class_embeddings = all_class_embeddings,
            param_scope          = args.tta_param_scope,
            M                    = args.M,
            K_sub                = args.K_sub,
            top_ratio            = args.top_ratio,
            alpha                = args.alpha,
            l2_anchor_beta       = args.l2_anchor_beta,
            lr                   = args.lr,
            n_steps              = args.n_steps,
            episodic             = args.episodic,
            entropy_threshold    = args.entropy_threshold,
            use_task_diversity   = args.use_task_diversity,
            use_task_agreement   = (not args.no_task_agreement),
            gamma                = args.gamma,
            select_mode          = args.select_mode,
            use_teacher          = (not args.no_teacher),
            tcp_inference_model  = args.tcp_inference_model,
            naive_inference_model=args.naive_inference_model,
            ema_alpha            = args.ema_alpha,
            adapt_task_prompts   = (not args.no_adapt_prompts),
            ema_alpha_prompt     = args.ema_alpha_prompt,
            delta_margin         = args.delta_margin,
            tp_anchor_beta       = args.tp_anchor_beta,
            gamma_margin         = args.gamma_margin,
            tau_task             = args.tau_task,
            naive_use_task_entropy = args.naive_use_task_entropy,
            use_dapc             = args.use_dapc,
            dapc_loss_weight     = args.dapc_loss_weight,
            class_loss_weight    = args.class_loss_weight,
            entropy_loss_weight  = args.entropy_loss_weight,
            dapc_tau_anchor      = args.dapc_tau_anchor,
            dapc_beta            = args.dapc_beta,
        )
        if efficiency_params is None:
            efficiency_params = {
                "updated_object": f"{args.tta_param_scope} backbone parameters",
                "updated_params": int(tta_model.updated_params),
                "total_params": int(tta_model.total_params),
                "update_ratio": float(tta_model.update_ratio),
                "ln_layers": int(tta_model.num_ln_layers),
            }

        if routing_tune_only:
            if "CESC" not in seq_dataset.task_names:
                raise ValueError(
                    "CESC tuning requires a task named 'CESC', but configured "
                    f"tasks are {seq_dataset.task_names}"
                )
            cesc_task_id = seq_dataset.task_names.index("CESC")
            routing_task_ids = (
                range(num_tasks)
                if args.tcp_tune_routing_all_folds
                else range(cesc_task_id + 1)
            )

            for task_id in routing_task_ids:
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                if (
                    not args.no_reset_prompt_per_task
                    and not args.no_adapt_prompts
                ):
                    tta_model.reset_task_prompts()

                measure_routing = (
                    args.tcp_tune_routing_all_folds
                    or task_id == cesc_task_id
                )
                routing_acc, num_slides, task_elapsed_s = (
                    adapt_task_for_tcp_routing_tune(
                        test_loader=test_loader,
                        task_id=task_id,
                        tta_model=tta_model,
                        device=device,
                        measure_routing=measure_routing,
                    )
                )
                tune_num_slides += num_slides
                tune_timed_elapsed_s += task_elapsed_s
                if measure_routing:
                    all_results.append(
                        {
                            "fold": fold_id,
                            "task_id": task_id,
                            "task_name": seq_dataset.task_names[task_id],
                            "mode": "tcp",
                            "n_samples": num_slides,
                            "routing_acc": routing_acc,
                        }
                    )
                    print(
                        f"===== [Fold {fold_id}] Task {task_id} "
                        f"({seq_dataset.task_names[task_id]}) "
                        f"RoutingAcc={routing_acc * 100:.4f}% ====="
                    )
                else:
                    print(
                        f"[Fold {fold_id}] warm-up Task {task_id} "
                        f"({seq_dataset.task_names[task_id]}): "
                        f"adapted {num_slides} slides"
                    )

            tta_model.hard_reset()
            continue

        all_baccs     = []
        all_accs      = []
        all_aucs      = []
        all_preds_g   = []
        all_targets_g = []
        acc_per_task  = {}
        routing_acc_per_task = {}
        fold_time     = 0.0
        num_total     = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            # Optionally bound prompt drift to the current task stream.
            if (not args.no_reset_prompt_per_task) and (not args.no_adapt_prompts):
                tta_model.reset_task_prompts()

            result = eval_task_tta(
                test_loader          = test_loader,
                task_id              = task_id,
                tta_model            = tta_model,
                task_to_global_class = seq_dataset.task_to_global_class,
                device               = device,
                mode                 = args.mode,
                num_classes_per_task = seq_dataset.num_classes,
                verbose_loss         = args.verbose_loss,
            )
            results, preds_all, targets_all, probs_all, \
                conv_preds, conv_targets, task_time, task_routing_acc, \
                loss_summary = result

            num_total += len(test_loader)
            fold_time += task_time / len(test_loader)

            acc_per_task[task_id] = results["/acc"]
            routing_acc_per_task[task_id] = task_routing_acc
            task_bacc = balanced_accuracy_score(targets_all, preds_all)
            task_acc = sum(preds_all == targets_all) / len(test_loader)
            all_baccs.append(task_bacc)
            all_accs.append(task_acc)
            all_preds_g.append(conv_preds)
            all_targets_g.append(conv_targets)

            metric_line = (
                f"===== [Fold {fold_id}] Task {task_id} "
                f"({seq_dataset.task_names[task_id]}) "
                f"ACC={task_acc*100:.4f}% | BAcc={task_bacc*100:.4f}%"
            )
            if args.mode == "tcp":
                metric_line += f" | RoutingAcc={task_routing_acc*100:.4f}%"
            print(metric_line + " =====")
            if loss_summary:
                loss_line = (
                    "      Loss: "
                    f"total={loss_summary.get('loss/total_with_reg', float('nan')):.6f} | "
                    f"class_ent={loss_summary.get('loss/class_ent', float('nan')):.6f}"
                )
                if args.mode == "tcp" or args.naive_use_task_entropy:
                    loss_line += (
                        f" | task_ent={loss_summary.get('loss/task_ent', float('nan')):.6f}"
                    )
                if args.mode == "tcp":
                    loss_line += (
                        f" | task_agree={loss_summary.get('loss/task_agree', float('nan')):.6f}"
                    )
                loss_line += (
                    f" | l2_anchor={loss_summary.get('loss/l2_anchor', float('nan')):.6f}"
                )
                if args.mode == "tcp":
                    loss_line += (
                        f" | dapc={loss_summary.get('loss/dapc', 0.0):.6f}"
                    )
                loss_line += (
                    f" | adapted={loss_summary.get('adapted_count', 0):.0f}/"
                    f"{loss_summary.get('total_count', 0):.0f}"
                )
                print(loss_line)
                if args.use_dapc:
                    print(
                        "      Reliability: "
                        f"anchor_conf={loss_summary.get('reliability/anchor_conf', float('nan')):.4f} | "
                        f"teacher_conf={loss_summary.get('reliability/teacher_conf', float('nan')):.4f} | "
                        f"agreement={loss_summary.get('reliability/agreement', float('nan'))*100:.2f}% | "
                        f"dapc_active={loss_summary.get('reliability/dapc_active', 0.0)*100:.2f}% | "
                        f"views={loss_summary.get('reliability/dapc_used_views', 0.0)*100:.2f}% | "
                        f"blend={loss_summary.get('reliability/dapc_blended', 0.0)*100:.2f}%"
                    )
                if args.mode == "tcp":
                    print(
                        "      TCP gate: "
                        f"mean_conf={loss_summary.get('tcp_conf_mean', float('nan')):.6f} | "
                        f"fallback={loss_summary.get('tcp_fallback_rate', float('nan'))*100:.2f}% | "
                        f"tau={args.tau_task:.2f}"
                    )
            loss_result_keys = (
                "loss/total_with_reg",
                "loss/total",
                "loss/class_ent",
                "loss/task_ent",
                "loss/task_div",
                "loss/task_agree",
                "loss/l2_anchor",
                "loss/dapc",
                "loss/entropy_objective",
                "loss/task_margin",
            )
            result_row = {
                "fold": fold_id,
                "task_id": task_id,
                "task_name": seq_dataset.task_names[task_id],
                "mode": args.mode,
                "bacc": task_bacc,
                "acc": task_acc,
                "n_samples": len(test_loader),
                "elapsed_s": task_time,
                "routing_acc": task_routing_acc,
                **{
                    key.replace("/", "_"): loss_summary.get(key, float("nan"))
                    for key in loss_result_keys
                },
                "adapted_count": loss_summary.get("adapted_count", 0),
                "total_count": loss_summary.get("total_count", len(test_loader)),
                "tcp_conf_mean": loss_summary.get("tcp_conf_mean", float("nan")),
                "tcp_fallback_rate": loss_summary.get(
                    "tcp_fallback_rate", float("nan")
                ),
                "l2_anchor_beta": args.l2_anchor_beta,
                "regularizer": "l2_anchor",
                "naive_use_task_entropy": args.naive_use_task_entropy,
                "use_dapc": args.use_dapc,
                "dapc_loss_weight": args.dapc_loss_weight,
                "entropy_loss_weight": args.entropy_loss_weight,
                "class_loss_weight": args.class_loss_weight,
                "subbag_select_mode": args.select_mode,
                **{
                    key.replace("/", "_"): loss_summary.get(key, float("nan"))
                    for key in (
                        "reliability/anchor_conf",
                        "reliability/teacher_conf",
                        "reliability/agreement",
                        "reliability/dapc_active",
                        "reliability/dapc_used_views",
                        "reliability/dapc_blended",
                    )
                },
            }
            all_results.append(result_row)

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)

            if args.mode == "tcp":
                # Local class indices
                for i in range(seq_dataset.num_classes[task_id]):
                    all_aucs.append(safe_ovr_auc(targets_all, probs_all, i))
            else:
                # Global class indices (targets_all already in global space)
                global_idxs = sorted(seq_dataset.task_to_global_class[task_id].values())
                for g_idx in global_idxs:
                    all_aucs.append(safe_ovr_auc(targets_all, probs_all, g_idx))

        n_adapted = tta_model.n_adapted
        n_skipped = tta_model.n_skipped
        n_total   = n_adapted + n_skipped
        print(f"[Fold {fold_id}] adapted={n_adapted}/{n_total} "
              f"({100*n_adapted/max(n_total,1):.1f}%) | "
              f"entropy_threshold={args.entropy_threshold}")
        tta_model.hard_reset()

        all_preds_g   = np.concatenate(all_preds_g)
        all_targets_g = np.concatenate(all_targets_g)

        fold_acc = np.mean(all_accs)
        fold_bacc = np.mean(all_baccs)
        fold_macro_f1 = f1_score(all_targets_g, all_preds_g, average="macro")
        fold_weighted_f1 = f1_score(
            all_targets_g, all_preds_g, average="weighted"
        )
        overall_accs.append(fold_acc)
        overall_baccs.append(fold_bacc)
        overall_macro_f1s.append(fold_macro_f1)
        overall_weighted_f1s.append(fold_weighted_f1)
        for result_row in all_results[-num_tasks:]:
            result_row["fold_acc"] = fold_acc
            result_row["fold_bacc"] = fold_bacc
            result_row["fold_macro_f1"] = fold_macro_f1
            result_row["fold_weighted_f1"] = fold_weighted_f1
        global_labels = np.arange(sum(seq_dataset.num_classes))
        overall_recalls.append(recall_score(
            all_targets_g,
            all_preds_g,
            labels=global_labels,
            average=None,
            zero_division=0,
        ))
        overall_precisions.append(precision_score(
            all_targets_g,
            all_preds_g,
            labels=global_labels,
            average=None,
            zero_division=0,
        ))
        overall_aucs.append(np.array(all_aucs))
        overall_times.append(fold_time / num_tasks)
        all_acc_per_task.append(acc_per_task)
        valid_routing = [v for v in routing_acc_per_task.values() if not np.isnan(v)]
        overall_routing_acc.append(np.mean(valid_routing) if valid_routing else float("nan"))
        overall_routing_acc_per_task.append(routing_acc_per_task)

        print(f"[Fold {fold_id}] Acc={np.mean(all_accs)*100:.4f}% "
              f"BAcc={np.mean(all_baccs)*100:.4f}%")

    if routing_tune_only:
        if not all_results:
            raise RuntimeError("TCP routing-only tuning produced no result.")
        routing_values = [
            float(row["routing_acc"]) for row in all_results
        ]
        if args.tcp_tune_cesc_fold0:
            objective_name = "cesc_routing_acc_fold0"
            objective_value = routing_values[0]
            print(
                "\n===== TCP tuning objective: fold-0 CESC RoutingAcc ====="
            )
            print(f"CESC Routing Accuracy: {objective_value * 100:.4f}%")
        else:
            objective_name = "routing_metrics_all_folds"
            objective_value = float(np.mean(routing_values))
            print("\n===== TCP routing-only tuning diagnostics =====")
            for task_id in range(num_tasks):
                values = [
                    float(row["routing_acc"])
                    for row in all_results
                    if int(row["task_id"]) == task_id
                ]
                print(
                    f"Routing Task {task_id} "
                    f"({seq_dataset.task_names[task_id]}): "
                    f"{np.mean(values) * 100:.4f}% "
                    f"({np.std(values) * 100:.4f}%)"
                )
            print(
                f"Routing Accuracy (macro): "
                f"{objective_value * 100:.4f}%"
            )

        if args.result_csv:
            result_csv_path = Path(args.result_csv)
            result_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with result_csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(all_results[0].keys())
                )
                writer.writeheader()
                writer.writerows(all_results)
            print(f"[INFO] Saved result CSV: {result_csv_path}")

        total_elapsed_s = float(time.perf_counter() - eval_wall_start)
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            if device.type == "cuda" else 0.0
        )
        efficiency = {
            "method": "CAST-Slide",
            "eval_setting": (
                "tcp_tune_cesc_fold0"
                if args.tcp_tune_cesc_fold0
                else "tcp_tune_routing_all_folds"
            ),
            "mode": "tcp",
            "objective": objective_name,
            "objective_value": objective_value,
            "num_slides": tune_num_slides,
            "timed_elapsed_s": tune_timed_elapsed_s,
            "wall_elapsed_s": total_elapsed_s,
            "time_per_slide_s": (
                tune_timed_elapsed_s / max(tune_num_slides, 1)
            ),
            "peak_vram_eval_mb": peak_vram_mb,
            **(efficiency_params or {}),
        }
        efficiency_path = Path(args.efficiency_json)
        efficiency_path.parent.mkdir(parents=True, exist_ok=True)
        efficiency_path.write_text(
            json.dumps(efficiency, indent=2), encoding="utf-8"
        )
        print(f"[EFFICIENCY] {json.dumps(efficiency, sort_keys=True)}")
        print(f"[INFO] Saved efficiency JSON: {efficiency_path}")
        raise SystemExit(0)

    reset_label = "episodic" if args.episodic else "continual"
    print(
        f"\n===== Class-IL TTA ({args.mode.upper()}, {reset_label}, "
        f"{args.tta_param_scope}, M={args.M}) ====="
    )
    print(f"Accuracy:       {np.mean(overall_accs)*100:.4f}%"
          f" ({np.std(overall_accs)*100:.4f}%)")
    print(f"Balanced Acc:   {np.mean(overall_baccs)*100:.4f}%"
          f" ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Macro F1:       {np.mean(overall_macro_f1s)*100:.4f}%"
          f" ({np.std(overall_macro_f1s)*100:.4f}%)")
    print(f"Weighted F1:    {np.mean(overall_weighted_f1s)*100:.4f}%"
          f" ({np.std(overall_weighted_f1s)*100:.4f}%)")
    print(f"Inference time: {np.mean(overall_times):.3f}s"
          f" ({np.std(overall_times):.3f}s)")

    metric_labels = class_metric_labels(
        seq_dataset.task_names, seq_dataset.num_classes
    )
    recall_values = np.stack(overall_recalls)
    precision_values = np.stack(overall_precisions)
    auc_values = np.stack(overall_aucs)

    print("\nRecall per class:")
    for label, value, std in zip(
        metric_labels,
        np.nanmean(recall_values, axis=0),
        np.nanstd(recall_values, axis=0),
    ):
        print(f"  {label}: {value*100:.4f}% ({std*100:.4f}%)")

    print("\nPrecision per class:")
    for label, value, std in zip(
        metric_labels,
        np.nanmean(precision_values, axis=0),
        np.nanstd(precision_values, axis=0),
    ):
        print(f"  {label}: {value*100:.4f}% ({std*100:.4f}%)")

    print("\nAUC per class:")
    for label, value, std in zip(
        metric_labels,
        np.nanmean(auc_values, axis=0),
        np.nanstd(auc_values, axis=0),
    ):
        print(f"  {label}: {value*100:.4f}% ({std*100:.4f}%)")

    print("\nAcc per task:")
    accs = {t: [] for t in range(num_tasks)}
    for fold_acc in all_acc_per_task:
        for t in range(num_tasks):
            accs[t].append(fold_acc[t])
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs[t])*100:.4f}%"
              f" ({np.std(accs[t])*100:.4f}%)")

    if args.mode == "tcp":
        print("\nRouting Accuracy per task (pred_task == true_task):")
        routing_by_task = {t: [] for t in range(num_tasks)}
        for fold_r in overall_routing_acc_per_task:
            for t in range(num_tasks):
                v = fold_r.get(t, float("nan"))
                if not np.isnan(v):
                    routing_by_task[t].append(v)
        for t in range(num_tasks):
            vals = routing_by_task[t]
            if vals:
                print(f"  Routing Task {t}: {np.mean(vals)*100:.4f}%"
                      f" ({np.std(vals)*100:.4f}%)")
            else:
                print(f"  Routing Task {t}: N/A")
        overall_routing_valid = [v for v in overall_routing_acc if not np.isnan(v)]
        if overall_routing_valid:
            print(f"Routing Accuracy (mean): {np.mean(overall_routing_valid)*100:.4f}%"
                  f" ({np.std(overall_routing_valid)*100:.4f}%)")

    if args.result_csv:
        result_csv_path = Path(args.result_csv)
        result_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with result_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[INFO] Saved result CSV: {result_csv_path}")

    total_elapsed_s = float(time.perf_counter() - eval_wall_start)
    total_slide_count = int(sum(row["n_samples"] for row in all_results))
    timed_elapsed_s = float(sum(row["elapsed_s"] for row in all_results))
    time_per_slide_s = timed_elapsed_s / max(total_slide_count, 1)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    efficiency = {
        "method": "CAST-Slide",
        "eval_setting": "class_il",
        "mode": args.mode,
        "param_scope": args.tta_param_scope,
        "tta_steps": int(args.n_steps),
        "patches_per_wsi": int(K_PATCHES),
        "subbags": int(args.M),
        "patches_per_subbag": int(args.K_sub),
        "num_slides": total_slide_count,
        "timing_scope": "per-slide online TTA update plus final prediction; checkpoint/model setup excluded",
        "timing_cuda_synchronized": device.type == "cuda",
        "adapt_merge_elapsed_s": None,
        "inference_only_elapsed_s": None,
        "online_adapt_inference_elapsed_s": timed_elapsed_s,
        "end_to_end_elapsed_s": timed_elapsed_s,
        "timed_elapsed_s": timed_elapsed_s,
        "wall_elapsed_s": total_elapsed_s,
        "inference_only_time_per_slide_s": None,
        "end_to_end_time_per_slide_s": time_per_slide_s,
        "time_per_slide_s": time_per_slide_s,
        "end_to_end_throughput_slides_per_s": total_slide_count / max(timed_elapsed_s, 1e-12),
        "throughput_slides_per_s": total_slide_count / max(timed_elapsed_s, 1e-12),
        "peak_vram_eval_mb": peak_vram_mb,
        "peak_vram_adapt_mb": peak_vram_mb,
        "backprop": True,
        "source_free": True,
        "label_free": True,
        **(efficiency_params or {}),
    }
    efficiency_path = Path(args.efficiency_json)
    efficiency_path.parent.mkdir(parents=True, exist_ok=True)
    efficiency_path.write_text(json.dumps(efficiency, indent=2), encoding="utf-8")
    print(f"[EFFICIENCY] {json.dumps(efficiency, sort_keys=True)}")
    print(f"[INFO] Saved efficiency JSON: {efficiency_path}")
