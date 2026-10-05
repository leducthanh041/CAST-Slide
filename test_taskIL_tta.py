"""TASK-IL evaluation for CAST-Slide."""
"""
Task-IL TTA evaluation (upper bound with adaptation).

Task identity is known at inference time, so:
  - No task routing or task-prompt dependency
  - Confident sub-bags are selected using class predictions only
  - Loss = class entropy + class-only source/L2 anchoring

Usage:
    python test_taskIL_tta.py \\
        --save_dir ./checkpoints/finetuned \\
        --merge_model_path ./checkpoints/merged
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from cast_slide.constants import EMBED_DIM, K_PATCHES, NUM_TASKS, TITAN_PS_ARG
from cast_slide.datasets import Sequential_Generic_MIL_Dataset
from cast_slide.metrics import pad_numpy_arrays
from cast_slide.utils import get_eval_metrics, seed_torch
from cast_slide.tta_adapter import CASTSlide, load_task_weights

PROJECT_ROOT = Path(__file__).resolve().parent
HOT_DIR_NAMES = {"checkpoints", "logs", "sqlite"}


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


def eval_task_taskil_tta(
    test_loader,
    task_id:     int,
    tta_model:   CASTSlide,
    device,
    verbose_loss: bool = False,
) -> tuple:
    """
    Task-IL TTA inference for 1 task.
    task_id is known -> tta_model uses mode='task_il' with fixed_task_id=task_id.
    pred_class is LOCAL (0..C_task-1), targets are LOCAL.
    """
    preds_all   = []
    probs_all   = []
    targets_all = []
    loss_logs   = []
    elapsed_s   = 0.0

    for features, coords, label in tqdm(test_loader, leave=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        slide_t0 = time.perf_counter()
        features = features.to(device)
        coords   = coords.long().to(device)

        idx = torch.randperm(features.shape[0])[:K_PATCHES]
        features, coords = features[idx], coords[idx]

        pred_class, probs, _, adapt_log = tta_model.adapt_and_predict(
            features, coords
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_s += time.perf_counter() - slide_t0
        if verbose_loss:
            loss_logs.append(adapt_log)

        preds_all.append(np.array([pred_class]))
        probs_all.append(probs.numpy())
        targets_all.append(label.numpy())

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

    if verbose_loss and loss_logs:
        adapted = [d for d in loss_logs if d.get("slide/adapted")]
        if adapted:
            mean_loss = np.mean([d.get("loss/total_with_reg", 0) for d in adapted])
            mean_source_anchor = np.mean([
                d.get("loss/taskil_source_anchor", 0) for d in adapted
            ])
            print(f"    [TTA] task={task_id} adapted={len(adapted)}/{len(loss_logs)} "
                  f"mean_loss={mean_loss:.4f} "
                  f"source_anchor={mean_source_anchor:.4f}")

    return metrics, preds_arr, targets_arr, probs_arr, elapsed_s


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Task-IL TTA evaluation (upper bound)")
    parser.add_argument("--config",           type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir",         type=str, required=True)
    parser.add_argument("--merge_model_path", type=str, required=True)
    # Adaptation parameters
    parser.add_argument("--M",                 type=int,   default=8)
    parser.add_argument("--K_sub",             type=int,   default=300)
    parser.add_argument("--top_ratio",         type=float, default=0.5)
    parser.add_argument("--beta",              type=float, default=1.0)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--n_steps",           type=int,   default=1)
    parser.add_argument("--tta_param_scope",   type=str,   default="ln_only",
                        choices=["ln_only", "full"],
                        help="Backbone parameter scope for TTA.")
    parser.add_argument("--entropy_threshold", type=float, default=0.4)
    parser.add_argument(
        "--taskil_source_anchor_weight",
        type=float,
        default=1.0,
        help="Weight for Task-IL consistency with the frozen merged source prediction.",
    )
    parser.add_argument(
        "--episodic",
        action="store_true",
        help="[Deprecated/Ignored] CAST-Slide uses continual adaptation.",
    )
    parser.add_argument("--verbose_loss",      action="store_true")
    parser.add_argument(
        "--efficiency_json",
        type=str,
        default="",
        help="Optional JSON path to save updated params, TTA steps, throughput, and peak VRAM.",
    )
    args = parser.parse_args()
    if args.episodic:
        print("[WARN] --episodic is ignored; running continual adaptation without reset.")
    args.episodic = False

    local_hot_root        = ensure_local_hot_storage()
    args.save_dir         = str(resolve_hot_path(args.save_dir,         local_hot_root))
    args.merge_model_path = str(resolve_hot_path(args.merge_model_path, local_hot_root))
    if args.efficiency_json:
        args.efficiency_json = str(resolve_hot_path(args.efficiency_json, local_hot_root))
    else:
        args.efficiency_json = str(Path(args.merge_model_path) / "efficiency_taskil_tta.json")

    cfg    = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, cfg.training.seed)

    num_tasks   = cfg.training.num_tasks
    seq_dataset = Sequential_Generic_MIL_Dataset(cfg)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_model = base_model.to(device)

    overall_baccs    = []
    overall_accs     = []
    overall_recalls  = []
    overall_precisions = []
    overall_aucs     = []
    all_acc_per_task = []
    efficiency_params = None
    total_timed_s = 0.0
    total_slides = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eval_wall_start = time.perf_counter()

    for fold_id in tqdm(range(cfg.training.num_folds), desc="Folds"):
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

        all_baccs    = []
        all_accs     = []
        fold_recalls = []
        fold_precisions = []
        fold_aucs    = []
        acc_per_task = {}

        for task_id in range(num_tasks):
            tta_model = CASTSlide(
                backbone          = base_model.vision_encoder,
                task_prompts      = None,
                task_weights      = task_weights,
                num_classes       = seq_dataset.num_classes,
                device            = device,
                mode              = "task_il",
                fixed_task_id     = task_id,
                param_scope       = args.tta_param_scope,
                M                 = args.M,
                K_sub             = args.K_sub,
                top_ratio         = args.top_ratio,
                l2_anchor_beta    = args.beta,
                lr                = args.lr,
                n_steps           = args.n_steps,
                episodic          = args.episodic,
                entropy_threshold = args.entropy_threshold,
                taskil_source_anchor_weight=args.taskil_source_anchor_weight,
            )
            if efficiency_params is None:
                efficiency_params = {
                    "updated_object": f"{args.tta_param_scope} backbone parameters",
                    "updated_params": int(tta_model.updated_params),
                    "total_params": int(tta_model.total_params),
                    "update_ratio": float(tta_model.update_ratio),
                    "ln_layers": int(tta_model.num_ln_layers),
                }

            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            results, preds_all, targets_all, probs_all, task_elapsed = eval_task_taskil_tta(
                test_loader  = test_loader,
                task_id      = task_id,
                tta_model    = tta_model,
                device       = device,
                verbose_loss = args.verbose_loss,
            )
            total_timed_s += task_elapsed
            total_slides += len(test_loader)

            bacc = balanced_accuracy_score(targets_all, preds_all)
            acc  = sum(preds_all == targets_all) / len(test_loader)
            acc_per_task[task_id] = acc
            all_baccs.append(bacc)
            all_accs.append(acc)

            local_labels = np.arange(seq_dataset.num_classes[task_id])
            fold_recalls.extend(recall_score(
                targets_all,
                preds_all,
                labels=local_labels,
                average=None,
                zero_division=0,
            ).tolist())
            fold_precisions.extend(precision_score(
                targets_all,
                preds_all,
                labels=local_labels,
                average=None,
                zero_division=0,
            ).tolist())
            fold_aucs.extend(
                safe_ovr_auc(targets_all, probs_all, class_id)
                for class_id in local_labels
            )

            n_adapted = tta_model.n_adapted
            n_total   = n_adapted + tta_model.n_skipped
            print(f"  Fold {fold_id} | {seq_dataset.task_names[task_id]}: "
                  f"BAcc={bacc*100:.4f}% Acc={acc*100:.4f}% "
                  f"adapted={n_adapted}/{n_total}")

            tta_model.hard_reset()

        overall_baccs.append(np.mean(all_baccs))
        overall_accs.append(np.mean(all_accs))
        overall_recalls.append(np.asarray(fold_recalls, dtype=float))
        overall_precisions.append(np.asarray(fold_precisions, dtype=float))
        overall_aucs.append(np.asarray(fold_aucs, dtype=float))
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] BAcc={np.mean(all_baccs)*100:.4f}% "
              f"Acc={np.mean(all_accs)*100:.4f}%")

    print(f"\n===== Task-IL TTA Results ({args.tta_param_scope}) =====")
    print(f"Balanced Acc: {np.mean(overall_baccs)*100:.4f}%"
          f" ({np.std(overall_baccs)*100:.4f}%)")
    print(f"Accuracy:     {np.mean(overall_accs)*100:.4f}%"
          f" ({np.std(overall_accs)*100:.4f}%)")

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
        print(f"  {seq_dataset.task_names[t]}: {np.mean(accs[t])*100:.4f}%"
              f" ({np.std(accs[t])*100:.4f}%)")

    total_elapsed_s = float(time.perf_counter() - eval_wall_start)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    efficiency = {
        "method": "CAST-Slide",
        "eval_setting": "task_il",
        "mode": "task_il",
        "param_scope": args.tta_param_scope,
        "tta_steps": int(args.n_steps),
        "selection": "class_confidence",
        "patches_per_wsi": int(K_PATCHES),
        "subbags": int(args.M),
        "patches_per_subbag": int(args.K_sub),
        "num_slides": int(total_slides),
        "timing_scope": "per-slide online TTA update plus final prediction; checkpoint/model setup excluded",
        "timing_cuda_synchronized": device.type == "cuda",
        "adapt_merge_elapsed_s": None,
        "inference_only_elapsed_s": None,
        "online_adapt_inference_elapsed_s": float(total_timed_s),
        "end_to_end_elapsed_s": float(total_timed_s),
        "timed_elapsed_s": float(total_timed_s),
        "wall_elapsed_s": total_elapsed_s,
        "inference_only_time_per_slide_s": None,
        "end_to_end_time_per_slide_s": float(total_timed_s / max(total_slides, 1)),
        "time_per_slide_s": float(total_timed_s / max(total_slides, 1)),
        "end_to_end_throughput_slides_per_s": float(total_slides / max(total_timed_s, 1e-12)),
        "throughput_slides_per_s": float(total_slides / max(total_timed_s, 1e-12)),
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
