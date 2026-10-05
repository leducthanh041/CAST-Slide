"""Create a six-task by six-stage TITAN attention-map figure.

One single-specimen WSI is selected from each task's fold-4 training split.
Selection is independent of classification correctness. The selected slides
are then held fixed while native TITAN pooling attention is extracted from
every continual merged checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from cast_slide.constants import (
    K_PATCHES,
    TASK_NAMES,
    TASK_TO_GLOBAL_CLASS,
)
from cast_slide.datasets import Sequential_Generic_MIL_Dataset
from cast_slide.titan_attention import TitanSelectedHeadCapture
from create_cast_slide_heatmaps import (
    PROJECT_ROOT,
    aggregate_scores,
    make_tta_model,
    render_maps,
    representative_indices,
    resolve,
    save_clam_blockmaps,
    save_h5,
    score_attention_context_bag,
)


def load_args(config_path: Path, validate_only: bool):
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    values = cfg["runtime"] | cfg["tta"] | cfg["visualization"] | cfg["selection"]
    values["raw_roots"] = cfg["raw_roots"]
    values["validate_only"] = validate_only
    values["config"] = str(config_path)
    return SimpleNamespace(**values)


def slide_rows(split):
    if hasattr(split, "slide_data"):
        for row in split.slide_data.itertuples(index=False):
            yield str(row.slide_id).removesuffix(".svs"), int(row.label)
    else:
        for slide_id, label in zip(split.data, split.label):
            yield str(slide_id).removesuffix(".svs"), int(label)


def selected_split(dataset, split_path: str, split_name: str):
    splits = dataset.return_splits(from_id=False, csv_path=split_path)
    split_by_name = dict(zip(("train", "val", "test"), splits))
    if split_name not in split_by_name:
        raise ValueError(f"Unsupported selection split: {split_name}")
    split = split_by_name[split_name]
    if split is None:
        raise RuntimeError(f"Split '{split_name}' is empty in {split_path}")
    return split


def index_raw_slides(raw_roots):
    exact, barcode = {}, {}
    for raw_root_value in raw_roots:
        raw_root = resolve(raw_root_value)
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw WSI root not found: {raw_root}")
        for path in raw_root.glob("*.svs"):
            exact.setdefault(path.stem, []).append(path)
            barcode.setdefault(path.stem.split(".", 1)[0], []).append(path)
    return exact, barcode


def resolve_raw_slide(slide_id, exact, barcode):
    exact_matches = exact.get(slide_id, [])
    if len(exact_matches) == 1:
        return exact_matches[0], "exact_slide_id"
    prefix_matches = barcode.get(slide_id.split(".", 1)[0], [])
    if len(prefix_matches) == 1:
        return prefix_matches[0], "tcga_barcode_fallback"
    return None, "missing" if not prefix_matches else "ambiguous_barcode"


def feature_path(seq_dataset, task_id, slide_id):
    return Path(seq_dataset.datasets[task_id].data_dir) / "h5_files" / f"{slide_id}.h5"


def feature_geometry(path: Path):
    with h5py.File(path, "r") as handle:
        coords = handle["coords"][:]
    return int(len(coords)), coords.min(axis=0), coords.max(axis=0)


def specimen_qc(slide_path: Path, max_size: int):
    import cv2
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    try:
        slide_dimensions = tuple(map(int, slide.dimensions))
        thumb = slide.get_thumbnail((max_size, max_size)).convert("RGB")
    finally:
        slide.close()
    rgb = np.asarray(thumb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    tissue = ((hsv[..., 1] >= 18) & (hsv[..., 2] <= 245)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    min_area = max(64, int(0.0015 * tissue.size))
    components = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
         if stats[i, cv2.CC_STAT_AREA] >= min_area),
        reverse=True,
    )
    areas = [area for area, _ in components]
    total = max(1, sum(areas))
    largest_fraction = areas[0] / total if areas else 0.0
    second_fraction = areas[1] / total if len(areas) > 1 else 0.0
    if components:
        largest_id = components[0][1]
        tissue_x = int(stats[largest_id, cv2.CC_STAT_LEFT])
        tissue_y = int(stats[largest_id, cv2.CC_STAT_TOP])
        tissue_width = int(stats[largest_id, cv2.CC_STAT_WIDTH])
        tissue_height = int(stats[largest_id, cv2.CC_STAT_HEIGHT])
    else:
        tissue_x = tissue_y = 0
        tissue_width = tissue_height = 0
    tissue_aspect_ratio = tissue_width / max(1, tissue_height)
    metrics = {
        "significant_components": len(areas),
        "largest_tissue_fraction": largest_fraction,
        "second_tissue_fraction": second_fraction,
        "dominant_tissue_width": tissue_width,
        "dominant_tissue_height": tissue_height,
        "dominant_tissue_x": tissue_x,
        "dominant_tissue_y": tissue_y,
        "dominant_tissue_aspect_ratio": tissue_aspect_ratio,
        "thumbnail_width": thumb.width,
        "thumbnail_height": thumb.height,
        "slide_width": slide_dimensions[0],
        "slide_height": slide_dimensions[1],
        "tissue_component_boxes": [
            {
                "x": int(stats[index, cv2.CC_STAT_LEFT]),
                "y": int(stats[index, cv2.CC_STAT_TOP]),
                "width": int(stats[index, cv2.CC_STAT_WIDTH]),
                "height": int(stats[index, cv2.CC_STAT_HEIGHT]),
                "area": int(area),
            }
            for area, index in components
        ],
    }
    return thumb, metrics


def is_valid_single_wsi(metrics, args):
    """Require visible tissue; output is cropped to its dominant specimen."""
    return metrics["significant_components"] >= 1


def coordinates_fit_wsi(coord_max, qc, patch_size):
    return (
        float(coord_max[0] + patch_size) <= 1.05 * qc["slide_width"]
        and float(coord_max[1] + patch_size) <= 1.05 * qc["slide_height"]
    )


def load_features(path: Path, device=None):
    with h5py.File(path, "r") as handle:
        features = torch.from_numpy(handle["features"][:].astype(np.float32, copy=False))
        coords = torch.from_numpy(handle["coords"][:].astype(np.int64, copy=False)).long()
    if device is not None:
        features = features.to(device)
        coords = coords.to(device)
    return features, coords


def extract_attention_scores(model, features, coords, coords_np, args):
    """Extract the configured native TITAN attention on all WSI patches."""
    backend = getattr(args, "attention_backend", "pooling_coverage")
    if backend == "pooling_coverage":
        return aggregate_scores(model, features, coords, coords_np, args)
    if backend != "selected_transformer_head":
        raise ValueError(f"Unsupported attention_backend={backend}")
    layer_index = int(args.selected_layer_index)
    head_index = int(args.selected_head_index)
    model.backbone.eval()
    print(
        f"[HEAD-ATTENTION] patches={len(coords_np)} layer_index={layer_index} "
        f"head_index={head_index}",
        flush=True,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with TitanSelectedHeadCapture(
            model.backbone, [layer_index], [head_index]
        ) as capture:
            model.backbone(features, coords, model.ps)
        raw = capture.patch_scores(coords, int(model.ps.item()))[
            (layer_index, head_index)
        ]
    # Keep the established storage/render key for backward-compatible layout;
    # metadata records that these are selected-head, not pooling, scores.
    return {
        "titan_pool_attention_median": raw.float().cpu().numpy().astype(
            np.float32, copy=False
        )
    }


def naive_prediction(model, features, coords, args, reset_before_slide=True):
    indices_np = representative_indices(
        coords.cpu().numpy(), args.k_patches, args.seed, args.spatial_bins
    )
    indices = torch.as_tensor(indices_np, device=features.device)
    if reset_before_slide:
        model.hard_reset()
    pred_class, probs, pred_task, adaptation = model.adapt_and_predict(
        features[indices], coords[indices]
    )
    return {
        "predicted_global_class": int(pred_class),
        "predicted_task": int(pred_task),
        "confidence": float(probs.max().item()),
        "adaptation": adaptation,
        "adaptation_indices": indices_np,
    }


def attention_change_metrics(before, after, top_fraction):
    """Summarize a causal pre/post-WSl attention change on identical patches."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(f"Attention shape mismatch: {before.shape} vs {after.shape}")
    before_rank = rank_values(before)
    after_rank = rank_values(after)
    corr = float(np.corrcoef(before_rank, after_rank)[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    top_k = max(1, int(np.ceil(top_fraction * len(before))))
    before_top = set(np.argpartition(before, len(before) - top_k)[-top_k:].tolist())
    after_top = set(np.argpartition(after, len(after) - top_k)[-top_k:].tolist())
    return {
        "mean_absolute_delta": float(np.mean(np.abs(after - before))),
        "maximum_absolute_delta": float(np.max(np.abs(after - before))),
        "rank_spearman": corr,
        "top_patch_jaccard": len(before_top & after_top) / max(1, len(before_top | after_top)),
        "top_fraction": float(top_fraction),
    }


def write_manifest(rows, output_dir: Path):
    json_path = output_dir / "selected_slides.json"
    with json_path.open("w") as handle:
        json.dump(
            rows, handle, indent=2, allow_nan=False,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray) else value,
        )
    csv_path = output_dir / "selected_slides.csv"
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"qc", "selection_prediction"}}
        flat.update({f"qc_{k}": v for k, v in row["qc"].items()})
        flat.update({
            f"prediction_{k}": v
            for k, v in row["selection_prediction"].items()
            if k not in {"adaptation", "adaptation_indices"}
        })
        flat_rows.append(flat)
    if flat_rows:
        fieldnames = []
        for row in flat_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def load_fixed_selection(manifest_path: Path, output_dir: Path):
    with manifest_path.open() as handle:
        selected = json.load(handle)
    if len(selected) != len(TASK_NAMES):
        raise RuntimeError(
            f"Fixed manifest must contain {len(TASK_NAMES)} WSIs, got {len(selected)}"
        )
    by_task = {row["task_name"]: row for row in selected}
    missing_tasks = [name for name in TASK_NAMES if name not in by_task]
    if missing_tasks:
        raise RuntimeError(f"Fixed manifest is missing tasks: {missing_tasks}")
    selected = [by_task[name] for name in TASK_NAMES]
    for row in selected:
        for key in ("raw_wsi", "feature_h5"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"Fixed {key} not found: {row[key]}")
        original = output_dir / row["task_name"] / "original_wsi.jpg"
        if not original.is_file():
            raise FileNotFoundError(f"Fixed original JPG not found: {original}")
    print(f"[FIXED] reusing six selected WSIs from {manifest_path}", flush=True)
    return selected


def build_override_selection(seq_dataset, raw_index, args, output_dir):
    """Resolve an explicit six-task slide manifest without modifying source data."""
    overrides = dict(getattr(args, "slide_overrides", {}) or {})
    missing = [name for name in TASK_NAMES if name not in overrides]
    if missing:
        raise ValueError(f"slide_overrides must specify all six tasks; missing={missing}")
    exact, barcode = raw_index
    selected = []
    qc_dir = output_dir / "selection_qc_online_causal"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for task_id, task_name in enumerate(TASK_NAMES):
        requested = str(overrides[task_name]).removesuffix(".svs")
        split_path = seq_dataset._split_csv(task_id, args.fold)
        found = None
        for split_name, split in zip(
            ("train", "val", "test"),
            seq_dataset.datasets[task_id].return_splits(
                from_id=False, csv_path=split_path
            ),
        ):
            if split is None:
                continue
            for slide_id, local_label in slide_rows(split):
                if slide_id == requested:
                    found = (split_name, local_label)
                    break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(
                f"Override {task_name} slide is absent from fold-{args.fold} splits: "
                f"{requested}"
            )
        split_name, local_label = found
        raw_path, raw_match = resolve_raw_slide(requested, exact, barcode)
        feat_path = feature_path(seq_dataset, task_id, requested)
        if raw_path is None or not feat_path.is_file():
            raise FileNotFoundError(
                f"Override lacks raw/features: task={task_name} slide={requested} "
                f"raw={raw_path} features={feat_path}"
            )
        patches, coord_min, coord_max = feature_geometry(feat_path)
        thumbnail, qc = specimen_qc(raw_path, args.qc_thumbnail_size)
        qc["feature_coord_min"] = coord_min.tolist()
        qc["feature_coord_max"] = coord_max.tolist()
        qc["coordinates_fit_raw_wsi"] = coordinates_fit_wsi(
            coord_max, qc, args.vis_patch_size
        )
        if not qc["coordinates_fit_raw_wsi"]:
            raise RuntimeError(f"Raw/feature geometry mismatch for {requested}")
        qc_path = qc_dir / f"{task_id + 1}_{task_name}_{requested}.jpg"
        thumbnail.save(qc_path, quality=95)
        selected.append({
            "task_id": task_id,
            "task_name": task_name,
            "slide_id": requested,
            "local_label": int(local_label),
            "true_global_class": TASK_TO_GLOBAL_CLASS[task_id][int(local_label)],
            "split": split_name,
            "fold": args.fold,
            "raw_wsi": str(raw_path),
            "raw_match": raw_match,
            "feature_h5": str(feat_path),
            "num_patches": patches,
            "qc_thumbnail": str(qc_path),
            "qc": qc,
            "selection_prediction": {
                "selection_requires_correct_prediction": False,
                "selection_method": "explicit_online_causal_override",
            },
            "selection_rejections": {},
        })
        print(
            f"[OVERRIDE] task={task_name} split={split_name} slide={requested} "
            f"patches={patches} largest={qc['largest_tissue_fraction']:.4f} "
            f"second={qc['second_tissue_fraction']:.4f}",
            flush=True,
        )
    return selected


def build_candidate_pools(seq_dataset, raw_index, args, output_dir):
    pools = []
    qc_dir = output_dir / "selection_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    exact, barcode = raw_index
    for task_id, task_name in enumerate(TASK_NAMES):
        split_path = seq_dataset._split_csv(task_id, args.fold)
        task_split = selected_split(
            seq_dataset.datasets[task_id], split_path, args.selection_split
        )
        candidates = []
        rejection_counts = {}
        for slide_id, local_label in slide_rows(task_split):
            raw_path, raw_match = resolve_raw_slide(slide_id, exact, barcode)
            feat_path = feature_path(seq_dataset, task_id, slide_id)
            if raw_path is None or not feat_path.is_file():
                reason = raw_match if raw_path is None else "missing_feature"
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            patches, coord_min, coord_max = feature_geometry(feat_path)
            if patches < args.min_patches or patches > args.max_patches:
                rejection_counts["patch_count"] = rejection_counts.get("patch_count", 0) + 1
                continue
            candidates.append((abs(patches - args.preferred_patches), slide_id,
                               local_label, raw_path, raw_match, feat_path, patches,
                               coord_min, coord_max))
        candidates.sort()
        correct_candidates = []
        checked = 0
        for (_, slide_id, local_label, raw_path, raw_match, feat_path, patches,
             coord_min, coord_max) in candidates:
            if checked >= args.max_candidates_per_task:
                break
            thumbnail, qc = specimen_qc(raw_path, args.qc_thumbnail_size)
            qc["feature_coord_min"] = coord_min.tolist()
            qc["feature_coord_max"] = coord_max.tolist()
            qc["coordinates_fit_raw_wsi"] = coordinates_fit_wsi(
                coord_max, qc, args.vis_patch_size
            )
            if not qc["coordinates_fit_raw_wsi"]:
                rejection_counts["raw_feature_geometry_mismatch"] = (
                    rejection_counts.get("raw_feature_geometry_mismatch", 0) + 1
                )
                continue
            if not is_valid_single_wsi(qc, args):
                rejection_counts["no_visible_tissue"] = (
                    rejection_counts.get("no_visible_tissue", 0) + 1
                )
                continue
            checked += 1
            true_global = TASK_TO_GLOBAL_CLASS[task_id][local_label]
            qc_path = qc_dir / (
                f"{task_id + 1}_{task_name}_{len(correct_candidates) + 1}_"
                f"{slide_id}.jpg"
            )
            thumbnail.save(qc_path, quality=95)
            candidate = {
                "task_id": task_id,
                "task_name": task_name,
                "slide_id": slide_id,
                "local_label": local_label,
                "true_global_class": true_global,
                "split": args.selection_split,
                "fold": args.fold,
                "raw_wsi": str(raw_path),
                "raw_match": raw_match,
                "feature_h5": str(feat_path),
                "num_patches": patches,
                "qc_thumbnail": str(qc_path),
                "qc": qc,
                "selection_prediction": {
                    "selection_requires_correct_prediction": False,
                },
                "selection_rejections": rejection_counts,
            }
            correct_candidates.append(candidate)
            print(
                f"[POOL] {task_name} candidate={len(correct_candidates)}/"
                f"{args.selection_pool_size}: {slide_id} patches={patches} "
                f"global={true_global} single_specimen=yes"
            )
            if len(correct_candidates) >= args.selection_pool_size:
                break
        if not correct_candidates:
            raise RuntimeError(
                f"No single-specimen candidate found for {task_name}; "
                f"checked={checked}, rejections={rejection_counts}"
            )
        pools.append(correct_candidates)
        with (output_dir / "candidate_pools.json").open("w") as handle:
            json.dump(
                pools, handle, indent=2, allow_nan=False,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray) else value,
            )
    return pools


def rank_values(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def attention_stage_metrics(scores, coords, top_fraction):
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.clip(scores, 0, None)
    probs = scores / max(float(scores.sum()), 1e-12)
    n = len(probs)
    top_k = max(1, int(np.ceil(top_fraction * n)))
    top_indices = np.argpartition(probs, n - top_k)[-top_k:]
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    focus = 1.0 - entropy / max(np.log(max(2, n)), 1e-12)
    lo = coords.min(axis=0).astype(np.float64)
    span = np.maximum(coords.max(axis=0) - lo, 1).astype(np.float64)
    top_coords = (coords[top_indices] - lo) / span
    top_span = np.maximum(top_coords.max(axis=0) - top_coords.min(axis=0), 0)
    spatial_compactness = 1.0 - float(np.prod(top_span))
    return {
        "focus": focus,
        "top_attention_mass": float(probs[top_indices].sum()),
        "spatial_compactness": spatial_compactness,
        "top_indices": np.sort(top_indices),
        "ranks": rank_values(scores),
    }


def score_candidate_attention(stage_entries, args):
    stage_metrics = [
        attention_stage_metrics(entry["scores"], entry["coords"], args.attention_top_fraction)
        for entry in stage_entries
    ]
    correlations, jaccards = [], []
    for left in range(len(stage_metrics)):
        for right in range(left + 1, len(stage_metrics)):
            corr = float(np.corrcoef(
                stage_metrics[left]["ranks"], stage_metrics[right]["ranks"]
            )[0, 1])
            correlations.append(0.0 if not np.isfinite(corr) else corr)
            a = set(stage_metrics[left]["top_indices"].tolist())
            b = set(stage_metrics[right]["top_indices"].tolist())
            jaccards.append(len(a & b) / max(1, len(a | b)))
    mean_focus = float(np.mean([entry["focus"] for entry in stage_metrics]))
    mean_mass = float(np.mean([entry["top_attention_mass"] for entry in stage_metrics]))
    mean_compactness = float(np.mean([
        entry["spatial_compactness"] for entry in stage_metrics
    ]))
    mean_corr = float(np.mean(correlations))
    mean_jaccard = float(np.mean(jaccards))
    # Correlation is mapped from [-1, 1] to [0, 1]. No image-space smoothing
    # or score manipulation is used to make the result appear consistent.
    selection_score = (
        0.20 * mean_focus
        + 0.15 * mean_mass
        + 0.25 * mean_compactness
        + 0.25 * ((mean_corr + 1.0) / 2.0)
        + 0.15 * mean_jaccard
    )
    return {
        "selection_score": selection_score,
        "mean_focus": mean_focus,
        "mean_top_attention_mass": mean_mass,
        "mean_spatial_compactness": mean_compactness,
        "mean_stage_rank_correlation": mean_corr,
        "mean_top_region_jaccard": mean_jaccard,
        "per_stage": [
            {key: value for key, value in metrics.items()
             if key not in {"top_indices", "ranks"}}
            for metrics in stage_metrics
        ],
    }


def select_by_attention_consistency(candidate_pools, args, device, output_dir):
    pilot = {
        (task_id, candidate_id): []
        for task_id, pool in enumerate(candidate_pools)
        for candidate_id in range(len(pool))
    }
    for stage in range(1, 7):
        checkpoint = stage_checkpoint(args, stage)
        model, _, _ = make_tta_model(args, device, checkpoint)
        for task_id, pool in enumerate(candidate_pools):
            for candidate_id, candidate in enumerate(pool):
                print(
                    f"[PILOT] stage={stage}/6 task={candidate['task_name']} "
                    f"candidate={candidate_id + 1}/{len(pool)} "
                    f"slide={candidate['slide_id']}",
                    flush=True,
                )
                features, coords = load_features(Path(candidate["feature_h5"]), device)
                prediction = naive_prediction(model, features, coords, args)
                indices = torch.as_tensor(
                    prediction["adaptation_indices"], device=device
                )
                attention = extract_attention_scores(
                    model, features[indices], coords[indices],
                    coords[indices].cpu().numpy(), args,
                )["titan_pool_attention_median"]
                pilot[(task_id, candidate_id)].append({
                    "stage": stage,
                    "scores": attention,
                    "coords": coords[indices].cpu().numpy(),
                    "predicted_global_class": prediction["predicted_global_class"],
                })
                del features, coords
                torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()

    selected, ranking_rows = [], []
    for task_id, pool in enumerate(candidate_pools):
        ranked = []
        for candidate_id, candidate in enumerate(pool):
            metrics = score_candidate_attention(pilot[(task_id, candidate_id)], args)
            candidate["attention_selection"] = metrics
            ranked.append(candidate)
        ranked.sort(
            key=lambda item: item["attention_selection"]["selection_score"],
            reverse=True,
        )
        chosen = ranked[0]
        selected.append(chosen)
        for rank, candidate in enumerate(ranked, start=1):
            ranking_rows.append({
                "task": candidate["task_name"],
                "rank": rank,
                "slide_id": candidate["slide_id"],
                **{key: value for key, value in candidate["attention_selection"].items()
                   if key != "per_stage"},
                "selected": rank == 1,
            })
        print(
            f"[SELECTED] {chosen['task_name']}: {chosen['slide_id']} "
            f"attention_score={chosen['attention_selection']['selection_score']:.4f} "
            f"rank_corr={chosen['attention_selection']['mean_stage_rank_correlation']:.4f} "
            f"top_jaccard={chosen['attention_selection']['mean_top_region_jaccard']:.4f}"
        )
    with (output_dir / "candidate_attention_ranking.json").open("w") as handle:
        json.dump(ranking_rows, handle, indent=2, allow_nan=False)
    with (output_dir / "candidate_attention_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking_rows[0]))
        writer.writeheader(); writer.writerows(ranking_rows)
    write_manifest(selected, output_dir)
    return selected


def build_cesc_candidate_pool(seq_dataset, raw_index, args, output_dir):
    """Collect fold-4 CESC candidates, preferring test then val then train."""
    task_id = TASK_NAMES.index("CESC")
    exact, barcode = raw_index
    split_path = seq_dataset._split_csv(task_id, args.fold)
    splits = seq_dataset.datasets[task_id].return_splits(
        from_id=False, csv_path=split_path
    )
    by_name = dict(zip(("train", "val", "test"), splits))
    split_order = list(getattr(args, "cesc_candidate_splits", ["test", "val", "train"]))
    excluded = {
        str(value).removesuffix(".svs")
        for value in getattr(args, "cesc_exclude_slides", [])
    }
    candidates = []
    qc_dir = output_dir / "CESC_candidate_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    for split_rank, split_name in enumerate(split_order):
        split = by_name.get(split_name)
        if split is None:
            continue
        for slide_id, local_label in slide_rows(split):
            if slide_id in excluded:
                continue
            raw_path, raw_match = resolve_raw_slide(slide_id, exact, barcode)
            feat_path = feature_path(seq_dataset, task_id, slide_id)
            if raw_path is None or not feat_path.is_file():
                continue
            patches, coord_min, coord_max = feature_geometry(feat_path)
            if patches < args.min_patches or patches > args.cesc_candidate_max_patches:
                continue
            thumbnail, qc = specimen_qc(raw_path, args.qc_thumbnail_size)
            qc["feature_coord_min"] = coord_min.tolist()
            qc["feature_coord_max"] = coord_max.tolist()
            qc["coordinates_fit_raw_wsi"] = coordinates_fit_wsi(
                coord_max, qc, args.vis_patch_size
            )
            if not qc["coordinates_fit_raw_wsi"]:
                continue
            strict_single = (
                qc["largest_tissue_fraction"] >= args.cesc_min_largest_fraction
                and qc["second_tissue_fraction"] <= args.cesc_max_second_fraction
            )
            qc_path = qc_dir / f"{split_name}_{slide_id}.jpg"
            thumbnail.save(qc_path, quality=95)
            candidates.append({
                "task_id": task_id,
                "task_name": "CESC",
                "slide_id": slide_id,
                "local_label": int(local_label),
                "true_global_class": TASK_TO_GLOBAL_CLASS[task_id][int(local_label)],
                "split": split_name,
                "fold": args.fold,
                "raw_wsi": str(raw_path),
                "raw_match": raw_match,
                "feature_h5": str(feat_path),
                "num_patches": patches,
                "qc_thumbnail": str(qc_path),
                "qc": qc,
                "strict_single_specimen": strict_single,
                "split_priority": split_rank,
                "selection_prediction": {
                    "selection_requires_correct_prediction": False,
                    "selection_method": "head4_online_causal_consistency",
                },
                "selection_rejections": {},
                "specimen_crop_margin": args.specimen_crop_margin,
            })
    if not candidates:
        raise RuntimeError("No CESC candidate has aligned SVS/features for Head #4 ranking")
    candidates.sort(key=lambda item: (
        not item["strict_single_specimen"],
        item["split_priority"],
        -item["qc"]["largest_tissue_fraction"],
        item["qc"]["second_tissue_fraction"],
        abs(item["num_patches"] - args.preferred_patches),
    ))
    candidates = candidates[: int(args.cesc_max_candidates)]
    with (output_dir / "CESC_candidate_pool.json").open("w") as handle:
        json.dump(candidates, handle, indent=2, allow_nan=False)
    print(
        f"[CESC-POOL] candidates={len(candidates)} "
        f"strict={sum(item['strict_single_specimen'] for item in candidates)}",
        flush=True,
    )
    return candidates


def _save_pre_cesc_ln_snapshot(model, path, stage, checkpoint):
    backbone_state = model.backbone.state_dict()
    teacher_state = model.teacher.state_dict() if model.teacher is not None else {}
    names = list(model.adapt_names)
    payload = {
        "stage": stage,
        "source_checkpoint": str(checkpoint),
        "param_scope": model.param_scope,
        "adapt_names": names,
        "backbone_adapted": {
            name: backbone_state[name].detach().cpu() for name in names
        },
        "teacher_adapted": {
            name: teacher_state[name].detach().cpu()
            for name in names if name in teacher_state
        },
        "optimizer": deepcopy(model.optimizer.state_dict()),
        "n_adapted": int(model.n_adapted),
        "n_skipped": int(model.n_skipped),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def rank_cesc_head4_candidates(
    selected_prefix, candidates, args, device, output_dir
):
    """Rank CESC candidates from identical pre-CESC online-causal states."""
    observations = {item["slide_id"]: [] for item in candidates}
    snapshots_dir = output_dir / "pre_CESC_snapshots"
    for stage in tqdm(range(1, 7), desc="CESC Head #4 ranking", unit="stage"):
        checkpoint = stage_checkpoint(args, stage)
        model, _, _ = make_tta_model(args, device, checkpoint)
        for item in selected_prefix:
            features, coords = load_features(Path(item["feature_h5"]), device)
            naive_prediction(model, features, coords, args, reset_before_slide=False)
            del features, coords
            torch.cuda.empty_cache()
        _save_pre_cesc_ln_snapshot(
            model, snapshots_dir / f"stage_{stage}_pre_CESC_ln_state.pt",
            stage, checkpoint,
        )
        backbone_state = deepcopy(model.backbone.state_dict())
        teacher_state = (
            deepcopy(model.teacher.state_dict()) if model.teacher is not None else None
        )
        optimizer_state = deepcopy(model.optimizer.state_dict())
        counters = (model.n_adapted, model.n_skipped)
        for candidate in tqdm(
            candidates, desc=f"Stage {stage}/6 CESC candidates", unit="WSI", leave=False
        ):
            model.backbone.load_state_dict(backbone_state, strict=True)
            if model.teacher is not None:
                model.teacher.load_state_dict(teacher_state, strict=True)
            model.optimizer.load_state_dict(optimizer_state)
            model.n_adapted, model.n_skipped = counters
            features, coords = load_features(Path(candidate["feature_h5"]), device)
            prediction = naive_prediction(
                model, features, coords, args, reset_before_slide=False
            )
            scores = extract_attention_scores(
                model, features, coords, coords.cpu().numpy(), args
            )["titan_pool_attention_median"]
            observations[candidate["slide_id"]].append({
                "stage": stage,
                "scores": scores,
                "coords": coords.cpu().numpy(),
                "predicted_global_class": prediction["predicted_global_class"],
            })
            del features, coords
            torch.cuda.empty_cache()
        del model, backbone_state, teacher_state, optimizer_state
        gc.collect(); torch.cuda.empty_cache()
    ranked = []
    for candidate in candidates:
        metrics = score_candidate_attention(
            observations[candidate["slide_id"]], args
        )
        # Prefer stable and spatially concentrated Head #4 maps; morphology is
        # a gate/tie-breaker, not a substitute for attention consistency.
        metrics["head4_noise_aware_score"] = (
            0.35 * ((metrics["mean_stage_rank_correlation"] + 1.0) / 2.0)
            + 0.25 * metrics["mean_top_region_jaccard"]
            + 0.20 * metrics["mean_focus"]
            + 0.20 * metrics["mean_spatial_compactness"]
        )
        row = deepcopy(candidate)
        row["attention_selection"] = metrics
        ranked.append(row)
    ranked.sort(key=lambda item: (
        not item["strict_single_specimen"],
        -item["attention_selection"]["head4_noise_aware_score"],
    ))
    with (output_dir / "CESC_head4_candidate_ranking.json").open("w") as handle:
        json.dump(ranked, handle, indent=2, allow_nan=False)
    csv_rows = []
    for rank, item in enumerate(ranked, 1):
        metrics = item["attention_selection"]
        csv_rows.append({
            "rank": rank,
            "selected": rank == 1,
            "slide_id": item["slide_id"],
            "split": item["split"],
            "num_patches": item["num_patches"],
            "strict_single_specimen": item["strict_single_specimen"],
            "largest_tissue_fraction": item["qc"]["largest_tissue_fraction"],
            "second_tissue_fraction": item["qc"]["second_tissue_fraction"],
            "head4_noise_aware_score": metrics["head4_noise_aware_score"],
            "rank_correlation": metrics["mean_stage_rank_correlation"],
            "top_region_jaccard": metrics["mean_top_region_jaccard"],
            "focus": metrics["mean_focus"],
            "spatial_compactness": metrics["mean_spatial_compactness"],
        })
    with (output_dir / "CESC_head4_candidate_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    chosen = ranked[0]
    print(
        f"[CESC-SELECTED] slide={chosen['slide_id']} split={chosen['split']} "
        f"score={chosen['attention_selection']['head4_noise_aware_score']:.4f}",
        flush=True,
    )
    return chosen


def reference_percentiles(values, reference):
    reference = np.sort(np.asarray(reference, dtype=np.float32))
    values = np.asarray(values, dtype=np.float32)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    average_rank = (left + right - 1) / 2.0
    return (100.0 * average_rank / max(1, len(reference) - 1)).astype(np.float32)


def stage_checkpoint(args, stage: int):
    """Map one-based continual stage to the checkpoint used for visualization."""
    fold_dir = f"fold_{args.fold}"
    if stage == 1:
        return resolve(args.finetuned_dir) / fold_dir / "task_0.pt"
    if stage == 6:
        # Match the checkpoint loaded by test_classIL_tta.py naive inference.
        return resolve(args.merged_dir) / fold_dir / "merged_final.pth"
    return resolve(args.merged_dir) / fold_dir / f"merged_task_{stage - 1}.pth"


def fit_image_to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fit without distortion and center-pad to an exact publication canvas."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def specimen_crop_box(qc, target_width: int, target_height: int, margin: float):
    """Map the dominant thumbnail component onto another whole-slide image."""
    sx = target_width / qc["thumbnail_width"]
    sy = target_height / qc["thumbnail_height"]
    x0 = qc["dominant_tissue_x"] * sx
    y0 = qc["dominant_tissue_y"] * sy
    x1 = (qc["dominant_tissue_x"] + qc["dominant_tissue_width"]) * sx
    y1 = (qc["dominant_tissue_y"] + qc["dominant_tissue_height"]) * sy
    pad = margin * max(x1 - x0, y1 - y0)
    return (
        max(0, int(round(x0 - pad))),
        max(0, int(round(y0 - pad))),
        min(target_width, int(round(x1 + pad))),
        min(target_height, int(round(y1 + pad))),
    )


def save_original(row, output_dir, max_size, panel_width, panel_height):
    import openslide

    slide = openslide.OpenSlide(row["raw_wsi"])
    try:
        whole_thumbnail = slide.get_thumbnail((max_size, max_size)).convert("RGB")
        crop_box = specimen_crop_box(
            row["qc"], whole_thumbnail.width, whole_thumbnail.height,
            row["specimen_crop_margin"],
        )
        image = whole_thumbnail.crop(crop_box)
    finally:
        slide.close()
    path = output_dir / row["task_name"] / "original_wsi.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = fit_image_to_canvas(image, panel_width, panel_height)
    image.save(path, quality=96)
    return path


def compose_grid_horizontal(selected, original_paths, heatmap_paths, output_path, args):
    """Datasets as rows and continual stages as columns with a horizontal arrow."""
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    rows, cols = len(selected), 7
    # Wide panels and compact rows for the transposed publication layout.
    fig, axes = plt.subplots(rows, cols, figsize=(3.75 * cols, 2.65 * rows))
    stage_colors = {"current": "#f4a261", "past": "#90be6d", "unseen": "#4d96d7"}
    for row, item in enumerate(selected):
        axes[row, 0].imshow(Image.open(original_paths[row]), aspect="auto")
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_linewidth(2); spine.set_edgecolor("black")
        for stage in range(1, 7):
            ax = axes[row, stage]
            ax.imshow(Image.open(heatmap_paths[(stage, row)]), aspect="auto")
            state = (
                "current" if row == stage - 1
                else ("past" if row < stage - 1 else "unseen")
            )
            for spine in ax.spines.values():
                spine.set_linewidth(5); spine.set_edgecolor(stage_colors[state])
            ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(
        left=0.115, right=0.895, top=0.965, bottom=0.115,
        wspace=0.065, hspace=0.022,
    )
    fig.canvas.draw()
    for row, item in enumerate(selected):
        position = axes[row, 0].get_position()
        fig.text(
            position.x0 - 0.014, (position.y0 + position.y1) / 2,
            rf"$\bf{{Dataset\ {row + 1}}}$" + f"\n{item['task_name']}",
            ha="right", va="center", fontsize=21, fontweight="normal",
        )
    # Vertical dashed separators occupy the whitespace between stage columns.
    for stage in range(1, 6):
        left_panel = axes[0, stage].get_position()
        right_panel = axes[0, stage + 1].get_position()
        x = (left_panel.x1 + right_panel.x0) / 2.0
        fig.add_artist(Line2D(
            [x, x],
            [axes[-1, stage].get_position().y0, axes[0, stage].get_position().y1],
            transform=fig.transFigure, color="black", linewidth=2.2,
            linestyle=(0, (5, 4)),
        ))
    stage_left = axes[-1, 1].get_position().x0
    stage_right = axes[-1, 6].get_position().x1
    arrow_y = axes[-1, 1].get_position().y0 - 0.032
    fig.add_artist(FancyArrowPatch(
        (stage_left, arrow_y), (stage_right, arrow_y),
        transform=fig.transFigure,
        arrowstyle="Simple,head_length=28,head_width=28,tail_width=6",
        color="#4d96d7", linewidth=0,
    ))
    fig.text(
        (stage_left + stage_right) / 2, arrow_y - 0.018,
        "Continual Learning Stage", va="top", ha="center",
        fontsize=21, fontweight="bold",
    )
    color_stops = getattr(args, "attention_color_stops", None)
    if color_stops:
        attention_cmap = colors.LinearSegmentedColormap.from_list(
            "titan_attention_focus",
            [(float(stop[0]), str(stop[1])) for stop in color_stops], N=256,
        )
    else:
        attention_cmap = "jet"
    matrix_top = axes[0, 1].get_position().y1
    matrix_bottom = axes[-1, 1].get_position().y0
    cax = fig.add_axes([0.92, matrix_bottom, 0.022, matrix_top - matrix_bottom])
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=attention_cmap), cax=cax
    )
    colorbar.set_label("Attention Score", fontsize=21, fontweight="bold", labelpad=24)
    colorbar.ax.tick_params(labelsize=21, width=3, length=12)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    pdf_path = Path(output_path).with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"[GRID] PDF saved: {pdf_path}", flush=True)
    plt.close(fig)


def compose_grid(selected, original_paths, heatmap_paths, output_path, args):
    if getattr(args, "grid_orientation", "stages_vertical") == "stages_horizontal":
        return compose_grid_horizontal(
            selected, original_paths, heatmap_paths, output_path, args
        )
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    # Publication layout: datasets are columns; original WSI is row 1 and
    # continual stages 1..6 are rows 2..7.
    rows, cols = 7, len(selected)
    fig, axes = plt.subplots(rows, cols, figsize=(3.45 * cols, 3.12 * rows))
    for col, item in enumerate(selected):
        axes[0, col].imshow(Image.open(original_paths[col]))
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        for spine in axes[0, col].spines.values():
            spine.set_linewidth(2); spine.set_edgecolor("black")
    stage_colors = {"current": "#f4a261", "past": "#90be6d", "unseen": "#4d96d7"}
    for col, item in enumerate(selected):
        for stage in range(1, 7):
            ax = axes[stage, col]
            ax.imshow(Image.open(heatmap_paths[(stage, col)]))
            state = "current" if col == stage - 1 else ("past" if col < stage - 1 else "unseen")
            for spine in ax.spines.values():
                spine.set_linewidth(5); spine.set_edgecolor(stage_colors[state])
            ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(
        left=0.105, right=0.895, top=0.925, bottom=0.035,
        wspace=0.035, hspace=0.075,
    )
    fig.canvas.draw()
    for col, item in enumerate(selected):
        position = axes[0, col].get_position()
        fig.text(
            (position.x0 + position.x1) / 2, position.y1 + 0.018,
            rf"$\bf{{Dataset\ {col + 1}}}$" + f"\n{item['task_name']}",
            ha="center", va="bottom", fontsize=21, fontweight="normal",
        )
    # Horizontal separators sit in the deliberately enlarged whitespace
    # between successive stage rows, never on top of panel borders.
    for stage in range(1, 6):
        upper = axes[stage, 0].get_position()
        lower = axes[stage + 1, 0].get_position()
        y = (upper.y0 + lower.y1) / 2.0
        fig.add_artist(Line2D(
            [axes[stage, 0].get_position().x0, axes[stage, -1].get_position().x1],
            [y, y],
            transform=fig.transFigure, color="black",
            linewidth=2.2, linestyle=(0, (5, 4)),
        ))
    stage_top = axes[1, 0].get_position().y1
    stage_bottom = axes[6, 0].get_position().y0
    matrix_left = axes[1, 0].get_position().x0
    arrow_x = matrix_left - 0.025
    arrow = FancyArrowPatch(
        (arrow_x, stage_top), (arrow_x, stage_bottom), transform=fig.transFigure,
        arrowstyle="Simple,head_length=28,head_width=28,tail_width=6",
        color="#4d96d7", linewidth=0,
    )
    fig.add_artist(arrow)
    fig.text(
        arrow_x - 0.035, (stage_top + stage_bottom) / 2,
        "Continual Learning Stage", va="center", ha="center",
        fontsize=48, fontweight="bold", rotation=90,
    )
    color_stops = getattr(args, "attention_color_stops", None)
    if color_stops:
        attention_cmap = colors.LinearSegmentedColormap.from_list(
            "titan_attention_focus",
            [(float(stop[0]), str(stop[1])) for stop in color_stops],
            N=256,
        )
    else:
        attention_cmap = "jet"
    cax = fig.add_axes([0.92, stage_bottom, 0.022, stage_top - stage_bottom])
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=attention_cmap),
        cax=cax,
    )
    colorbar.set_label("Attention Score", fontsize=48, fontweight="bold", labelpad=24)
    colorbar.ax.tick_params(labelsize=46, width=3, length=12)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate_layout(seq_dataset, raw_index, args):
    exact, barcode = raw_index
    for stage in range(1, 7):
        checkpoint = stage_checkpoint(args, stage)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"[VALID] continual_stage={stage} checkpoint={checkpoint}")
    for task_id, task_name in enumerate(TASK_NAMES):
        split_path = Path(seq_dataset._split_csv(task_id, args.fold))
        if split_path.name != "splits_4.csv":
            raise RuntimeError(
                f"Expected zero-based fold 4 split for {task_name}, got {split_path}"
            )
        task_split = selected_split(
            seq_dataset.datasets[task_id], str(split_path), args.selection_split
        )
        available = 0
        for slide_id, _ in slide_rows(task_split):
            raw_path, _ = resolve_raw_slide(slide_id, exact, barcode)
            if raw_path and feature_path(seq_dataset, task_id, slide_id).is_file():
                available += 1
        print(
            f"[VALID] task={task_name} selection_split={args.selection_split} "
            f"fold_index={args.fold} "
            f"split={split_path.name} candidates={available}"
        )
        if available == 0:
            raise RuntimeError(f"No raw+feature train candidates for {task_name}")
    overrides = dict(getattr(args, "slide_overrides", {}) or {})
    if overrides:
        if set(overrides) != set(TASK_NAMES):
            raise ValueError(
                f"slide_overrides tasks must equal {TASK_NAMES}; got={list(overrides)}"
            )
        for task_id, task_name in enumerate(TASK_NAMES):
            requested = str(overrides[task_name]).removesuffix(".svs")
            split_path = seq_dataset._split_csv(task_id, args.fold)
            split_hits = []
            for split_name, split in zip(
                ("train", "val", "test"),
                seq_dataset.datasets[task_id].return_splits(
                    from_id=False, csv_path=split_path
                ),
            ):
                if split is not None and any(
                    slide_id == requested for slide_id, _ in slide_rows(split)
                ):
                    split_hits.append(split_name)
            raw_path, raw_match = resolve_raw_slide(requested, exact, barcode)
            feat_path = feature_path(seq_dataset, task_id, requested)
            print(
                f"[VALID-OVERRIDE] task={task_name} slide={requested} "
                f"split={split_hits} raw_match={raw_match} feature={feat_path.is_file()}",
                flush=True,
            )
            if len(split_hits) != 1 or raw_path is None or not feat_path.is_file():
                raise RuntimeError(
                    f"Invalid slide override for {task_name}: split={split_hits} "
                    f"raw={raw_path} feature={feat_path}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/attention_maps/ood_fold4_naive_6task.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--render-only", action="store_true",
        help="Recompose the grid from existing JPG panels without TITAN/TTA.",
    )
    parser.add_argument("--head-index", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--figure-filename")
    parser.add_argument("--run-scope", choices=("full", "cesc_only"))
    parser.add_argument("--cesc-slide-id")
    parser.add_argument("--cesc-left-specimen", action="store_true")
    parser.add_argument("--no-pre-wsi-attention", action="store_true")
    cli = parser.parse_args()
    config_path = resolve(cli.config)
    args = load_args(config_path, cli.validate_only)
    if cli.head_index is not None:
        args.selected_head_index = cli.head_index
    if cli.output_dir:
        args.output_dir = cli.output_dir
    if cli.figure_filename:
        args.figure_filename = cli.figure_filename
    if cli.run_scope:
        args.run_scope = cli.run_scope
    if cli.cesc_slide_id:
        args.slide_overrides = dict(args.slide_overrides)
        args.slide_overrides["CESC"] = cli.cesc_slide_id
        args.rank_cesc_candidates = False
        args.reuse_cesc_ranking = False
    if cli.no_pre_wsi_attention:
        args.save_pre_wsi_attention = False
    if args.fold != 4:
        raise ValueError("This experiment is required to use train fold 4")
    if args.mode != "naive" or args.naive_inference_model != "student":
        raise ValueError("Attention grid requires naive adapted-student inference")
    if args.k_patches != K_PATCHES:
        raise ValueError(f"Configured K={args.k_patches}, trained K={K_PATCHES}")
    if getattr(args, "attention_backend", "pooling_coverage") == "selected_transformer_head":
        if not 0 <= int(args.selected_layer_index) < 6:
            raise ValueError("selected_layer_index must be in [0,5]")
        if not 0 <= int(args.selected_head_index) < 12:
            raise ValueError("selected_head_index must be in [0,11]")
    if bool(getattr(args, "online_causal", False)) and (
        bool(getattr(args, "episodic", True))
        or bool(getattr(args, "reset_before_slide", True))
    ):
        raise ValueError(
            "online_causal requires tta.episodic=false and "
            "runtime.reset_before_slide=false"
        )
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cli.render_only:
        manifest = output_dir / "selected_slides.json"
        with manifest.open() as handle:
            selected = json.load(handle)
        by_task = {item["task_name"]: item for item in selected}
        selected = [by_task[name] for name in TASK_NAMES]
        original_paths = [
            output_dir / item["task_name"] / "original_wsi.jpg"
            for item in selected
        ]
        heatmap_paths = {
            (stage, col): output_dir / item["task_name"] / f"stage_{stage}" /
            "titan_pool_attention.jpg"
            for col, item in enumerate(selected)
            for stage in range(1, 7)
        }
        missing = [
            str(path) for path in original_paths + list(heatmap_paths.values())
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Render-only input panels are missing:\n  " + "\n  ".join(missing)
            )
        figure_path = output_dir / getattr(
            args, "figure_filename",
            "region_level_attention_maps_6tasks_fold4_naive.jpg",
        )
        compose_grid(selected, original_paths, heatmap_paths, figure_path, args)
        print(f"[DONE] render-only attention grid: {figure_path}", flush=True)
        return
    dataset_cfg = OmegaConf.load(resolve(args.dataset_config))
    seq_dataset = Sequential_Generic_MIL_Dataset(dataset_cfg)
    raw_index = index_raw_slides(args.raw_roots)
    validate_layout(seq_dataset, raw_index, args)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("TITAN attention extraction requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fixed_manifest_value = getattr(args, "fixed_selection_manifest", "")
    fixed_manifest = resolve(fixed_manifest_value) if fixed_manifest_value else None
    slide_overrides = getattr(args, "slide_overrides", None)
    if slide_overrides:
        selected = build_override_selection(
            seq_dataset, raw_index, args, output_dir
        )
    elif fixed_manifest is not None and fixed_manifest.is_file():
        selected = load_fixed_selection(fixed_manifest, output_dir)
    else:
        candidate_pools = build_candidate_pools(
            seq_dataset, raw_index, args, output_dir
        )
        # Correctness is intentionally not a selection criterion. Candidate
        # pools are ordered by distance to the preferred patch count.
        selected = [pool[0] for pool in candidate_pools]
    if len(selected) != 6 or len({item["slide_id"] for item in selected}) != 6:
        raise RuntimeError(
            "Expected exactly one unique WSI for each of the six tasks"
        )
    for item in selected:
        item.setdefault("specimen_crop_margin", args.specimen_crop_margin)
    if cli.cesc_left_specimen:
        cesc = next(item for item in selected if item["task_name"] == "CESC")
        boxes = cesc["qc"].get("tissue_component_boxes", [])
        if len(boxes) < 2:
            raise RuntimeError(
                "--cesc-left-specimen requires at least two detected components"
            )
        left = min(boxes, key=lambda box: box["x"])
        cesc["qc"].update({
            "dominant_tissue_x": left["x"],
            "dominant_tissue_y": left["y"],
            "dominant_tissue_width": left["width"],
            "dominant_tissue_height": left["height"],
            "dominant_tissue_aspect_ratio": (
                left["width"] / max(1, left["height"])
            ),
        })
        cesc["visualized_specimen"] = "left_component"
        print(
            f"[CESC-CROP] slide={cesc['slide_id']} left_box={left}", flush=True
        )
    if bool(getattr(args, "rank_cesc_candidates", False)):
        ranking_path = output_dir / "CESC_head4_candidate_ranking.json"
        if bool(getattr(args, "reuse_cesc_ranking", True)) and ranking_path.is_file():
            with ranking_path.open() as handle:
                ranked_cesc = json.load(handle)
            if not ranked_cesc:
                raise RuntimeError(f"Empty cached CESC ranking: {ranking_path}")
            ranking_choice = int(getattr(args, "cesc_ranking_choice", 1))
            if not 1 <= ranking_choice <= len(ranked_cesc):
                raise ValueError(
                    f"cesc_ranking_choice must be in [1,{len(ranked_cesc)}], "
                    f"got {ranking_choice}"
                )
            selected[-1] = ranked_cesc[ranking_choice - 1]
            print(
                f"[CESC-RESUME] ranking_choice={ranking_choice} selected slide="
                f"{selected[-1]['slide_id']} from {ranking_path}",
                flush=True,
            )
        else:
            candidates = build_cesc_candidate_pool(
                seq_dataset, raw_index, args, output_dir
            )
            selected[-1] = rank_cesc_head4_candidates(
                selected[:5], candidates, args, device, output_dir
            )
    run_scope = str(getattr(args, "run_scope", "full"))
    if run_scope not in {"full", "cesc_only"}:
        raise ValueError("run_scope must be 'full' or 'cesc_only'")
    if run_scope == "cesc_only":
        missing_cache = [
            str(output_dir / item["task_name"] / f"stage_{stage}" /
                "titan_pool_attention.jpg")
            for item in selected[:5]
            for stage in range(1, 7)
            if not (output_dir / item["task_name"] / f"stage_{stage}" /
                    "titan_pool_attention.jpg").is_file()
        ]
        missing_cache += [
            str(output_dir / item["task_name"] / "original_wsi.jpg")
            for item in selected[:5]
            if not (output_dir / item["task_name"] / "original_wsi.jpg").is_file()
        ]
        if missing_cache:
            raise FileNotFoundError(
                "CESC-only mode requires cached Head #4 panels for BRCA-TGCT:\n  "
                + "\n  ".join(missing_cache)
            )
        invalid_cache = []
        for item in selected[:5]:
            for stage in range(1, 7):
                metadata_path = (
                    output_dir / item["task_name"] / f"stage_{stage}" /
                    "metadata.json"
                )
                if not metadata_path.is_file():
                    invalid_cache.append(f"missing metadata: {metadata_path}")
                    continue
                with metadata_path.open() as handle:
                    cached = json.load(handle)
                if (
                    cached.get("attention_backend") != "selected_transformer_head"
                    or int(cached.get("transformer_layer_index", -1)) != 5
                    or int(cached.get("tensor_head_index", -1)) != 4
                ):
                    invalid_cache.append(f"not Head #4 layer 6: {metadata_path}")
        if invalid_cache:
            raise RuntimeError(
                "CESC-only cache validation failed:\n  "
                + "\n  ".join(invalid_cache)
            )
        print("[RUN-SCOPE] cesc_only: reuse 30 cached BRCA-TGCT panels", flush=True)
    write_manifest(selected, output_dir)

    if run_scope == "cesc_only":
        original_paths = [
            output_dir / row["task_name"] / "original_wsi.jpg"
            for row in selected[:5]
        ]
        original_paths.append(save_original(
            selected[-1], output_dir, args.original_thumbnail_size,
            args.panel_width, args.panel_height,
        ))
    elif fixed_manifest is not None and fixed_manifest.is_file() and not slide_overrides:
        original_paths = [
            output_dir / row["task_name"] / "original_wsi.jpg"
            for row in selected
        ]
    else:
        original_paths = [save_original(
            row, output_dir, args.original_thumbnail_size,
            args.panel_width, args.panel_height,
        ) for row in selected]
    stage_scores = {}
    stage_pre_scores = {}
    stage_metadata = {}
    for stage in tqdm(
        range(1, 7),
        desc="Continual stages",
        unit="stage",
        mininterval=1.0,
        dynamic_ncols=False,
        file=sys.stdout,
    ):
        checkpoint = stage_checkpoint(args, stage)
        model, _, task_paths = make_tta_model(args, device, checkpoint)
        if bool(getattr(args, "online_causal", False)) and model.episodic:
            raise RuntimeError("online_causal requires CAST-Slide episodic=False")
        for col, item in enumerate(tqdm(
            selected,
            desc=f"Stage {stage}/6 WSI stream",
            unit="WSI",
            leave=False,
            mininterval=1.0,
            dynamic_ncols=False,
            file=sys.stdout,
        )):
            tqdm.write(
                f"[FULL] stage={stage}/6 task={item['task_name']} "
                f"slide={item['slide_id']}",
                file=sys.stdout,
            )
            features, coords = load_features(Path(item["feature_h5"]), device)
            coords_np = coords.cpu().numpy()
            capture_current = run_scope == "full" or item["task_name"] == "CESC"
            pre_scores = None
            if capture_current and bool(getattr(args, "save_pre_wsi_attention", False)):
                tqdm.write(
                    f"[PRE-WSI] stage={stage}/6 stream={col + 1}/6 "
                    f"task={item['task_name']}",
                    file=sys.stdout,
                )
                pre_scores = extract_attention_scores(
                    model, features, coords, coords_np, args
                )
            prediction = naive_prediction(
                model, features, coords, args,
                reset_before_slide=bool(getattr(args, "reset_before_slide", True)),
            )
            if capture_current:
                scores = extract_attention_scores(
                    model, features, coords, coords_np, args
                )
                stage_scores[(stage, col)] = (coords_np, scores)
                if pre_scores is not None:
                    stage_pre_scores[(stage, col)] = (coords_np, pre_scores)
                    prediction["pre_post_attention_change"] = attention_change_metrics(
                        pre_scores["titan_pool_attention_median"],
                        scores["titan_pool_attention_median"],
                        args.attention_top_fraction,
                    )
                stage_metadata[(stage, col)] = {
                "stage": stage,
                "stream_index": col,
                "stream_position": col + 1,
                "stream_length": len(selected),
                "checkpoint": str(checkpoint),
                "task_checkpoints": task_paths,
                "prediction": prediction,
                "true_global_class": item["true_global_class"],
                "correct": prediction["predicted_global_class"] == item["true_global_class"],
                "attention": (
                    "TITAN final-layer selected-head CLS-to-patch attention"
                    if getattr(args, "attention_backend", "pooling_coverage")
                    == "selected_transformer_head"
                    else "TITAN contrastive pooling query-to-patch attention"
                ),
                "attention_backend": getattr(
                    args, "attention_backend", "pooling_coverage"
                ),
                "transformer_layer_index": getattr(
                    args, "selected_layer_index", None
                ),
                "tensor_head_index": getattr(args, "selected_head_index", None),
                "attention_class_specific": False,
                "inference": "naive adapted student",
                "adaptation_protocol": (
                    "online_causal_selected_six_wsi_stream"
                    if bool(getattr(args, "online_causal", False))
                    else "episodic_per_slide"
                ),
                "reset_before_slide": bool(
                    getattr(args, "reset_before_slide", True)
                ),
                "attention_timing": "immediately_after_current_wsi_adaptation",
                }
            del features, coords
            torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()

    full_wsi_consistency = []
    consistency_path = output_dir / "full_wsi_attention_consistency.json"
    if run_scope == "cesc_only" and consistency_path.is_file():
        with consistency_path.open() as handle:
            full_wsi_consistency = [
                row for row in json.load(handle) if row.get("task") != "CESC"
            ]
    scored_columns = range(6) if run_scope == "full" else [5]
    for col in scored_columns:
        item = selected[col]
        entries = [
            {
                "scores": stage_scores[(stage, col)][1]["titan_pool_attention_median"],
                "coords": stage_scores[(stage, col)][0],
            }
            for stage in range(1, 7)
        ]
        metrics = score_candidate_attention(entries, args)
        item["full_wsi_attention_consistency"] = metrics
        full_wsi_consistency.append({
            "task": item["task_name"],
            "slide_id": item["slide_id"],
            **{key: value for key, value in metrics.items() if key != "per_stage"},
            "per_stage": metrics["per_stage"],
        })
        print(
            f"[FULL-WSI] {item['task_name']} focus={metrics['mean_focus']:.4f} "
            f"compactness={metrics['mean_spatial_compactness']:.4f} "
            f"rank_corr={metrics['mean_stage_rank_correlation']:.4f} "
            f"top_jaccard={metrics['mean_top_region_jaccard']:.4f}"
        )
    with consistency_path.open("w") as handle:
        json.dump(full_wsi_consistency, handle, indent=2, allow_nan=False)
    write_manifest(selected, output_dir)

    heatmap_paths = {
        (stage, col): output_dir / item["task_name"] / f"stage_{stage}" /
        "titan_pool_attention.jpg"
        for col, item in enumerate(selected[:5])
        for stage in range(1, 7)
    } if run_scope == "cesc_only" else {}
    rendered_columns = range(6) if run_scope == "full" else [5]
    for col in rendered_columns:
        item = selected[col]
        reference = np.concatenate([
            stage_scores[(stage, col)][1]["titan_pool_attention_median"]
            for stage in range(1, 7)
        ])
        for stage in range(1, 7):
            coords_np, scores = stage_scores[(stage, col)]
            scores["titan_pool_attention_percentile"] = reference_percentiles(
                scores["titan_pool_attention_median"], reference
            )
            stage_dir = output_dir / item["task_name"] / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            save_h5(stage_dir / "scores.h5", coords_np,
                    stage_metadata[(stage, col)]["prediction"]["adaptation_indices"], scores)
            save_clam_blockmaps(stage_dir, item["slide_id"], coords_np, scores)
            with (stage_dir / "metadata.json").open("w") as handle:
                json.dump(stage_metadata[(stage, col)], handle, indent=2, allow_nan=False,
                          default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)
            render_maps(Path(item["raw_wsi"]), coords_np, scores, stage_dir, args)
            if (stage, col) in stage_pre_scores:
                pre_coords, pre_scores = stage_pre_scores[(stage, col)]
                pre_scores["titan_pool_attention_percentile"] = reference_percentiles(
                    pre_scores["titan_pool_attention_median"], reference
                )
                save_h5(
                    stage_dir / "pre_wsi_scores.h5", pre_coords,
                    stage_metadata[(stage, col)]["prediction"]["adaptation_indices"],
                    pre_scores,
                )
                pre_dir = stage_dir / "pre_wsi"
                pre_dir.mkdir(parents=True, exist_ok=True)
                render_maps(
                    Path(item["raw_wsi"]), pre_coords, pre_scores, pre_dir, args
                )
                pre_heatmap_path = pre_dir / "titan_pool_attention.jpg"
                pre_heatmap = Image.open(pre_heatmap_path).convert("RGB")
                pre_crop_box = specimen_crop_box(
                    item["qc"], pre_heatmap.width, pre_heatmap.height,
                    args.specimen_crop_margin,
                )
                fit_image_to_canvas(
                    pre_heatmap.crop(pre_crop_box),
                    args.panel_width,
                    args.panel_height,
                ).save(pre_heatmap_path, quality=args.jpeg_quality)
            heatmap_path = stage_dir / "titan_pool_attention.jpg"
            heatmap_image = Image.open(heatmap_path).convert("RGB")
            crop_box = specimen_crop_box(
                item["qc"], heatmap_image.width, heatmap_image.height,
                args.specimen_crop_margin,
            )
            standardized = fit_image_to_canvas(
                heatmap_image.crop(crop_box), args.panel_width, args.panel_height
            )
            standardized.save(heatmap_path, quality=args.jpeg_quality)
            heatmap_paths[(stage, col)] = heatmap_path
    if len(heatmap_paths) != 36 or not all(path.is_file() for path in heatmap_paths.values()):
        raise RuntimeError("Expected 36 heatmaps: six fixed WSIs across six stages")
    expected_size = (args.panel_width, args.panel_height)
    all_panel_paths = original_paths + list(heatmap_paths.values())
    bad_sizes = [str(path) for path in all_panel_paths
                 if Image.open(path).size != expected_size]
    if bad_sizes:
        raise RuntimeError(
            f"Publication panels must all be {expected_size}; invalid={bad_sizes}"
        )
    figure_path = output_dir / getattr(
        args, "figure_filename",
        "region_level_attention_maps_6tasks_fold4_naive.jpg",
    )
    compose_grid(selected, original_paths, heatmap_paths, figure_path, args)
    print(f"[DONE] attention grid: {figure_path}")


if __name__ == "__main__":
    main()
