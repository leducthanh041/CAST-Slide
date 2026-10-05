"""Whole-slide CAST-Slide native TITAN attention maps.

TTA is performed once on a spatially representative K-patch bag.  The adapted
inference model is then frozen while overlapping K-patch context bags cover
every preprocessed patch. Native TITAN pooling, last-layer, and rollout
attention are aggregated across contexts before CLAM overlays them on the WSI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoModel

from cast_slide.constants import K_PATCHES, NUM_CLASSES, TASK_NAMES, TITAN_PS_ARG
from cast_slide.prompts_zeroshot import (
    brca_prompts, cesc_prompts, esca_prompts, nsclc_prompts,
    rcc_prompts, tgct_prompts,
)
from cast_slide.task_prompt_io import load_task_prompts_for_tasks
from cast_slide.titan_attention import TitanAttentionCapture
from cast_slide.tta_adapter import CASTSlide, load_task_weights


PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_FUNCTIONS = {
    "BRCA": brca_prompts,
    "RCC": rcc_prompts,
    "NSCLC": nsclc_prompts,
    "ESCA": esca_prompts,
    "TGCT": tgct_prompts,
    "CESC": cesc_prompts,
}
def resolve(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    value = Path(os.path.expandvars(str(path))).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def spatial_order(coords: np.ndarray, seed: int, bins: int) -> np.ndarray:
    """Shuffle within spatial cells, then interleave cells across the WSI."""
    rng = np.random.default_rng(seed)
    lo = coords.min(axis=0).astype(np.float64)
    span = np.maximum(coords.max(axis=0) - lo, 1).astype(np.float64)
    cell_xy = np.minimum(((coords - lo) / span * bins).astype(int), bins - 1)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(cell_xy):
        cells[(int(cell[0]), int(cell[1]))].append(index)
    queues = []
    for key in sorted(cells):
        values = np.asarray(cells[key], dtype=np.int64)
        rng.shuffle(values)
        queues.append(values.tolist())
    rng.shuffle(queues)
    order = []
    while queues:
        next_queues = []
        for queue in queues:
            order.append(queue.pop())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return np.asarray(order, dtype=np.int64)


def representative_indices(coords: np.ndarray, k: int, seed: int, bins: int) -> np.ndarray:
    order = spatial_order(coords, seed, bins)
    selected = order[: min(k, len(order))]
    if len(selected) < k:
        selected = np.concatenate([selected, np.resize(order, k - len(selected))])
    return selected.astype(np.int64, copy=False)


def coverage_bags(
    coords: np.ndarray, k: int, repeats: int, seed: int, bins: int,
) -> list[np.ndarray]:
    bags = []
    n = len(coords)
    for repeat in range(repeats):
        order = spatial_order(coords, seed + 1009 * repeat, bins)
        for start in range(0, n, k):
            bag = order[start:start + k]
            if len(bag) < k:
                needed = k - len(bag)
                filler = np.resize(order[: max(1, min(k, n))], needed)
                bag = np.concatenate([bag, filler])
            bags.append(bag.astype(np.int64, copy=False))
    return bags


def percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2 or np.all(values == values[0]):
        return np.zeros_like(values)
    _, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    cumulative = np.cumsum(counts)
    # Average rank gives tied values the same percentile instead of assigning
    # arbitrary colors according to patch order.
    average_ranks = (cumulative - counts + cumulative - 1) / 2.0
    return (100.0 * average_ranks[inverse] / float(len(values) - 1)).astype(
        np.float32
    )


def build_global_embeddings(titan, device: torch.device) -> torch.Tensor:
    _, templates = brca_prompts()
    prompts = []
    for task_name in TASK_NAMES:
        task_prompts, _ = PROMPT_FUNCTIONS[task_name]()
        prompts.extend(task_prompts)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        result = titan.zero_shot_classifier(prompts, templates, device=str(device))
    return result.to(device)


def make_tta_model(args, device: torch.device, merged_checkpoint=None):
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)
    global_embeddings = build_global_embeddings(titan, device)
    merged = (
        resolve(merged_checkpoint)
        if merged_checkpoint is not None
        else resolve(args.merged_dir) / f"fold_{args.fold}" / "merged_final.pth"
    )
    merged_state = torch.load(merged, map_location="cpu")
    if any(key.startswith("backbone.") for key in merged_state):
        # Stage 1 has no merged_task_0 artifact. Its state is the backbone
        # portion of the task_0 finetuned checkpoint.
        merged_state = {
            key.removeprefix("backbone."): value
            for key, value in merged_state.items()
            if key.startswith("backbone.")
        }
    titan.vision_encoder.load_state_dict(merged_state)
    task_paths = [
        str(resolve(args.finetuned_dir) / f"fold_{args.fold}" / f"task_{task}.pt")
        for task in range(len(NUM_CLASSES))
    ]
    task_weights = load_task_weights(task_paths, device)
    task_prompts = load_task_prompts_for_tasks(
        resolve(args.task_prompts), TASK_NAMES, device
    )
    model = CASTSlide(
        backbone=titan.vision_encoder,
        task_prompts=task_prompts,
        task_weights=task_weights,
        num_classes=NUM_CLASSES,
        device=device,
        mode=args.mode,
        all_class_embeddings=global_embeddings,
        param_scope=args.tta_param_scope,
        M=args.M,
        K_sub=args.K_sub,
        top_ratio=args.top_ratio,
        alpha=args.alpha,
        l2_anchor_beta=args.l2_anchor_beta,
        lr=args.lr,
        n_steps=args.n_steps,
        episodic=bool(getattr(args, "episodic", True)),
        entropy_threshold=args.entropy_threshold,
        use_task_diversity=False,
        use_task_agreement=True,
        gamma=args.gamma,
        select_mode=args.select_mode,
        use_teacher=args.use_teacher,
        tcp_inference_model=args.tcp_inference_model,
        naive_inference_model=args.naive_inference_model,
        ema_alpha=args.ema_alpha,
        adapt_task_prompts=args.adapt_task_prompts,
        ema_alpha_prompt=args.ema_alpha_prompt,
        delta_margin=args.delta_margin,
        tp_anchor_beta=args.tp_anchor_beta,
        gamma_margin=args.gamma_margin,
        tau_task=args.tau_task,
        naive_use_task_entropy=args.naive_use_task_entropy,
        use_dapc=args.use_dapc,
        dapc_loss_weight=args.dapc_loss_weight,
        class_loss_weight=1.0,
        entropy_loss_weight=args.entropy_loss_weight,
        dapc_tau_anchor=args.dapc_tau_anchor,
        dapc_beta=args.dapc_beta,
    )
    return model, merged, task_paths


def score_attention_context_bag(
    model: CASTSlide,
    features: torch.Tensor,
    coords: torch.Tensor,
) -> dict[str, np.ndarray]:
    use_teacher_for_inference = (
        model.mode == "tcp" and model.tcp_inference_model == "teacher"
    ) or (
        model.mode == "naive" and model.naive_inference_model == "teacher"
    )
    inference_model = model.teacher if use_teacher_for_inference else model.backbone
    inference_model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with TitanAttentionCapture(
            inference_model, capture_self_attention=False
        ) as capture:
            inference_model(features, coords, model.ps)
        pool_scores = capture.pool_patch_scores(coords, int(model.ps.item()))
    return {"titan_pool_attention": pool_scores.float().cpu().numpy()}


def aggregate_scores(
    model: CASTSlide,
    features: torch.Tensor,
    coords: torch.Tensor,
    coords_np: np.ndarray,
    args,
) -> dict[str, np.ndarray]:
    bags = coverage_bags(
        coords_np, args.k_patches, args.coverage_repeats, args.seed,
        args.spatial_bins,
    )
    if args.max_bags:
        bags = bags[: args.max_bags]
    method_names = ("titan_pool_attention",)
    collected = {
        name: [[] for _ in range(len(coords_np))] for name in method_names
    }
    for bag_number, indices_np in enumerate(
        tqdm(
            bags,
            desc="TITAN full-patch attention",
            unit="bag",
            leave=False,
            mininterval=1.0,
            dynamic_ncols=False,
            file=sys.stdout,
        ),
        start=1,
    ):
        indices = torch.as_tensor(indices_np, device=features.device)
        bag_scores = score_attention_context_bag(
            model, features[indices], coords[indices]
        )
        # A slide with N < K needs repeated context tokens to preserve the
        # trained K-token input. Average duplicate token attributions so every
        # original patch contributes only once per coverage repeat.
        unique_indices = np.unique(indices_np)
        for global_index in unique_indices:
            local_positions = np.flatnonzero(indices_np == global_index)
            for name in method_names:
                collected[name][int(global_index)].append(
                    float(np.mean(bag_scores[name][local_positions]))
                )
    coverage = np.asarray(
        [len(v) for v in collected[method_names[0]]], dtype=np.int32
    )
    if not args.max_bags and np.any(coverage < args.coverage_repeats):
        raise RuntimeError("Coverage schedule left patches below requested repeats")

    def reduce_nested(nested, fn, missing=np.nan):
        return np.asarray([fn(v) if v else missing for v in nested], dtype=np.float32)

    output = {"coverage_count": coverage}
    for name in method_names:
        median = reduce_nested(collected[name], np.median)
        output[f"{name}_median"] = median
        output[f"{name}_mean"] = reduce_nested(collected[name], np.mean)
        output[f"{name}_std"] = reduce_nested(collected[name], np.std)
        output[f"{name}_percentile"] = percentile(median)
    return output


def import_clam_tools(clam_root: Path):
    sys.path.insert(0, str(clam_root))
    from vis_utils.heatmap_utils import drawHeatmap, initialize_wsi
    return drawHeatmap, initialize_wsi


def draw_attention_boundary(
    heatmap: Image.Image,
    coords: np.ndarray,
    values: np.ndarray,
    slide_size: tuple[int, int],
    args,
    cv2,
) -> Image.Image:
    """Draw clear P-threshold contours without allocating a full WSI mask."""
    width, height = heatmap.size
    mask_scale = min(
        1.0,
        args.boundary_mask_max_size / max(width, height),
    )
    mask_width = max(1, int(round(width * mask_scale)))
    mask_height = max(1, int(round(height * mask_scale)))
    score_canvas = np.zeros((mask_height, mask_width), dtype=np.float32)
    weight_canvas = np.zeros_like(score_canvas)
    sx = mask_width / slide_size[0]
    sy = mask_height / slide_size[1]
    for (x, y), score in zip(coords, values):
        x0 = max(0, int(np.floor(x * sx)))
        y0 = max(0, int(np.floor(y * sy)))
        x1 = min(mask_width, int(np.ceil((x + args.vis_patch_size) * sx)))
        y1 = min(mask_height, int(np.ceil((y + args.vis_patch_size) * sy)))
        if x1 <= x0 or y1 <= y0:
            continue
        score_canvas[y0:y1, x0:x1] += float(score)
        weight_canvas[y0:y1, x0:x1] += 1.0
    valid = weight_canvas > 0
    score_canvas[valid] /= weight_canvas[valid]
    patch_pixels = max(3, int(round(args.vis_patch_size * (sx + sy) / 2.0)))
    blur_kernel = max(5, patch_pixels * 2 + 1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    weighted = cv2.GaussianBlur(score_canvas * valid, (blur_kernel, blur_kernel), 0)
    support = cv2.GaussianBlur(valid.astype(np.float32), (blur_kernel, blur_kernel), 0)
    smooth = np.divide(
        weighted, support, out=np.zeros_like(weighted), where=support > 1e-4
    )
    binary = ((smooth >= args.boundary_threshold) & (support > 0.15)).astype(np.uint8)
    close_size = max(3, patch_pixels * 2 + 1)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = float(patch_pixels ** 2 * args.boundary_min_area_patches)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = contours[:args.boundary_max_contours]

    draw = ImageDraw.Draw(heatmap)
    scale_back = 1.0 / mask_scale
    polylines = []
    for contour in contours:
        points = [
            (int(point[0][0] * scale_back), int(point[0][1] * scale_back))
            for point in contour
        ]
        if len(points) >= 3:
            polylines.append(points + [points[0]])
    for points in polylines:
        draw.line(
            points, fill="black", width=args.boundary_line_width + 6,
            joint="curve",
        )
    for points in polylines:
        draw.line(
            points, fill="white", width=args.boundary_line_width,
            joint="curve",
        )
    return heatmap


def render_maps(
    slide_path: Path, coords: np.ndarray, scores: dict, out_dir: Path,
    args,
):
    draw_heatmap, initialize_wsi = import_clam_tools(resolve(args.clam_root))
    import cv2
    cv2.setNumThreads(max(1, int(os.environ.get("OPENCV_NUM_THREADS", "1"))))
    mask_pkl = out_dir / "segmentation_mask.pkl"
    seg_params = {
        "seg_level": -1,
        "sthresh": args.seg_sthresh,
        "mthresh": args.seg_mthresh,
        "close": args.seg_close,
        "use_otsu": args.seg_use_otsu,
        "keep_ids": [],
        "exclude_ids": [],
    }
    filter_params = {
        "a_t": args.filter_area_tissue,
        "a_h": args.filter_area_hole,
        "max_n_holes": args.filter_max_holes,
    }
    wsi_object = initialize_wsi(
        str(slide_path), seg_mask_path=str(mask_pkl),
        seg_params=seg_params, filter_params=filter_params,
    )
    print(
        f"[RENDER] WSI ready; vis_level={args.heatmap_vis_level} "
        f"patches={len(coords)}",
        flush=True,
    )
    if getattr(args, "save_supporting_images", True):
        thumbnail = wsi_object.wsi.get_thumbnail(
            (args.thumbnail_size, args.thumbnail_size)
        ).convert("RGB")
        thumbnail.save(out_dir / "original_thumbnail.jpg", quality=95)
        mask_level = wsi_object.wsi.get_best_level_for_downsample(32)
        mask_image = wsi_object.visWSI(
            vis_level=mask_level, line_thickness=args.mask_line_thickness,
            number_contours=True,
        )
        mask_image.save(out_dir / "segmentation_mask.jpg", quality=95)
    render_specs = [(
        "titan_pool_attention.jpg",
        scores["titan_pool_attention_percentile"],
        "jet",
    )]
    for filename, values, cmap in render_specs:
        print(f"[RENDER] drawHeatmap start: {filename}", flush=True)
        heatmap = draw_heatmap(
            values.reshape(-1, 1), coords, str(slide_path),
            wsi_object=wsi_object, cmap=cmap, alpha=args.heatmap_alpha,
            segment=True, use_holes=True, binarize=False,
            vis_level=args.heatmap_vis_level,
            blank_canvas=False, thresh=-1,
            patch_size=(args.vis_patch_size, args.vis_patch_size),
            convert_to_percentiles=False,
        )
        if heatmap.mode != "RGB":
            heatmap = heatmap.convert("RGB")
        if getattr(args, "draw_attention_boundary", True):
            heatmap = draw_attention_boundary(
                heatmap, coords, values, wsi_object.wsi.dimensions, args, cv2
            )
        heatmap.save(out_dir / filename, quality=args.jpeg_quality)
        print(f"[RENDER] saved: {out_dir / filename}", flush=True)


def save_h5(path: Path, coords: np.ndarray, adaptation_indices: np.ndarray, scores: dict):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords, compression="gzip")
        handle.create_dataset("adaptation_bag_indices", data=adaptation_indices)
        for name, values in scores.items():
            handle.create_dataset(name, data=values, compression="gzip")


def save_clam_blockmaps(out_dir: Path, slide_id: str, coords: np.ndarray, scores: dict):
    """Write CLAM-compatible blockmaps for downstream rendering/reuse."""
    for method in ("titan_pool_attention",):
        path = out_dir / f"{slide_id}_{method}_blockmap.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "attention_scores",
                data=scores[f"{method}_percentile"].reshape(-1, 1),
                compression="gzip",
            )
            handle.create_dataset("coords", data=coords, compression="gzip")
            handle.attrs["attention_method"] = method
            handle.attrs["score_scale"] = "whole-slide percentile"


def process_slide(spec: dict, model: CASTSlide, args, device, model_info: dict):
    slide_id = spec["slide_id"]
    label = spec["label"].upper()
    slide_path = resolve(spec["wsi"])
    feature_path = resolve(args.feature_dir) / f"{slide_id}.h5"
    out_dir = resolve(args.output_dir) / label / slide_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(feature_path, "r") as handle:
        features_np = handle["features"][:].astype(np.float32, copy=False)
        coords_np = handle["coords"][:].astype(np.int64, copy=False)
    if len(features_np) != len(coords_np):
        raise ValueError(f"Feature/coordinate mismatch for {slide_id}")
    adaptation_indices = representative_indices(
        coords_np, args.k_patches, args.seed, args.spatial_bins
    )
    if args.validate_only:
        print(f"[VALID] {label} {slide_id}: patches={len(coords_np)}")
        return

    model.hard_reset()
    features = torch.from_numpy(features_np).to(device)
    coords = torch.from_numpy(coords_np).long().to(device)
    adapt_idx = torch.as_tensor(adaptation_indices, device=device)
    before_class, before_probs, before_task, before_entropy = model._quick_inference(
        features[adapt_idx], coords[adapt_idx]
    )
    pred_class, probs, pred_task, adapt_log = model.adapt_and_predict(
        features[adapt_idx], coords[adapt_idx]
    )
    scores = aggregate_scores(model, features, coords, coords_np, args)
    save_h5(out_dir / "scores.h5", coords_np, adaptation_indices, scores)
    save_clam_blockmaps(out_dir, slide_id, coords_np, scores)
    metadata = {
        **model_info,
        "slide_id": slide_id,
        "ground_truth": label,
        "configured_split": spec.get("split", "unspecified"),
        "num_patches": len(coords_np),
        "k_patches": args.k_patches,
        "coverage_repeats": args.coverage_repeats,
        "seed": args.seed,
        "before_tta": {
            "predicted_task": before_task,
            "predicted_class": before_class,
            "probabilities": before_probs.flatten().tolist(),
            "entropy": before_entropy,
        },
        "after_tta": {
            "predicted_task": pred_task,
            "predicted_class": pred_class,
            "probabilities": probs.flatten().tolist(),
            "adaptation": adapt_log,
        },
    }
    with (out_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    render_maps(slide_path, coords_np, scores, out_dir, args)
    print(f"[DONE] {label} {slide_id}: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/heatmaps/ood_fold5_figure5.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-bags", type=int, default=0)
    cli = parser.parse_args()
    cfg_path = resolve(cli.config)
    cfg = load_config(cfg_path)
    values = cfg["runtime"] | cfg["tta"] | cfg["visualization"]
    values.setdefault("mode", "tcp")
    values.setdefault("use_teacher", True)
    values.setdefault("tcp_inference_model", "teacher")
    values.setdefault("naive_inference_model", "student")
    values.setdefault("naive_use_task_entropy", True)
    values.setdefault("adapt_task_prompts", True)
    values.setdefault("use_dapc", True)
    values["slides"] = cfg["slides"]
    values["validate_only"] = cli.validate_only
    values["max_bags"] = cli.max_bags
    values["config"] = str(cfg_path)
    return argparse.Namespace(**values)


def main():
    args = parse_args()
    if args.k_patches != K_PATCHES:
        raise ValueError(f"Configured K={args.k_patches}, trained K={K_PATCHES}")
    for spec in args.slides:
        if not resolve(spec["wsi"]).is_file():
            raise FileNotFoundError(resolve(spec["wsi"]))
        feature = resolve(args.feature_dir) / f"{spec['slide_id']}.h5"
        if not feature.is_file():
            raise FileNotFoundError(feature)
    if args.validate_only:
        for spec in args.slides:
            process_slide(spec, None, args, None, {})
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CAST-Slide heatmap generation requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, merged, task_paths = make_tta_model(args, device)
    model_info = {
        "setting": "ood",
        "fold": args.fold,
        "mode": args.mode,
        "merged_checkpoint": str(merged),
        "task_checkpoints": task_paths,
        "attention": ["contrastive_pool_query"],
        "attention_class_specific": False,
        "attention_context_aggregation": "median_across_spatial_K400_bags",
        "inference_model": (
            "ema_teacher"
            if (
                (args.mode == "tcp" and args.tcp_inference_model == "teacher")
                or (
                    args.mode == "naive"
                    and args.naive_inference_model == "teacher"
                )
            )
            else "adapted_student_backbone"
        ),
    }
    for spec in args.slides:
        process_slide(spec, model, args, device, model_info)


if __name__ == "__main__":
    main()
