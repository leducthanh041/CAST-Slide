"""Promote one cached CESC Head-4 candidate into the six-task final grid.

This is a render-only utility: it reuses cached coordinates and attention
scores and never reruns TITAN inference or CAST-Slide adaptation.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import cv2
import numpy as np
from PIL import Image

from create_cast_slide_heatmaps import render_maps, resolve
from create_continual_attention_grid import compose_grid, fit_image_to_canvas, load_args


DEFAULT_SLIDE = (
    "TCGA-C5-A1BJ-01A-01-TSA."
    "f24c19a0-b8af-44e5-a88f-a5b8e55aa55b"
)


def two_specimen_boxes(image: Image.Image, margin: int = 28):
    """Find the two dominant tissue components on the shared 2048px canvas."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = ((hsv[..., 1] > 18) & (gray < 245)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = sorted(
        (stats[index] for index in range(1, count)),
        key=lambda row: int(row[cv2.CC_STAT_AREA]), reverse=True,
    )[:2]
    if len(components) != 2:
        raise RuntimeError(f"Expected two tissue specimens, detected {len(components)}")
    boxes = []
    for component in components:
        x, y, width, height, _ = map(int, component)
        boxes.append((
            max(0, x - margin), max(0, y - margin),
            min(image.width, x + width + margin),
            min(image.height, y + height + margin),
        ))
    return sorted(boxes, key=lambda box: box[0])


def isolate_left_specimen(image: Image.Image, boxes, width: int, height: int):
    """Keep only the left specimen and enlarge it without distortion."""
    return fit_image_to_canvas(image.convert("RGB").crop(boxes[0]), width, height)


def save_full_original(candidate: dict, output: Path, args):
    import openslide

    slide = openslide.OpenSlide(candidate["raw_wsi"])
    try:
        image = slide.get_thumbnail(
            (args.original_thumbnail_size, args.original_thumbnail_size)
        ).convert("RGB")
    finally:
        slide.close()
    image = fit_image_to_canvas(image, args.panel_width, args.panel_height)
    boxes = two_specimen_boxes(image)
    image = isolate_left_specimen(
        image, boxes, args.panel_width, args.panel_height
    )
    path = output / "original_wsi.jpg"
    image.save(path, quality=args.jpeg_quality)
    return path, boxes


def render_cached_stages(source: Path, output: Path, candidate: dict, boxes, args):
    heatmaps = {}
    for stage in range(1, 7):
        score_path = source / f"stage_{stage}" / "scores.h5"
        if not score_path.is_file():
            raise FileNotFoundError(f"Missing cached stage scores: {score_path}")
        with h5py.File(score_path, "r") as handle:
            coords = np.asarray(handle["coords"])
            percentiles = np.asarray(handle["attention_percentile"])
        stage_dir = output / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        render_maps(
            Path(candidate["raw_wsi"]), coords,
            {"titan_pool_attention_percentile": percentiles}, stage_dir, args,
        )
        generated = stage_dir / "titan_pool_attention.jpg"
        image = fit_image_to_canvas(
            Image.open(generated).convert("RGB"),
            args.panel_width, args.panel_height,
        )
        image = isolate_left_specimen(
            image, boxes, args.panel_width, args.panel_height
        )
        image.save(generated, quality=args.jpeg_quality)
        heatmaps[stage] = generated
        print(f"[RENDER] stage={stage} output={generated}", flush=True)
    return heatmaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/attention_maps/ood_fold4_naive_6task_head4.yaml",
    )
    parser.add_argument("--slide-id", default=DEFAULT_SLIDE)
    cli = parser.parse_args()
    args = load_args(resolve(cli.config), validate_only=False)
    root = resolve(args.output_dir)
    source = root / "CESC_all_unreviewed_comparison" / cli.slide_id
    with (source / "candidate.json").open() as handle:
        candidate = json.load(handle)

    promoted = root / "CESC_promoted" / cli.slide_id
    promoted.mkdir(parents=True, exist_ok=True)
    original, boxes = save_full_original(candidate, promoted, args)
    cesc_heatmaps = render_cached_stages(
        source, promoted, candidate, boxes, args
    )

    with (root / "selected_slides.json").open() as handle:
        selected = json.load(handle)
    if [row["task_name"] for row in selected] != [
        "BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC"
    ]:
        raise RuntimeError("Unexpected selected_slides.json task ordering")
    candidate["task_name"] = "CESC"
    selected[-1] = candidate

    original_paths = [root / row["task_name"] / "original_wsi.jpg" for row in selected[:-1]]
    original_paths.append(original)
    heatmap_paths = {}
    for stage in range(1, 7):
        for row, item in enumerate(selected[:-1]):
            heatmap_paths[(stage, row)] = (
                root / item["task_name"] / f"stage_{stage}" / "titan_pool_attention.jpg"
            )
        heatmap_paths[(stage, 5)] = cesc_heatmaps[stage]

    final_path = root / args.figure_filename
    backup = final_path.with_name(final_path.stem + "_before_CESC_A1BJ.jpg")
    if final_path.is_file() and not backup.exists():
        shutil.copy2(final_path, backup)
        print(f"[BACKUP] {backup}", flush=True)
    compact_backup = final_path.with_name(
        final_path.stem + "_before_CESC_specimen_compaction.jpg"
    )
    if final_path.is_file() and not compact_backup.exists():
        shutil.copy2(final_path, compact_backup)
        print(f"[BACKUP] {compact_backup}", flush=True)
    left_only_backup = final_path.with_name(
        final_path.stem + "_before_CESC_left_specimen_only.jpg"
    )
    if final_path.is_file() and not left_only_backup.exists():
        shutil.copy2(final_path, left_only_backup)
        print(f"[BACKUP] {left_only_backup}", flush=True)
    compose_grid(selected, original_paths, heatmap_paths, final_path, args)
    manifest = promoted / "promoted_candidate.json"
    with manifest.open("w") as handle:
        json.dump(candidate, handle, indent=2, allow_nan=False)
    with (promoted / "specimen_compaction.json").open("w") as handle:
        json.dump({
            "source_boxes_xyxy": boxes,
            "selected_box_xyxy": boxes[0],
            "layout": "left specimen only, aspect-preserving fit",
            "canvas": [args.panel_width, args.panel_height],
        }, handle, indent=2)
    print(f"[DONE] final={final_path}", flush=True)
    print(f"[DONE] promoted_panels={promoted}", flush=True)


if __name__ == "__main__":
    main()
