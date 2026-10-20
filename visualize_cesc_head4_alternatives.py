"""Visualize unreviewed CESC candidates from cached pre-CESC TTA snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from create_cast_slide_heatmaps import render_maps, resolve, save_clam_blockmaps
from create_continual_attention_grid import (
    extract_attention_scores,
    fit_image_to_canvas,
    load_args,
    load_features,
    naive_prediction,
    reference_percentiles,
    stage_checkpoint,
)
from create_cast_slide_heatmaps import make_tta_model


def restore_pre_cesc_snapshot(model, snapshot_path: Path, device):
    payload = torch.load(snapshot_path, map_location=device)
    backbone_state = model.backbone.state_dict()
    for name, value in payload["backbone_adapted"].items():
        if name not in backbone_state:
            raise KeyError(f"Snapshot backbone parameter is absent: {name}")
        backbone_state[name].copy_(value.to(device))
    if model.teacher is not None:
        teacher_state = model.teacher.state_dict()
        for name, value in payload.get("teacher_adapted", {}).items():
            if name not in teacher_state:
                raise KeyError(f"Snapshot teacher parameter is absent: {name}")
            teacher_state[name].copy_(value.to(device))
    model.optimizer.load_state_dict(payload["optimizer"])
    model.n_adapted = int(payload.get("n_adapted", 0))
    model.n_skipped = int(payload.get("n_skipped", 0))
    return payload


def save_stage_scores(path, coords, raw, percentiles, adaptation_indices, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords, compression="gzip")
        handle.create_dataset("attention_raw", data=raw, compression="gzip")
        handle.create_dataset(
            "attention_percentile", data=percentiles, compression="gzip"
        )
        handle.create_dataset("adaptation_indices", data=adaptation_indices)
        for key in ("stage", "transformer_layer_index", "tensor_head_index"):
            handle.attrs[key] = metadata[key]


def compose_candidate(candidate_dir: Path, slide_id: str):
    paths = [
        candidate_dir / f"stage_{stage}" / "titan_head4_attention.jpg"
        for stage in range(1, 7)
    ]
    fig, axes = plt.subplots(1, 6, figsize=(21.0, 3.8))
    titles = [f"Stage {stage}" for stage in range(1, 7)]
    for ax, path, title in zip(axes, paths, titles):
        ax.imshow(Image.open(path), aspect="auto")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(slide_id, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.03, wspace=0.035)
    fig.savefig(
        candidate_dir / "head4_attention_6stages.jpg",
        dpi=180, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def run_candidate(candidate, args, device, base_output: Path):
    slide_id = candidate["slide_id"]
    candidate_dir = base_output / slide_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate.setdefault("specimen_crop_margin", args.specimen_crop_margin)

    features, coords = load_features(Path(candidate["feature_h5"]), device)
    coords_np = coords.cpu().numpy()
    stage_results = {}
    for stage in tqdm(range(1, 7), desc=slide_id, unit="stage"):
        checkpoint = stage_checkpoint(args, stage)
        model, _, task_paths = make_tta_model(args, device, checkpoint)
        snapshot_path = (
            resolve(args.output_dir) / "pre_CESC_snapshots" /
            f"stage_{stage}_pre_CESC_ln_state.pt"
        )
        snapshot = restore_pre_cesc_snapshot(model, snapshot_path, device)
        prediction = naive_prediction(
            model, features, coords, args, reset_before_slide=False
        )
        raw = extract_attention_scores(
            model, features, coords, coords_np, args
        )["titan_pool_attention_median"]
        stage_results[stage] = {
            "raw": raw,
            "prediction": prediction,
            "checkpoint": str(checkpoint),
            "task_checkpoints": task_paths,
            "snapshot": str(snapshot_path),
            "pre_cesc_n_adapted": int(snapshot.get("n_adapted", 0)),
            "pre_cesc_n_skipped": int(snapshot.get("n_skipped", 0)),
        }
        del model
        torch.cuda.empty_cache()

    reference = np.concatenate([stage_results[stage]["raw"] for stage in range(1, 7)])
    for stage in range(1, 7):
        result = stage_results[stage]
        percentiles = reference_percentiles(result["raw"], reference)
        stage_dir = candidate_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "slide_id": slide_id,
            "stage": stage,
            "split": candidate["split"],
            "num_patches": candidate["num_patches"],
            "attention": "TITAN final-layer selected-head CLS-to-patch attention",
            "attention_backend": "selected_transformer_head",
            "transformer_layer_index": int(args.selected_layer_index),
            "tensor_head_index": int(args.selected_head_index),
            "inference": "naive adapted student",
            "adaptation_protocol": "restored_online_causal_pre_CESC_snapshot",
            **{key: value for key, value in result.items() if key != "raw"},
        }
        save_stage_scores(
            stage_dir / "scores.h5", coords_np, result["raw"], percentiles,
            result["prediction"]["adaptation_indices"], metadata,
        )
        with (stage_dir / "metadata.json").open("w") as handle:
            json.dump(
                metadata, handle, indent=2, allow_nan=False,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray) else value,
            )
        score_dict = {"titan_pool_attention_percentile": percentiles}
        render_maps(Path(candidate["raw_wsi"]), coords_np, score_dict, stage_dir, args)
        save_clam_blockmaps(stage_dir, slide_id, coords_np, score_dict)
        generated = stage_dir / "titan_pool_attention.jpg"
        heatmap = Image.open(generated).convert("RGB")
        heatmap = fit_image_to_canvas(heatmap, args.panel_width, args.panel_height)
        heatmap.save(stage_dir / "titan_head4_attention.jpg", quality=args.jpeg_quality)
        generated.unlink()
    compose_candidate(candidate_dir, slide_id)
    # The requested deliverable is one six-stage JPG per CESC slide. Scores
    # and metadata remain, while stage JPGs are temporary composition inputs.
    for stage in range(1, 7):
        stage_image = candidate_dir / f"stage_{stage}" / "titan_head4_attention.jpg"
        if stage_image.is_file():
            stage_image.unlink()
    with (candidate_dir / "candidate.json").open("w") as handle:
        json.dump(candidate, handle, indent=2, allow_nan=False)
    del features, coords
    torch.cuda.empty_cache()
    print(f"[DONE] CESC candidate={candidate_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/attention_maps/ood_fold4_naive_6task_head4.yaml"
    )
    parser.add_argument("--ranks", default="3,4,5,6,7,8")
    cli = parser.parse_args()
    args = load_args(resolve(cli.config), validate_only=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CESC Head #4 visualization requires CUDA")
    ranking_path = resolve(args.output_dir) / "CESC_head4_candidate_ranking.json"
    with ranking_path.open() as handle:
        ranked = json.load(handle)
    ranks = [int(value) for value in cli.ranks.split(",") if value.strip()]
    if any(rank < 1 or rank > len(ranked) for rank in ranks):
        raise ValueError(f"Ranks must be in [1,{len(ranked)}]: {ranks}")
    output = resolve(args.output_dir) / "CESC_all_unreviewed_comparison"
    output.mkdir(parents=True, exist_ok=True)
    preserved = ranked[1]
    with (output / "preserved_reference.json").open("w") as handle:
        json.dump(preserved, handle, indent=2, allow_nan=False)
    print(f"[PRESERVED] reference={preserved['slide_id']}", flush=True)
    device = torch.device("cuda")
    for rank in tqdm(ranks, desc="Unreviewed CESC slides", unit="WSI"):
        candidate = ranked[rank - 1]
        completed = output / candidate["slide_id"] / "head4_attention_6stages.jpg"
        if completed.is_file():
            print(
                f"[SKIP] rank={rank} already completed: {completed}", flush=True
            )
            continue
        print(f"[CANDIDATE] rank={rank} slide={candidate['slide_id']}", flush=True)
        run_candidate(candidate, args, device, output)


if __name__ == "__main__":
    main()
