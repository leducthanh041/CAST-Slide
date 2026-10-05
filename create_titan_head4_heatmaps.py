"""Calibrate and visualize TITAN Transformer Head #4 on BRCA WSIs.

Calibration uses the original Hugging Face TITAN on the BRCA slide shown in
Extended Data Fig. 5. It renders tensor head indices 3 and 4 at all six
Transformer layers. Target visualization uses a configurable selected layer
and head on fold-5 MergeSlide before and after episodic naive CAST-Slide TTA.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel

from cast_slide.constants import K_PATCHES
from cast_slide.titan_attention import TitanSelectedHeadCapture
from create_cast_slide_heatmaps import (
    make_tta_model,
    percentile,
    render_maps,
    representative_indices,
    resolve,
)
from create_continual_attention_grid import reference_percentiles


def load_args(config_path: Path, cli):
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    values = cfg["runtime"] | cfg["tta"] | cfg["visualization"]
    values["calibration_slide"] = cfg["calibration_slide"]
    values["target_slides"] = cfg["target_slides"]
    values["phase"] = cli.phase
    values["validate_only"] = cli.validate_only
    if cli.layer_index is not None:
        values["selected_layer_index"] = cli.layer_index
    if cli.head_index is not None:
        values["selected_head_index"] = cli.head_index
    return SimpleNamespace(**values)


def load_slide_arrays(spec, args):
    slide_id = spec["slide_id"]
    feature_path = resolve(args.feature_dir) / f"{slide_id}.h5"
    slide_path = resolve(spec["wsi"])
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    with h5py.File(feature_path, "r") as handle:
        features = handle["features"][:].astype(np.float32, copy=False)
        coords = handle["coords"][:].astype(np.int64, copy=False)
    if features.shape != (len(coords), 768):
        raise ValueError(
            f"Expected aligned [N,768] features for {slide_id}; "
            f"features={features.shape} coords={coords.shape}"
        )
    return slide_path, feature_path, features, coords


def selected_head_scores(backbone, features, coords, patch_size, layers, heads):
    """One whole-slide forward; only selected CLS rows are materialized."""
    backbone.eval()
    torch.cuda.reset_peak_memory_stats(features.device)
    torch.cuda.synchronize(features.device)
    started = time.perf_counter()
    print(
        f"[ATTENTION] whole-slide forward start patches={len(features)} "
        f"layers={list(layers)} heads={list(heads)}",
        flush=True,
    )
    layer_indices = tuple(layers)
    progress = tqdm(
        total=len(layer_indices) + 2,
        desc=f"TITAN attention ({len(features)} patches)",
        unit="step",
        dynamic_ncols=True,
    )
    progress.set_postfix_str("building spatial tokens/ALiBi")
    progress.update(1)

    def report_layer(layer_index):
        progress.set_postfix_str(f"captured layer {layer_index + 1}")
        progress.update(1)

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with TitanSelectedHeadCapture(
                backbone,
                layer_indices,
                heads,
                progress_callback=report_layer,
            ) as capture:
                backbone(features, coords, patch_size)
            progress.set_postfix_str("aligning scores to patches")
            scores = capture.patch_scores(coords, int(patch_size))
            progress.update(1)
    finally:
        progress.close()
    torch.cuda.synchronize(features.device)
    print(
        f"[ATTENTION] whole-slide forward done elapsed={time.perf_counter() - started:.1f}s "
        f"peak_vram={torch.cuda.max_memory_allocated(features.device) / 2**30:.2f}GiB",
        flush=True,
    )
    return {
        key: value.float().cpu().numpy().astype(np.float32, copy=False)
        for key, value in scores.items()
    }


def save_score_h5(path, coords, raw_scores, percentile_scores, attrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords, compression="gzip")
        handle.create_dataset("attention_raw", data=raw_scores, compression="gzip")
        handle.create_dataset(
            "attention_percentile", data=percentile_scores, compression="gzip"
        )
        for key, value in attrs.items():
            handle.attrs[key] = value


def render_selected_map(slide_path, coords, values, out_dir, filename, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    render_maps(
        slide_path,
        coords,
        {"titan_pool_attention_percentile": values},
        out_dir,
        args,
    )
    generated = out_dir / "titan_pool_attention.jpg"
    target = out_dir / filename
    generated.replace(target)
    return target


def compose_calibration_grid(paths, output_path):
    layers = sorted({key[0] for key in paths})
    heads = sorted({key[1] for key in paths})
    fig, axes = plt.subplots(len(layers), len(heads), figsize=(5 * len(heads), 4.6 * len(layers)))
    axes = np.asarray(axes).reshape(len(layers), len(heads))
    for row, layer in enumerate(layers):
        for col, head in enumerate(heads):
            axes[row, col].imshow(Image.open(paths[(layer, head)]))
            axes[row, col].set_title(
                f"Transformer layer {layer + 1} | tensor head index {head}",
                fontsize=12,
            )
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compose_target_grid(slide_outputs, output_path):
    labels = list(slide_outputs)
    fig, axes = plt.subplots(len(labels), 2, figsize=(10, 4.8 * len(labels)))
    axes = np.asarray(axes).reshape(len(labels), 2)
    for row, label in enumerate(labels):
        for col, state in enumerate(("source", "adapted")):
            axes[row, col].imshow(Image.open(slide_outputs[label][state]))
            axes[row, col].set_title(f"{label} | {state}", fontsize=15)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_calibration(args, device):
    spec = args.calibration_slide
    slide_path, feature_path, features_np, coords_np = load_slide_arrays(spec, args)
    output_dir = resolve(args.output_dir) / "calibration_TCGA-B6-A0X7"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[CALIBRATION] slide={spec['slide_id']} patches={len(coords_np)} "
        f"layers={args.calibration_layer_indices} heads={args.calibration_head_indices}",
        flush=True,
    )
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    backbone = titan.vision_encoder.to(device)
    features = torch.from_numpy(features_np).to(device)
    coords = torch.from_numpy(coords_np).long().to(device)
    scores = selected_head_scores(
        backbone, features, coords, args.patch_size_lv0,
        args.calibration_layer_indices, args.calibration_head_indices,
    )
    rendered = {}
    for (layer_index, head_index), raw in tqdm(
        sorted(scores.items()), desc="Render Head #4 calibration candidates", unit="map"
    ):
        values = percentile(raw)
        candidate_dir = output_dir / f"layer_{layer_index + 1}" / f"head_index_{head_index}"
        attrs = {
            "model": "MahmoodLab/TITAN pretrained source",
            "attention": "Transformer CLS-to-patch self-attention",
            "transformer_layer_index": layer_index,
            "transformer_layer_number": layer_index + 1,
            "tensor_head_index": head_index,
            "paper_candidate": "Head #4",
            "whole_slide_context": True,
            "num_patches": len(coords_np),
        }
        save_score_h5(candidate_dir / "scores.h5", coords_np, raw, values, attrs)
        with (candidate_dir / "metadata.json").open("w") as handle:
            json.dump({**attrs, "slide_id": spec["slide_id"], "feature_h5": str(feature_path)}, handle, indent=2)
        filename = f"titan_transformer_layer{layer_index + 1}_head_index{head_index}.jpg"
        rendered[(layer_index, head_index)] = render_selected_map(
            slide_path, coords_np, values, candidate_dir, filename, args
        )
    compose_calibration_grid(rendered, output_dir / "head4_layer_index_calibration.jpg")
    del features, coords, backbone, titan
    torch.cuda.empty_cache()
    print(f"[DONE] calibration={output_dir}", flush=True)


def run_targets(args, device):
    layer_index = int(args.selected_layer_index)
    head_index = int(args.selected_head_index)
    output_root = (
        resolve(args.output_dir)
        / f"brca_idc_ilc_layer_{layer_index + 1}_head_index_{head_index}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    model, merged, task_paths = make_tta_model(args, device)
    slide_outputs = {}
    for spec in tqdm(args.target_slides, desc="BRCA Head #4 targets", unit="WSI"):
        slide_path, feature_path, features_np, coords_np = load_slide_arrays(spec, args)
        label = spec["label"].upper()
        slide_dir = output_root / f"{label}_{spec['slide_id']}"
        slide_dir.mkdir(parents=True, exist_ok=True)
        features = torch.from_numpy(features_np).to(device)
        coords = torch.from_numpy(coords_np).long().to(device)
        model.hard_reset()
        source = selected_head_scores(
            model.backbone, features, coords, int(model.ps.item()),
            [layer_index], [head_index],
        )[(layer_index, head_index)]
        adaptation_indices = representative_indices(
            coords_np, args.k_patches, args.seed, args.spatial_bins
        )
        indices = torch.as_tensor(adaptation_indices, device=device)
        pred_class, probs, pred_task, adapt_log = model.adapt_and_predict(
            features[indices], coords[indices]
        )
        adapted = selected_head_scores(
            model.backbone, features, coords, int(model.ps.item()),
            [layer_index], [head_index],
        )[(layer_index, head_index)]
        reference = np.concatenate([source, adapted])
        source_percentile = reference_percentiles(source, reference)
        adapted_percentile = reference_percentiles(adapted, reference)
        common_attrs = {
            "attention": "Transformer CLS-to-patch self-attention",
            "transformer_layer_index": layer_index,
            "transformer_layer_number": layer_index + 1,
            "tensor_head_index": head_index,
            "paper_label_provisional": "Head #4",
            "whole_slide_context": True,
            "fold": args.fold,
            "merged_checkpoint": str(merged),
            "num_patches": len(coords_np),
        }
        outputs = {}
        for state, raw, values in (
            ("source", source, source_percentile),
            ("adapted", adapted, adapted_percentile),
        ):
            state_dir = slide_dir / state
            save_score_h5(
                state_dir / "scores.h5", coords_np, raw, values,
                {**common_attrs, "state": state},
            )
            filename = (
                f"titan_transformer_layer{layer_index + 1}_"
                f"head_index{head_index}_{state}.jpg"
            )
            outputs[state] = render_selected_map(
                slide_path, coords_np, values, state_dir, filename, args
            )
        delta = adapted.astype(np.float64) - source.astype(np.float64)
        metadata = {
            **common_attrs,
            "slide_id": spec["slide_id"],
            "label": label,
            "feature_h5": str(feature_path),
            "task_checkpoints": task_paths,
            "adaptation_indices": adaptation_indices.tolist(),
            "prediction": {
                "global_class": pred_class,
                "task": pred_task,
                "confidence": float(probs.max().item()),
                "adaptation": adapt_log,
            },
            "source_adapted_change": {
                "mean_absolute_delta": float(np.mean(np.abs(delta))),
                "maximum_absolute_delta": float(np.max(np.abs(delta))),
                "pearson": float(np.corrcoef(source, adapted)[0, 1]),
            },
        }
        with (slide_dir / "metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
        slide_outputs[label] = outputs
        del features, coords
        torch.cuda.empty_cache()
    compose_target_grid(slide_outputs, output_root / "idc_ilc_head4_source_vs_adapted.jpg")
    print(f"[DONE] targets={output_root}", flush=True)


def validate(args):
    if args.k_patches != K_PATCHES:
        raise ValueError(f"Configured K={args.k_patches}, expected K={K_PATCHES}")
    if sorted(args.calibration_layer_indices) != list(range(6)):
        raise ValueError("Calibration must cover all six TITAN Transformer layers")
    if sorted(args.calibration_head_indices) != [3, 4]:
        raise ValueError("Calibration must compare tensor head indices 3 and 4")
    if not 0 <= args.selected_layer_index < 6:
        raise ValueError("selected_layer_index must be in [0,5]")
    if not 0 <= args.selected_head_index < 12:
        raise ValueError("selected_head_index must be in [0,11]")
    specs = [args.calibration_slide, *args.target_slides]
    for spec in specs:
        slide_path, feature_path, features, coords = load_slide_arrays(spec, args)
        print(
            f"[VALID] {spec.get('label', 'CALIBRATION')} slide={spec['slide_id']} "
            f"patches={len(coords)} raw={slide_path} features={feature_path}",
            flush=True,
        )
    fold_dir = f"fold_{args.fold}"
    checkpoint = resolve(args.merged_dir) / fold_dir / "merged_final.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(
        f"[VALID] selected paper Head #4 = layer_index "
        f"{args.selected_layer_index}, tensor_head_index {args.selected_head_index}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/attention_heads/ood_fold5_brca_head4.yaml")
    parser.add_argument("--phase", choices=("calibration", "targets", "all"), default="all")
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--head-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    cli = parser.parse_args()
    args = load_args(resolve(cli.config), cli)
    validate(args)
    if args.validate_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("TITAN Head #4 extraction requires CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    if args.phase in ("calibration", "all"):
        run_calibration(args, device)
    if args.phase in ("targets", "all"):
        run_targets(args, device)


if __name__ == "__main__":
    main()
