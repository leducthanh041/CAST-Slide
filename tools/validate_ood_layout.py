#!/usr/bin/env python3
"""Validate the local OOD dataset/checkpoint layout without loading artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MERGESLIDE_DATA_ROOT", PROJECT_ROOT.parent / "dataset")).expanduser()
CHECKPOINT_ROOT = Path(
    os.environ.get("MERGESLIDE_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints_ood")
).expanduser()
TASKS = (
    ("brca", "TCGA-BRCA_processed"),
    ("rcc", "TCGA-RCC_processed"),
    ("nsclc", "TCGA-NSCLC_processed"),
    ("esca", "TCGA-ESCA_processed"),
    ("tgct", "TCGA-TGCT_processed"),
    ("cesc", "TCGA-CESC_processed"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "merge", "eval"), default="eval")
    args = parser.parse_args()

    missing: list[Path] = []
    for task_name, feature_dir in TASKS:
        feature_root = DATA_ROOT / feature_dir / "features"
        for child in ("h5_files", "pt_files"):
            path = feature_root / child
            if not path.is_dir():
                missing.append(path)
        split_root = DATA_ROOT / "wsi_dataset_annotation_cross_sites" / f"tcga_{task_name}"
        for fold in range(10):
            path = split_root / f"splits_{fold}.csv"
            if not path.is_file():
                missing.append(path)

    if args.stage in {"merge", "eval"}:
        for fold in range(10):
            for task in range(6):
                path = CHECKPOINT_ROOT / "finetuned" / f"fold_{fold}" / f"task_{task}.pt"
                if not path.is_file():
                    missing.append(path)

    if args.stage == "eval":
        for fold in range(10):
            fold_root = CHECKPOINT_ROOT / "merged" / f"fold_{fold}"
            for name in ("merged_final.pth", *(f"merged_task_{task}.pth" for task in range(1, 6))):
                path = fold_root / name
                if not path.is_file():
                    missing.append(path)

    print(f"[LAYOUT] project_root={PROJECT_ROOT}")
    print(f"[LAYOUT] data_root={DATA_ROOT}")
    print(f"[LAYOUT] checkpoint_root={CHECKPOINT_ROOT}")
    if missing:
        print(f"[ERROR] Missing {len(missing)} required paths:")
        for path in missing[:30]:
            print(f"  {path}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        return 1

    print(f"[OK] OOD {args.stage} layout is complete for 6 tasks and 10 folds.")
    extras = sorted((CHECKPOINT_ROOT / "finetuned").glob("fold_*/task_6.pt"))
    if extras:
        print(f"[WARN] Found {len(extras)} task_6.pt files; the configured 6-task stream ignores them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

