"""Checkpoint save helpers with optional primary/mirror storage."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "checkpoints"
NFS_MIRROR_DIR = "checkpoints_nfs_mirror"


def get_local_hot_root() -> Path:
    return Path(os.environ.get("MERGESLIDE_LOCAL_ROOT", PROJECT_ROOT)).expanduser()


def get_nfs_checkpoint_mirror_root() -> Path:
    default_root = PROJECT_ROOT / NFS_MIRROR_DIR
    root = Path(os.environ.get("MERGESLIDE_NFS_CHECKPOINT_ROOT", default_root)).expanduser()
    return root if root.is_absolute() else PROJECT_ROOT / root


def _absolute(path: str | Path) -> Path:
    raw_path = Path(path).expanduser()
    return raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path


def _relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def get_checkpoint_mirror_path(primary_path: str | Path) -> Path | None:
    """Return the companion checkpoint path, or None when it would duplicate primary."""
    primary = _absolute(primary_path)
    primary_resolved = primary.resolve(strict=False)

    if get_local_hot_root().resolve(strict=False) == PROJECT_ROOT.resolve(strict=False):
        return None

    docker_checkpoint_root = (get_local_hot_root() / CHECKPOINT_DIR).resolve(strict=False)
    nfs_mirror_root = get_nfs_checkpoint_mirror_root()

    rel_from_docker = _relative_to(primary_resolved, docker_checkpoint_root)
    if rel_from_docker is not None:
        mirror_path = nfs_mirror_root / rel_from_docker
    else:
        repo_checkpoint_root = PROJECT_ROOT / CHECKPOINT_DIR
        rel_from_repo_checkpoints = _relative_to(primary, repo_checkpoint_root)
        if rel_from_repo_checkpoints is not None:
            mirror_path = docker_checkpoint_root / rel_from_repo_checkpoints
        else:
            rel_from_project = _relative_to(primary, PROJECT_ROOT)
            mirror_path = (
                docker_checkpoint_root / rel_from_project
                if rel_from_project is not None
                else docker_checkpoint_root / primary.name
            )

    if mirror_path.resolve(strict=False) == primary_resolved:
        return None
    return mirror_path


def save_checkpoint_with_mirror(obj: Any, primary_path: str | Path) -> tuple[Path, Path | None]:
    """Save checkpoint to primary_path and copy it to the companion mirror path."""
    primary = _absolute(primary_path)
    primary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, primary)

    if os.environ.get("MERGESLIDE_DISABLE_CHECKPOINT_MIRROR") == "1":
        return primary, None

    mirror_path = get_checkpoint_mirror_path(primary)
    if mirror_path is None:
        return primary, None

    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary, mirror_path)
    return primary, mirror_path
