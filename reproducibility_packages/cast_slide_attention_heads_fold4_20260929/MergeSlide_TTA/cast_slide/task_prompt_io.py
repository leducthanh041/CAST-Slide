from pathlib import Path

import torch


TASK_PROMPT_ARTIFACT_ORDERS = {
    6: ["BRCA", "RCC", "NSCLC", "ESCA", "TGCT", "CESC"],
    7: ["BRCA", "RCC", "NSCLC", "HEROHE", "ESCA", "TGCT", "CESC"],
}


def load_task_prompts_for_tasks(
    path: Path,
    task_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    task_prompts = torch.load(path, map_location="cpu")
    if task_prompts.ndim != 2:
        raise ValueError(
            f"Expected task prompt tensor [T, D], got {tuple(task_prompts.shape)}"
        )

    artifact_order = TASK_PROMPT_ARTIFACT_ORDERS.get(task_prompts.shape[0])
    if artifact_order is None:
        raise ValueError(
            f"{path} must contain 6 or 7 task rows; got {task_prompts.shape[0]}"
        )

    missing = [name for name in task_names if name not in artifact_order]
    if missing:
        raise ValueError(f"Task prompt artifact has no rows for tasks: {missing}")

    row_by_name = {name: row for row, name in enumerate(artifact_order)}
    row_ids = [row_by_name[name] for name in task_names]
    return task_prompts[row_ids].to(device)
