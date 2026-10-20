"""
Model architecture and training utilities shared across train/eval scripts.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from cast_slide.constants import EMBED_DIM, TITAN_PS_ARG


class CustomSequential(nn.Module):
    """
    Wraps TITAN's vision_encoder (backbone) + a task-specific linear head (mlp).
    backbone: TITAN slide aggregator — takes (features, coords, TITAN_PS_ARG)
    mlp:      Linear(EMBED_DIM → n_classes) or nn.Identity() during inference
    """
    def __init__(self, titan_model: nn.Module, mlp: nn.Module):
        super().__init__()
        self.backbone = titan_model.vision_encoder
        self.mlp = mlp

    def forward(self, features: torch.Tensor, coords: torch.Tensor,
                ps: torch.Tensor) -> torch.Tensor:
        x = self.backbone(features, coords, ps)
        return self.mlp(x)


def build_model(task_id: int, num_classes: int, prompt_classifier: torch.Tensor,
                classifier_class_ranges: dict, device: str) -> CustomSequential:
    """
    Load a fresh TITAN model and attach a prompt-initialized, frozen MLP head.

    Args:
        task_id: Index of the current task (0-based).
        num_classes: Number of output classes for this task.
        prompt_classifier: Tensor [EMBED_DIM, TOTAL_CLASSES] from zero_shot_classifier.
        task_class_ranges: Dict mapping task_id → [start, end] class indices.
        device: Target device string.

    Returns:
        CustomSequential model with frozen MLP.
    """
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)

    mlp = nn.Linear(EMBED_DIM, num_classes).to(device)
    mlp.bias.data.zero_()

    # Init MLP weights from prompt prototypes (frozen — never updated during training)
    start, end = classifier_class_ranges[task_id]
    mlp.weight.data = prompt_classifier[:, start:end + 1].T

    for param in mlp.parameters():
        param.requires_grad = False

    return CustomSequential(titan, mlp).to(device)


def build_prompt_classifier(device: str) -> tuple[torch.Tensor, list]:
    """
    Build the 13-class zero-shot classifier from TITAN's text encoder.
    Returns (classifier_tensor [EMBED_DIM, 13], templates).
    Heavy operation — call once and reuse.
    """
    from cast_slide.prompts_zeroshot import (
        brca_prompts, rcc_prompts, nsclc_prompts,
        esca_prompts, tgct_prompts, cesc_prompts,
    )

    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan = titan.to(device)

    _, templates = brca_prompts()
    all_prompts = []
    prompt_functions = [
        brca_prompts, rcc_prompts, nsclc_prompts,
        esca_prompts, tgct_prompts, cesc_prompts,
    ]
    for fn in prompt_functions:
        class_prompts, _ = fn()
        all_prompts.extend(class_prompts)

    with torch.autocast("cuda", torch.float16), torch.inference_mode():
        classifier = titan.zero_shot_classifier(all_prompts, templates, device=device)

    del titan  # free VRAM before training loop
    torch.cuda.empty_cache()
    return classifier, templates


def cosine_lr(optimizer: torch.optim.Optimizer, base_lr: float,
              warmup_length: int, steps: int):
    """Cosine LR scheduler with linear warmup. Returns a step-callable."""
    def _assign_lr(new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr * pg.get("lr_scale", 1.0)

    def _step(step: int) -> float:
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_lr(lr)
        return lr

    return _step


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
