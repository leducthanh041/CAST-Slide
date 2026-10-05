"""Losses and reliability selection for CAST-Slide."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy over batch. logits: [N, C]"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
    """Marginal entropy over batch (SHOT diversity term). logits: [N, C]
    Only valid when rows correspond to DIFFERENT true labels (e.g. class-level,
    multiple sub-bags may genuinely contain different tissue/class content).
    Do NOT use this for task-routing logits of sub-bags from the same slide.
    """
    mean_probs = F.softmax(logits, dim=1).mean(dim=0).clamp(min=1e-8)
    return -(mean_probs * mean_probs.log()).sum()


def task_agreement_loss(task_logits: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon-Divergence based agreement loss across M sub-bags.

    All M sub-bags come from the SAME slide -> should agree on task routing.
    Minimizing this loss pulls the routing distributions of the sub-bags
    toward a consistent task decision.

    task_logits: [M, T]
    Returns a scalar in [0, log 2] (JSD upper bound), 0 = perfect agreement.
    """
    probs = F.softmax(task_logits, dim=1).clamp(min=1e-8)
    mean_probs = probs.mean(dim=0, keepdim=True).clamp(min=1e-8)
    kl = (probs * (probs.log() - mean_probs.log())).sum(dim=1)
    return kl.mean()


def dual_level_tta_loss(
    class_logits:        torch.Tensor,
    task_logits:         Optional[torch.Tensor],
    alpha:                float = 0.5,
    class_weight:         float = 1.0,
    use_task_diversity:   bool  = False,
    use_task_agreement:   bool  = True,
    gamma:                float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined class-level and task-level objective.

    L_class = H_class_ent
    L_task  = H_task_ent [- H_task_div] + gamma * JSD_agreement

    total = class_weight * L_class + alpha * L_task

    Args:
        class_logits        : [N, C_task] (tcp) or [N, C_total] (naive)
        task_logits          : [N, T], or None when task branch is disabled
        alpha                : task loss weight (0 disables the task branch)
        class_weight         : class loss weight (0 disables the class branch)
        use_task_diversity   : optional diversity term on task logits
        use_task_agreement   : JSD consistency across sub-bags' task routing
        gamma                : weight of the agreement term relative to l_task_ent
    """
    l_class_ent = entropy_loss(class_logits)
    l_class = l_class_ent

    if task_logits is None or alpha == 0.0:
        l_task_ent = class_logits.new_zeros(())
        l_task_div = class_logits.new_zeros(())
        l_task_agree = class_logits.new_zeros(())
        l_task = class_logits.new_zeros(())
    else:
        l_task_ent = entropy_loss(task_logits)
        l_task_div = (
            diversity_loss(task_logits)
            if use_task_diversity
            else task_logits.new_zeros(())
        )
        l_task_agree = (
            task_agreement_loss(task_logits)
            if use_task_agreement
            else task_logits.new_zeros(())
        )
        l_task = l_task_ent - l_task_div + gamma * l_task_agree

    total = class_weight * l_class + alpha * l_task

    log = {
        "loss/class_ent":    l_class_ent.item(),
        "loss/class_weighted": (class_weight * l_class).item(),
        "loss/task_ent":     l_task_ent.item(),
        "loss/task_div":     l_task_div.item() if use_task_diversity else 0.0,
        "loss/task_agree":   l_task_agree.item() if use_task_agreement else 0.0,
        "loss/total":        total.item(),
    }
    return total, log


def select_confident_subbags(
    logits:    torch.Tensor,
    top_ratio: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-(top_ratio * N) sub-bags with lowest entropy (most confident).
    TPT-style confidence selection.

    Args:
        logits    : [N, C]
        top_ratio : fraction to keep

    Returns:
        selected_logits : [K, C]
        selected_idx    : [K]
    """
    with torch.no_grad():
        ent = -(F.softmax(logits, dim=1).clamp(min=1e-8) *
                F.log_softmax(logits, dim=1)).sum(dim=1)
    k   = max(1, int(ent.size(0) * top_ratio))
    idx = torch.argsort(ent)[:k]
    return logits[idx], idx


def select_confident_subbags_intersection(
    class_logits: torch.Tensor,
    task_logits:  torch.Tensor,
    top_ratio:    float = 0.5,
) -> torch.Tensor:
    """
    EATA-style selection keeps only sub-bags that are
    confident on BOTH class and task logits (intersection), instead of
    the union used in v1 (which lets a task-confident-but-wrong sub-bag
    leak into the gradient via the class branch, and vice versa).

    Falls back to the union's top-1 if intersection is empty, so at least
    one sub-bag is always kept.

    Returns:
        selected_idx : [K] (K may be < top_ratio*N)
    """
    _, idx_class = select_confident_subbags(class_logits, top_ratio)
    _, idx_task  = select_confident_subbags(task_logits,  top_ratio)
    set_class = set(idx_class.tolist())
    set_task  = set(idx_task.tolist())
    inter = sorted(set_class & set_task)
    if len(inter) == 0:
        inter = [int(idx_class[0].item())]
    return torch.tensor(inter, dtype=torch.long, device=class_logits.device)


def select_confident_subbags_by_mode(
    class_logits: torch.Tensor,
    task_logits: Optional[torch.Tensor],
    top_ratio: float,
    mode: str,
) -> torch.Tensor:
    if mode == "task_only" and task_logits is None:
        raise RuntimeError("task_only selection requires task logits")
    if task_logits is None or mode == "class_only":
        return select_confident_subbags(class_logits, top_ratio)[1]
    if mode == "task_only":
        return select_confident_subbags(task_logits, top_ratio)[1]
    if mode == "intersection":
        return select_confident_subbags_intersection(
            class_logits, task_logits, top_ratio
        )
    if mode == "union":
        idx_class = select_confident_subbags(class_logits, top_ratio)[1]
        idx_task = select_confident_subbags(task_logits, top_ratio)[1]
        return torch.unique(torch.cat([idx_class, idx_task]))
    raise ValueError(f"unsupported selection mode: {mode}")


def task_margin_loss(
    embeds:       torch.Tensor,
    task_prompts: torch.Tensor,
    margin:       float = 0.1,
) -> torch.Tensor:
    """
    Push slide embeddings to separate the top-1 routed task from the
    runner-up task in prompt-similarity space, WITHOUT assuming the
    current routing is correct (unlike an entropy/CE loss against a
    pseudo task-label). This is complementary to task_agreement_loss:
      - task_agreement_loss (JSD) : sub-bags of the SAME slide should
        AGREE with each other on which task they belong to.
      - task_margin_loss (this)   : whichever task currently leads should
        lead by a clear margin over the runner-up (sharper routing surface).

    embeds       : [N, 768] student embeddings (grad-enabled)
    task_prompts : [T, 768] (use .detach() at call site -- this loss should
                   only shape the embedding space, not the prompts directly)
    margin       : desired gap between top-1 and top-2 task score

    Returns 0 when the gap already exceeds `margin` for a given sub-bag.
    """
    scores = embeds.float() @ task_prompts.detach().T
    top2 = scores.topk(2, dim=-1).values
    return F.relu(top2[:, 1] - top2[:, 0] + margin).mean()


def l2_anchor_loss(
    params:   List[torch.Tensor],
    params_0: List[torch.Tensor],
) -> torch.Tensor:
    """
    L2 regularization toward initial parameter values (EATA simplified).

    params   : current LN params (requires_grad=True)
    params_0 : initial LN params (detached)
    """
    return sum(((p - p0) ** 2).sum() for p, p0 in zip(params, params_0))
