"""Extract native TITAN token attention without modifying cached HF code."""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F


def _head_average_self_attention(module, x, attn_bias):
    """Reconstruct the exact pre-dropout SDPA weights used by TITAN."""
    batch, tokens, channels = x.shape
    qkv = module.qkv(x).reshape(
        batch, tokens, 3, module.num_heads, module.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, _ = qkv.unbind(0)
    query = module.q_norm(query)
    key = module.k_norm(key)
    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
    logits.mul_(module.scale)
    if attn_bias is not None:
        logits.add_(attn_bias.float())
    return logits.softmax(dim=-1).mean(dim=1)


def _selected_cls_self_attention(module, x, attn_bias, head_indices):
    """Reconstruct exact CLS-to-all-token attention for selected TITAN heads."""
    batch, tokens, _ = x.shape
    qkv = module.qkv(x).reshape(
        batch, tokens, 3, module.num_heads, module.head_dim
    ).permute(2, 0, 3, 1, 4)
    query, key, _ = qkv.unbind(0)
    query = module.q_norm(query)[:, head_indices, :1]
    key = module.k_norm(key)[:, head_indices]
    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
    logits.mul_(module.scale)
    if attn_bias is not None:
        bias = attn_bias.float()
        if bias.dim() == 4:
            bias = bias[:, head_indices, :1, :]
        elif bias.dim() == 3:
            bias = bias[head_indices, :1, :].unsqueeze(0)
        else:
            raise RuntimeError(f"Unsupported TITAN attention bias shape: {bias.shape}")
        logits.add_(bias)
    return logits.softmax(dim=-1).squeeze(-2)


def _project_mha_query_key(module, query, key):
    if module.in_proj_weight is not None:
        q_weight, k_weight, _ = module.in_proj_weight.chunk(3)
        if module.in_proj_bias is None:
            q_bias = k_bias = None
        else:
            q_bias, k_bias, _ = module.in_proj_bias.chunk(3)
    else:
        q_weight = module.q_proj_weight
        k_weight = module.k_proj_weight
        q_bias = None if module.in_proj_bias is None else module.in_proj_bias[:module.embed_dim]
        k_bias = (
            None if module.in_proj_bias is None
            else module.in_proj_bias[module.embed_dim:2 * module.embed_dim]
        )
    return F.linear(query, q_weight, q_bias), F.linear(key, k_weight, k_bias)


def _head_average_pool_attention(module, query, key):
    """Return cross-attention from TITAN's contrastive pooling query."""
    query, key = _project_mha_query_key(module, query, key)
    target_len, batch, _ = query.shape
    source_len = key.shape[0]
    head_dim = module.head_dim
    query = query.view(target_len, batch, module.num_heads, head_dim).permute(1, 2, 0, 3)
    key = key.view(source_len, batch, module.num_heads, head_dim).permute(1, 2, 0, 3)
    logits = torch.matmul(query.float(), key.float().transpose(-2, -1))
    logits.mul_(head_dim ** -0.5)
    return logits.softmax(dim=-1).mean(dim=1)


def _patch_token_mapping(coords: torch.Tensor, patch_size: int):
    """Map TITAN's spatially sorted non-background tokens to input rows."""
    coords = coords.squeeze(0) if coords.dim() == 3 else coords
    offset = coords.min(dim=0).values
    grid = torch.floor_divide(coords - offset, patch_size)
    grid = grid - grid.min(dim=0).values
    width = int(grid[:, 1].max().item()) + 1
    flat = grid[:, 0] * width + grid[:, 1]
    unique_flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    return unique_flat, inverse


def _expand_token_scores(token_scores, inverse, num_rows):
    if token_scores.numel() != int(inverse.max().item()) + 1:
        raise RuntimeError(
            "TITAN attention token count does not match occupied coordinate cells: "
            f"tokens={token_scores.numel()}, cells={int(inverse.max().item()) + 1}"
        )
    result = token_scores[inverse]
    if result.numel() != num_rows:
        raise RuntimeError("Failed to align TITAN attention with input patch rows")
    return result


class TitanAttentionCapture(AbstractContextManager):
    """Capture native self/pooling attention during one TITAN forward pass."""

    def __init__(self, backbone, capture_self_attention: bool = True):
        self.backbone = backbone
        self.capture_self_attention = capture_self_attention
        self.self_attention = []
        self.pool_attention = None
        self.handles = []

    def _self_pre_hook(self, module, args, kwargs):
        x = args[0]
        attn_bias = kwargs.get("attn_bias", args[1] if len(args) > 1 else None)
        self.self_attention.append(
            _head_average_self_attention(module, x, attn_bias).detach()
        )

    def _pool_pre_hook(self, module, args, kwargs):
        query, key = args[:2]
        self.pool_attention = _head_average_pool_attention(
            module, query, key
        ).detach()

    def __enter__(self):
        if self.capture_self_attention:
            blocks = self.backbone.blocks.modules_list
            for block in blocks:
                self.handles.append(
                    block.attn.register_forward_pre_hook(
                        self._self_pre_hook, with_kwargs=True
                    )
                )
        pooler = self.backbone.attn_pool_contrastive
        if pooler is None:
            raise RuntimeError("TITAN backbone has no contrastive attention pooler")
        self.handles.append(
            pooler.attn.register_forward_pre_hook(
                self._pool_pre_hook, with_kwargs=True
            )
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def patch_scores(self, coords: torch.Tensor, patch_size: int):
        if not self.self_attention or self.pool_attention is None:
            raise RuntimeError("No TITAN attention was captured")
        layers = self.self_attention
        tokens = layers[0].shape[-1]
        if any(layer.shape[-2:] != (tokens, tokens) for layer in layers):
            raise RuntimeError("Inconsistent TITAN self-attention token shapes")

        identity = torch.eye(tokens, device=layers[0].device).unsqueeze(0)
        rollout = identity
        for layer in layers:
            augmented = layer.float() + identity
            augmented = augmented / augmented.sum(dim=-1, keepdim=True)
            rollout = torch.bmm(augmented, rollout)

        pool = self.pool_attention.float()[0, 0, 1:]
        last = layers[-1].float()[0, 0, 1:]
        rolled = rollout.float()[0, 0, 1:]
        _, inverse = _patch_token_mapping(coords, patch_size)
        num_rows = coords.squeeze(0).shape[0]
        return {
            "titan_pool_attention": _expand_token_scores(pool, inverse, num_rows),
            "titan_last_layer_attention": _expand_token_scores(last, inverse, num_rows),
            "titan_attention_rollout": _expand_token_scores(rolled, inverse, num_rows),
        }

    def pool_patch_scores(self, coords: torch.Tensor, patch_size: int):
        """Return only the attention that directly forms TITAN's slide embedding."""
        if self.pool_attention is None:
            raise RuntimeError("No TITAN pooling attention was captured")
        pool = self.pool_attention.float()[0, 0, 1:]
        _, inverse = _patch_token_mapping(coords, patch_size)
        num_rows = coords.squeeze(0).shape[0]
        return _expand_token_scores(pool, inverse, num_rows)


class TitanSelectedHeadCapture(AbstractContextManager):
    """Capture selected transformer-head CLS attention without an N x N map."""

    def __init__(
        self, backbone, layer_indices, head_indices, progress_callback=None
    ):
        self.backbone = backbone
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.head_indices = tuple(int(value) for value in head_indices)
        self.progress_callback = progress_callback
        self.attention = {}
        self.handles = []

    def _make_hook(self, layer_index):
        def hook(module, args, kwargs):
            x = args[0]
            attn_bias = kwargs.get(
                "attn_bias", args[1] if len(args) > 1 else None
            )
            self.attention[layer_index] = _selected_cls_self_attention(
                module, x, attn_bias, self.head_indices
            ).detach()
            if self.progress_callback is not None:
                self.progress_callback(layer_index)
        return hook

    def __enter__(self):
        blocks = self.backbone.blocks.modules_list
        num_layers = len(blocks)
        num_heads = int(blocks[0].attn.num_heads)
        if not self.layer_indices or not self.head_indices:
            raise ValueError("At least one layer and one head must be selected")
        if any(index < 0 or index >= num_layers for index in self.layer_indices):
            raise ValueError(
                f"Layer indices must be in [0, {num_layers - 1}]: "
                f"{self.layer_indices}"
            )
        if any(index < 0 or index >= num_heads for index in self.head_indices):
            raise ValueError(
                f"Head indices must be in [0, {num_heads - 1}]: "
                f"{self.head_indices}"
            )
        for layer_index in self.layer_indices:
            self.handles.append(
                blocks[layer_index].attn.register_forward_pre_hook(
                    self._make_hook(layer_index), with_kwargs=True
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def patch_scores(self, coords: torch.Tensor, patch_size: int):
        missing = set(self.layer_indices) - set(self.attention)
        if missing:
            raise RuntimeError(f"No TITAN attention captured for layers: {missing}")
        _, inverse = _patch_token_mapping(coords, patch_size)
        num_rows = coords.squeeze(0).shape[0]
        result = {}
        for layer_index in self.layer_indices:
            layer_attention = self.attention[layer_index]
            if layer_attention.shape[1] != len(self.head_indices):
                raise RuntimeError("Captured head dimension is inconsistent")
            for selected_position, head_index in enumerate(self.head_indices):
                # Index 0 is the CLS key; spatial patches start at index 1.
                patch_attention = layer_attention[0, selected_position, 1:]
                result[(layer_index, head_index)] = _expand_token_scores(
                    patch_attention.float(), inverse, num_rows
                )
        return result
