"""Shared exact layer signals for streaming visual-token memory.

The attention helper scores *keys* by attention received from an observation
query window. It does not substitute a retired token's outgoing attention
mass. Queries and keys must already share their intended positional frame.
"""
from __future__ import annotations
import math
from typing import Callable, Optional, Sequence
import torch

def layer_evolution_scores(previous: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Per-token FP32 cosine change, with identical clamp semantics."""
    if previous.shape != hidden.shape:
        raise ValueError('layer evolution requires aligned previous/hidden shapes')
    return (1.0 - torch.nn.functional.cosine_similarity(previous.float(), hidden.float(), dim=-1)).clamp(0.0, 1.0)

def incoming_attention_scores(q: torch.Tensor, k: torch.Tensor, query_positions, key_positions, *, causal: bool=True, query_block_size: int=8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal incoming-key importance [T] and a zero compatibility scalar. Q:[B,Hq,R,D], K:[B,Hkv,T,D]. Grouped matmul avoids repeated K copies."""
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError('q/k must be [B,H,R,D] / [B,H,T,D]')
    batch, q_heads, query_count, dimension = q.shape
    key_batch, kv_heads, key_count, key_dimension = k.shape
    if batch < 1 or q_heads < 1 or kv_heads < 1 or (dimension < 1):
        raise ValueError('batch, head counts and head dimension must be positive')
    if batch != key_batch or dimension != key_dimension:
        raise ValueError('q/k batch and head dimensions must match')
    if q_heads % kv_heads:
        raise ValueError('query head count must be divisible by KV head count')
    if q.device != k.device:
        raise ValueError('q/k must share a device')
    if isinstance(query_block_size, bool) or not isinstance(query_block_size, int) or query_block_size < 1:
        raise ValueError('query_block_size must be a positive integer')
    q_positions = torch.as_tensor(query_positions, device=q.device)
    k_positions = torch.as_tensor(key_positions, device=q.device)
    if q_positions.ndim != 1 or q_positions.numel() != query_count:
        raise ValueError('query_positions must have one entry per query')
    if k_positions.ndim != 1 or k_positions.numel() != key_count:
        raise ValueError('key_positions must have one entry per key')
    score_sum = torch.zeros(key_count, dtype=torch.float32, device=q.device)
    summary_sum = torch.zeros((), dtype=torch.float32, device=q.device)
    if query_count == 0 or key_count == 0:
        return (score_sum, summary_sum)
    groups = q_heads // kv_heads
    transposed_keys = k.float().transpose(-2, -1).unsqueeze(2)
    scale = 1.0 / math.sqrt(dimension)
    for start in range(0, query_count, query_block_size):
        end = min(query_count, start + query_block_size)
        qb = q[:, :, start:end].float()
        grouped_q = qb.reshape(batch, kv_heads, groups, end - start, dimension)
        raw_logits = (torch.matmul(grouped_q, transposed_keys) * scale).reshape(batch, q_heads, end - start, key_count)
        if causal:
            visible = k_positions[None, :] <= q_positions[start:end, None]
            raw_logits.masked_fill_(~visible[None, None], -torch.inf)
        parts = [raw_logits]
        logits = torch.cat(parts, dim=-1) if len(parts) > 1 else raw_logits
        if logits.shape[-1] == 0:
            continue
        valid = torch.isfinite(logits).any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~valid, 0.0)
        probabilities = safe_logits.softmax(dim=-1) * valid
        score_sum += probabilities[..., :key_count].sum(dim=(0, 1, 2))
        summary_sum += probabilities[..., key_count:].sum()
    denominator = batch * q_heads * query_count
    return (score_sum / denominator, summary_sum / denominator)
