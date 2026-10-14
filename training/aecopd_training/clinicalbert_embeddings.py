"""Bio_ClinicalBERT 分块和编码器的纯函数，便于小样本验证。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TokenWindow:
    source_row: int
    token_ids: tuple[int, ...]
    aggregation_weight: int


def split_token_windows(
    token_ids: Sequence[int],
    source_row: int,
    *,
    max_content_tokens: int = 510,
    overlap_tokens: int = 64,
) -> list[TokenWindow]:
    """完整覆盖长章节；重叠仅提供上下文，不重复计算聚合权重。"""

    if max_content_tokens <= 0 or not 0 <= overlap_tokens < max_content_tokens:
        raise ValueError("invalid content length or overlap")
    values = tuple(int(token) for token in token_ids)
    if not values:
        return [TokenWindow(source_row, (), 1)]
    result: list[TokenWindow] = []
    start = 0
    while start < len(values):
        end = min(start + max_content_tokens, len(values))
        new_tokens = end - start if not result else end - start - overlap_tokens
        result.append(TokenWindow(source_row, values[start:end], max(1, new_tokens)))
        if end == len(values):
            break
        start = end - overlap_tokens
    return result


def pack_token_windows(
    windows: Sequence[TokenWindow],
    *,
    token_budget: int,
    max_batch_size: int,
) -> list[list[TokenWindow]]:
    """按填充后的长度限额组批，避免长章节触发显存溢出。"""

    if token_budget < 3 or max_batch_size <= 0:
        raise ValueError("token budget and batch size must be positive")
    batches: list[list[TokenWindow]] = []
    current: list[TokenWindow] = []
    longest = 0
    for window in sorted(windows, key=lambda item: len(item.token_ids)):
        length = len(window.token_ids) + 2
        if length > token_budget:
            raise ValueError("one window exceeds the token budget")
        proposed = max(longest, length)
        if current and (len(current) >= max_batch_size or proposed * (len(current) + 1) > token_budget):
            batches.append(current)
            current = []
            longest = 0
        current.append(window)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def masked_content_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """忽略 padding、[CLS] 与 [SEP]，得到窗口内容均值。"""

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden state and attention mask are misaligned")
    content_mask = attention_mask.to(torch.bool).clone()
    lengths = content_mask.sum(dim=1)
    if torch.any(lengths < 2):
        raise ValueError("every window needs CLS and SEP")
    content_mask[:, 0] = False
    content_mask.scatter_(1, (lengths - 1).unsqueeze(1), False)
    weights = content_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
