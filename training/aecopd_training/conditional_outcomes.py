"""从队列表构造条件区间和 365 天后数据库尾部标签。

类别编码固定为 0=无事件/未观察到，1=AECOPD，2=死亡。开放尾部是
数据库定义结局，不等价于拥有患者终身完整随访。
"""

from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor


INTERVAL_NAMES = ("days_0_30", "days_31_90", "days_91_365")
INTERVAL_UPPER_DAYS = (30.0, 90.0, 365.0)
OUTCOME_NAMES = ("no_event", "aecopd", "death")
TAIL_OUTCOME_NAMES = ("none_observed", "aecopd", "death")


@dataclass(frozen=True)
class ConditionalOutcome:
    """一次索引住院的三个区间标签、适用掩码和尾部标签。"""

    interval_labels: tuple[int, int, int]
    interval_mask: tuple[bool, bool, bool]
    tail_label: int
    tail_mask: bool


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


class ConditionalOutcomeStore:
    """Hadm-addressable conditional interval and database-tail targets."""

    def __init__(self, outcomes: dict[int, ConditionalOutcome]) -> None:
        if not outcomes:
            raise ValueError("conditional outcome store cannot be empty")
        self._outcomes = dict(outcomes)

    @classmethod
    def from_cohort(cls, path: str | Path) -> "ConditionalOutcomeStore":
        """按固定区间边界从 cohort gzip CSV 建立 hadm_id 标签索引。"""
        outcomes: dict[int, ConditionalOutcome] = {}
        with gzip.open(Path(path), "rt", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                hadm_id = int(row["hadm_id"])
                if hadm_id in outcomes:
                    raise ValueError(f"duplicate hadm_id in cohort: {hadm_id}")
                event_type = int(row["event_type"])
                time_days = float(row["time_days"])
                if event_type not in {0, 1, 2} or not math.isfinite(time_days):
                    raise ValueError(f"invalid outcome for hadm_id={hadm_id}")

                labels = [0, 0, 0]
                mask = [True, True, True]
                if event_type > 0:
                    # 边界归属为 <=30、(30,90]、(90,365]；事件后的所有
                    # 任务退出风险集，避免同一事件重复成为后续阳性。
                    interval = next(
                        (index for index, upper in enumerate(INTERVAL_UPPER_DAYS) if time_days <= upper),
                        None,
                    )
                    if interval is None:
                        raise ValueError(f"event beyond 365 days is marked observed: hadm_id={hadm_id}")
                    labels[interval] = event_type
                    for later in range(interval + 1, len(mask)):
                        mask[later] = False
                    tail_label = 0
                    tail_mask = False
                else:
                    if time_days != 365.0:
                        raise ValueError(f"administrative censoring must be at day 365: hadm_id={hadm_id}")
                    aecopd = _optional_float(row.get("next_observed_aecopd_days"))
                    death = _optional_float(row.get("observed_death_days"))
                    candidates: list[tuple[float, int]] = []
                    if aecopd is not None and aecopd > 365.0:
                        candidates.append((aecopd, 1))
                    if death is not None and death > 365.0:
                        candidates.append((death, 2))
                    # tuple 先比较时间，再比较类别；完全同一时刻时
                    # AECOPD(1) 稳定优先于死亡(2)，与 ADR 0006 一致。
                    tail_label = min(candidates, default=(math.inf, 0))[1]
                    tail_mask = True
                outcomes[hadm_id] = ConditionalOutcome(
                    interval_labels=tuple(labels),  # type: ignore[arg-type]
                    interval_mask=tuple(mask),  # type: ignore[arg-type]
                    tail_label=tail_label,
                    tail_mask=tail_mask,
                )
        return cls(outcomes)

    def __len__(self) -> int:
        return len(self._outcomes)

    def __getitem__(self, hadm_id: int) -> ConditionalOutcome:
        try:
            return self._outcomes[int(hadm_id)]
        except KeyError as exc:
            raise KeyError(f"missing conditional outcome for hadm_id={hadm_id}") from exc

    def batch(
        self,
        hadm_ids: Iterable[int],
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """按住院顺序生成可直接加入模型 batch 的标签张量。"""
        values = [self[hadm_id] for hadm_id in hadm_ids]
        if not values:
            raise ValueError("outcome batch cannot be empty")
        return {
            "conditional_interval_labels": torch.tensor(
                [value.interval_labels for value in values], dtype=torch.long, device=device
            ),
            "conditional_interval_mask": torch.tensor(
                [value.interval_mask for value in values], dtype=torch.bool, device=device
            ),
            "conditional_tail_labels": torch.tensor(
                [value.tail_label for value in values], dtype=torch.long, device=device
            ),
            "conditional_tail_mask": torch.tensor(
                [value.tail_mask for value in values], dtype=torch.bool, device=device
            ),
        }
