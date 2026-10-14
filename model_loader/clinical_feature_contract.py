"""把原始结构化事件映射为受控临床特征。

该模块是原始事件表与模型装载器之间的唯一映射 seam。调用方只需要提供
一条 CSV 事件记录；药物归类、一对多复方映射、默认排除和合同校验都封装
在此模块内。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


VALID_FEATURE_LAYERS = frozenset({"patient_state", "care_process"})
VALID_VALUE_TYPES = frozenset(
    {"numeric_trajectory", "clinical_state", "binary_exposure"}
)


@dataclass(frozen=True)
class MappedClinicalFeature:
    """一条原始事件映射后的标准临床概念。"""

    canonical_concept: str
    feature_layer: str
    value_type: str
    rule_id: str


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    source_domain: str
    source_concept: str
    model_scope: str
    canonical_concept: str
    feature_layer: str
    value_type: str
    keep: bool

    def matches(self, row: Mapping[str, str]) -> bool:
        return all(
            expected == "*" or expected == actual
            for expected, actual in (
                (self.source_domain, (row.get("event_domain") or "").strip()),
                (self.source_concept, (row.get("concept_name") or "").strip()),
                (self.model_scope, (row.get("model_scope") or "").strip()),
            )
        )

    @property
    def specificity(self) -> int:
        return sum(
            value != "*"
            for value in (self.source_domain, self.source_concept, self.model_scope)
        )


class ClinicalFeatureContract:
    """加载并执行可审核的临床特征映射合同。

    未命中的事件默认排除。若多条最高特异度规则同时命中，则全部输出，
    从而支持 ICS/LABA 等复方治疗的一对多映射。
    """

    def __init__(self, rules: tuple[_Rule, ...], *, source_path: Path, sha256: str) -> None:
        self._rules = rules
        self.source_path = source_path
        self.sha256 = sha256

    @classmethod
    def from_csv(cls, path: str | Path) -> "ClinicalFeatureContract":
        source_path = Path(path)
        payload = source_path.read_bytes()
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Clinical feature contract is empty: {source_path}")

        rules: list[_Rule] = []
        seen_rule_ids: set[str] = set()
        required = {
            "rule_id",
            "source_domain",
            "source_concept",
            "model_scope",
            "canonical_concept",
            "feature_layer",
            "value_type",
            "keep",
        }
        missing_columns = required.difference(rows[0])
        if missing_columns:
            raise ValueError(
                f"Clinical feature contract missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(rows, start=2):
            rule_id = (row.get("rule_id") or "").strip()
            if not rule_id or rule_id in seen_rule_ids:
                raise ValueError(
                    f"Invalid or duplicate rule_id at {source_path}:{row_number}: {rule_id!r}"
                )
            seen_rule_ids.add(rule_id)
            keep_value = (row.get("keep") or "").strip().lower()
            if keep_value not in {"0", "1", "false", "true", "no", "yes", "n", "y"}:
                raise ValueError(
                    f"Invalid keep value at {source_path}:{row_number}: {keep_value!r}"
                )
            keep = keep_value in {"1", "true", "yes", "y"}
            feature_layer = (row.get("feature_layer") or "").strip()
            value_type = (row.get("value_type") or "").strip()
            canonical_concept = (row.get("canonical_concept") or "").strip()
            if keep:
                if feature_layer not in VALID_FEATURE_LAYERS:
                    raise ValueError(
                        f"Invalid feature_layer at {source_path}:{row_number}: {feature_layer!r}"
                    )
                if value_type not in VALID_VALUE_TYPES:
                    raise ValueError(
                        f"Invalid value_type at {source_path}:{row_number}: {value_type!r}"
                    )
                if not canonical_concept:
                    canonical_concept = "$SOURCE"
            rules.append(
                _Rule(
                    rule_id=rule_id,
                    source_domain=(row.get("source_domain") or "").strip() or "*",
                    source_concept=(row.get("source_concept") or "").strip() or "*",
                    model_scope=(row.get("model_scope") or "").strip() or "*",
                    canonical_concept=canonical_concept,
                    feature_layer=feature_layer,
                    value_type=value_type,
                    keep=keep,
                )
            )
        return cls(
            tuple(rules),
            source_path=source_path.resolve(),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def map_event(self, row: Mapping[str, str]) -> tuple[MappedClinicalFeature, ...]:
        matches = [rule for rule in self._rules if rule.matches(row)]
        if not matches:
            return ()
        maximum_specificity = max(rule.specificity for rule in matches)
        selected = [rule for rule in matches if rule.specificity == maximum_specificity]
        source_concept = (row.get("concept_name") or "").strip()
        mapped = []
        for rule in selected:
            if not rule.keep:
                continue
            canonical_concept = (
                source_concept if rule.canonical_concept == "$SOURCE" else rule.canonical_concept
            )
            mapped.append(
                MappedClinicalFeature(
                    canonical_concept=canonical_concept,
                    feature_layer=rule.feature_layer,
                    value_type=rule.value_type,
                    rule_id=rule.rule_id,
                )
            )
        return tuple(mapped)

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": "aecopd-clinical-feature-contract-1",
            "path": str(self.source_path),
            "sha256": self.sha256,
            "rule_count": len(self._rules),
            "default_policy": "exclude_unmatched",
        }
