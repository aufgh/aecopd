"""从已核验的院内实际给药构造药理作用组，不把药名或给药途径输入模型。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from training.v5.event_data import EventData


# 抗菌药仅是全身抗菌暴露代理，不能推断肺部感染适应证或治疗有效。
# 利尿剂单列共病/容量管理组，不并入 COPD 特异治疗组。
EFFECT_GROUPS = {
    "short_beta_agonist": ("saba",),
    "short_muscarinic_antagonist": ("sama",),
    "long_beta_agonist": ("laba",),
    "long_muscarinic_antagonist": ("lama",),
    "inhaled_corticosteroid": ("inhaled_steroid",),
    "systemic_corticosteroid": ("systemic_steroid",),
    "common_systemic_antibacterial": ("macrolide", "beta_lactam", "fluoroquinolone", "tetracycline"),
    "diuretic": ("loop_diuretic", "other_diuretic"),
}
# 两个高共暴露治疗组合用于联合消融；不增加输入维度或新药理类别。
EFFECT_ABLATIONS = {
    **{group: (group,) for group in EFFECT_GROUPS},
    "short_bronchodilators": ("short_beta_agonist", "short_muscarinic_antagonist"),
    "maintenance_inhalers": ("long_beta_agonist", "long_muscarinic_antagonist",
                            "inhaled_corticosteroid"),
}
CORE_ACUTE_GROUPS = ("short_beta_agonist", "short_muscarinic_antagonist",
                     "systemic_corticosteroid", "common_systemic_antibacterial")
VALUES_PER_GROUP = 6
FEATURES = ("exposed", "administrations_log", "hospital_day_bins_log", "first_day_log",
            "last_day_log", "span_days_log")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach_medication_effects(data: EventData, processed_dir: Path, *, mode: str) -> EventData:
    """在相同词表/参数量下遮蔽作用组；复方的同一来源只计一次。"""
    if mode not in ("active", "masked", "exposure_only", "core_acute",
                    *(f"without_{group}" for group in EFFECT_ABLATIONS)):
        raise ValueError("unknown medication-effect ablation")
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events_path = processed_dir / "events.csv.gz"
    if (manifest["status"] != "preprocessed_and_checked"
            or _sha(manifest_path) != data.source_hashes["processed_manifest"]
            or _sha(events_path) != manifest["output_sha256"]["events.csv.gz"]):
        raise ValueError("medication source contract mismatch")
    hadm_index = {str(hadm): i for i, hadm in enumerate(data.base.hadm_ids)}
    if len(hadm_index) != len(data.base.hadm_ids):
        raise ValueError("duplicate model admissions")
    by_class = defaultdict(list)
    for group, members in EFFECT_GROUPS.items():
        for member in members:
            by_class[member].append(group)
    # source_id 为 eMAR ID + seq；同一复方可对应多类，但同一作用组只算一次。
    administration_days: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    with gzip.open(events_path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "medication" or row["concept"] not in by_class:
                continue
            position = hadm_index.get(row["hadm_id"])
            if position is None:
                raise ValueError("accepted medication outside model cohort")
            day = float(row["elapsed_days"])
            if not math.isfinite(day) or day < 0 or not row["source_id"]:
                raise ValueError("invalid accepted administration")
            for group in by_class[row["concept"]]:
                entries = administration_days[(position, group)]
                previous = entries.setdefault(row["source_id"], day)
                if abs(previous - day) > 1e-5:
                    raise ValueError("one eMAR administration has conflicting times")
    features = np.zeros((len(hadm_index), len(EFFECT_GROUPS) * VALUES_PER_GROUP), dtype=np.float32)
    for (position, group), entries in administration_days.items():
        days = sorted(entries.values())
        first, last = days[0], days[-1]
        offset = list(EFFECT_GROUPS).index(group) * VALUES_PER_GROUP
        features[position, offset:offset + VALUES_PER_GROUP] = (
            1., math.log1p(len(days)), math.log1p(len({math.floor(day) for day in days})),
            math.log1p(first), math.log1p(last), math.log1p(last - first))
    if mode == "masked":
        features.fill(0.)
    elif mode == "exposure_only":
        features.reshape(len(hadm_index), len(EFFECT_GROUPS), VALUES_PER_GROUP)[:, :, 1:] = 0.
    elif mode == "core_acute":
        for group_index, group in enumerate(EFFECT_GROUPS):
            if group not in CORE_ACUTE_GROUPS:
                offset = group_index * VALUES_PER_GROUP
                features[:, offset:offset + VALUES_PER_GROUP] = 0.
    elif mode.startswith("without_"):
        for group in EFFECT_ABLATIONS[mode.removeprefix("without_")]:
            offset = list(EFFECT_GROUPS).index(group) * VALUES_PER_GROUP
            features[:, offset:offset + VALUES_PER_GROUP] = 0.
    return replace(data, medication_by_admission=features)
