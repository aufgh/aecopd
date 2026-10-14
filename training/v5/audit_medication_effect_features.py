"""仅输出药理作用组住院级覆盖与暴露摘要，不输出患者级资料。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from training.v5.event_data import EventData
from training.v5.medication_effect_data import EFFECT_GROUPS, VALUES_PER_GROUP, attach_medication_effects


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(processed: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with np.load(processed / "event_arrays.npz", allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
    dummy = EventData(SimpleNamespace(hadm_ids=hadms), np.empty((0, 4)),
                      np.empty(0), np.empty(0), (),
                      {"processed_manifest": _sha(processed / "manifest.json")})
    features = attach_medication_effects(dummy, processed, mode="active").medication_by_admission
    rows = features.reshape(len(hadms), len(EFFECT_GROUPS), VALUES_PER_GROUP)
    groups = {}
    for index, name in enumerate(EFFECT_GROUPS):
        exposed = rows[:, index, 0] > 0
        groups[name] = {
            "admissions_with_accepted_administration": int(exposed.sum()),
            "fraction_of_all_admissions": float(exposed.mean()),
            "median_unique_administrations_among_exposed": (
                float(np.median(np.expm1(rows[exposed, index, 1]))) if exposed.any() else None),
            "median_hospital_day_bins_among_exposed": (
                float(np.median(np.expm1(rows[exposed, index, 2]))) if exposed.any() else None),
        }
    result = {"status": "audited_aggregate_only", "admissions": len(hadms),
              "processed_manifest_sha256": _sha(processed / "manifest.json"),
              "any_target_medication_admissions": int((rows[:, :, 0] > 0).any(axis=1).sum()),
              "groups": groups,
              "limitation": "No accepted target class is not proof that no drug was administered."}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-event-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.processed_event_dir, args.output)
    print(json.dumps({"status": result["status"], "admissions": result["admissions"]}))
