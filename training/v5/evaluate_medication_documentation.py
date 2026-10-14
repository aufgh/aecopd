"""在有/无出院时可见 eMAR 文书的住院层内审查药理组对照。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from training.v5.compare_screened_group_ablations import _predictions, _summary_for_cause


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def evaluate(root: Path, raw: Path, output: Path, *, seeds: tuple[int, ...]) -> dict:
    if output.exists():
        raise FileExistsError(output)
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    emar_file = raw / "raw/emar.csv.gz"
    if raw_manifest["status"] != "validated" or _sha(emar_file) != raw_manifest["sources"]["raw/emar"]["sha256"]:
        raise ValueError("unvalidated eMAR ledger")
    admission_times = {}
    for row in _rows(raw / "cohort/admissions.csv.gz"):
        admission_times[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                            datetime.fromisoformat(row["dischtime"]))
    documented = set()
    for row in _rows(emar_file):
        times = admission_times.get(row["hadm_id"])
        if times is None or row["charttime"] in ("", "\\N") or row["storetime"] in ("", "\\N"):
            continue
        chart, stored = datetime.fromisoformat(row["charttime"]), datetime.fromisoformat(row["storetime"])
        if times[0] <= chart <= stored <= times[1]:
            documented.add(int(row["hadm_id"]))
    result = {"status": "completed", "source_sha256": _sha(emar_file),
              "definition": "at least one eMAR record with admission <= charttime <= storetime <= discharge; not proof of complete medication capture",
              "seeds": {}}
    for seed in seeds:
        before = root / f"masked_seed{seed}"
        after = root / f"active_seed{seed}"
        reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in (before, after)]
        if any(report["status"] != "training_completed" or report["event_encoder"] != "abnormal_medication"
               for report in reports) or reports[0]["medication_effect_mode"] != "masked" or reports[1]["medication_effect_mode"] != "active":
            raise ValueError("incorrect paired reports")
        result["seeds"][str(seed)] = {}
        for split in ("validation", "historical_test"):
            old, new = _predictions(before, split), _predictions(after, split)
            for key in ("subject_ids", "hadm_ids", "times", "events"):
                if not np.array_equal(old[key], new[key]):
                    raise ValueError("different paired prediction rows")
            available = np.isin(old["hadm_ids"], list(documented))
            result["seeds"][str(seed)][split] = {}
            for subgroup, selector in (("any_emar_record", available), ("no_emar_record", ~available)):
                if selector.sum() < 100:
                    raise ValueError("eMAR subgroup too small")
                subgroup_result = {"admissions": int(selector.sum()), "patients": int(np.unique(old["subject_ids"][selector]).size)}
                for cause, name in ((1, "readmission"), (2, "death")):
                    baseline = _summary_for_cause(old["times"][selector], old["events"][selector],
                                                  old["competing_cif"][selector], cause)
                    candidate = _summary_for_cause(new["times"][selector], new["events"][selector],
                                                   new["competing_cif"][selector], cause)
                    subgroup_result[name] = {
                        "masked": {key: baseline[key] for key in ("mean_auc_7_365", "integrated_brier_7_365")},
                        "active": {key: candidate[key] for key in ("mean_auc_7_365", "integrated_brier_7_365")},
                        "delta_active_minus_masked": {key: candidate[key] - baseline[key]
                                                       for key in ("mean_auc_7_365", "integrated_brier_7_365")}}
                result["seeds"][str(seed)][split][subgroup] = subgroup_result
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--raw-ledger-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    args = parser.parse_args()
    report = evaluate(args.run_root, args.raw_ledger_dir, args.output, seeds=tuple(args.seeds))
    print(json.dumps({"status": report["status"], "seeds": tuple(report["seeds"])}, ensure_ascii=False))
