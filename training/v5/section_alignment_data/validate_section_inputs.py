"""独立校验章节级输入的角色、键、掩码及旧检验／给药摘要一致性。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from training.v5.clinical_concepts import LABS, medication_classes
from training.v5.section_alignment_data.build_section_inputs import SECTION_ROLES


def _rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _days(after: str, before: str) -> Decimal:
    seconds = (datetime.fromisoformat(after) - datetime.fromisoformat(before)).total_seconds()
    return Decimal(str(seconds)) / Decimal(86400)


def validate(
    aligned_dir: Path, inputs_dir: Path, old_dir: Path,
) -> dict[str, object]:
    schema = json.loads((inputs_dir / "schema.json").read_text(encoding="utf-8"))
    expected = json.loads((inputs_dir / "qa_summary.json").read_text(encoding="utf-8"))
    if tuple(schema["roles"]) != SECTION_ROLES or not schema["no_outcome_labels_in_outputs"]:
        raise ValueError("section input schema drift")
    if _sha256(aligned_dir / "qa_validation.json") != schema["source_sha256"]["aligned_qa_validation"]:
        raise ValueError("validated source changed")
    if _sha256(old_dir / "base.csv") != schema["source_sha256"]["v5_base"]:
        raise ValueError("base source changed")

    cohort = {row["hadm_id"]: row for row in _rows(aligned_dir / "cohort_manifest.csv.gz")}
    base = {row["hadm_id"]: row for row in _rows(old_dir / "base.csv")}
    note_sections = {row["section_id"]: (row["hadm_id"], row["section_role"], row["content_status"])
                     for row in _rows(aligned_dir / "note_sections.csv.gz")}
    note_hadms = {hadm for hadm, _, _ in note_sections.values()}
    slot_seen: set[tuple[str, str]] = set()
    text_counts: Counter[tuple[str, str]] = Counter()
    slot_count = 0
    for row in _rows(inputs_dir / "section_slots.csv.gz"):
        hadm, role = row["hadm_id"], row["section_role"]
        key = hadm, role
        if hadm not in cohort or row["subject_id"] != cohort[hadm]["subject_id"]:
            raise ValueError("slot outside cohort")
        if role not in SECTION_ROLES or key in slot_seen:
            raise ValueError("unknown or repeated section slot")
        slot_seen.add(key)
        ids = json.loads(row["text_section_ids"])
        if len(ids) != len(set(ids)):
            raise ValueError("repeated text section ID in slot")
        for sid in ids:
            if note_sections.get(sid, (None, None, None))[:2] != key:
                raise ValueError("slot references unrelated text section")
            text_counts[key] += int(note_sections[sid][2] == "informative")
        if int(row["text_nonblank_count"]) != text_counts[key]:
            raise ValueError("text count mismatch")
        if int(row["has_discharge_note"]) != int(hadm in note_hadms):
            raise ValueError("note presence mismatch")
        slot_count += 1
    if slot_count != len(cohort) * len(SECTION_ROLES):
        raise ValueError("slot count mismatch")

    old_labs = {(row["hadm_id"], LABS[int(row["itemid"])][0]): row
                for row in _rows(old_dir / "labs.csv")}
    old_medications: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _rows(old_dir / "medications.csv"):
        for category in medication_classes(row["medication"]):
            key = row["hadm_id"], category
            if key in old_medications:
                first, last = old_medications[key]
                old_medications[key] = (min(first, row["first_charttime"]),
                                        max(last, row["last_charttime"]))
            else:
                old_medications[key] = row["first_charttime"], row["last_charttime"]

    token_seen: set[tuple[str, str, str, str]] = set()
    by_slot: Counter[tuple[str, str]] = Counter()
    types: Counter[str] = Counter()
    labs_seen = 0
    meds_seen = 0
    first_by_patient: dict[str, tuple[str, str]] = {}
    for hadm, row in cohort.items():
        subject = row["subject_id"]
        candidate = row["admittime"], hadm
        if subject not in first_by_patient or candidate < first_by_patient[subject]:
            first_by_patient[subject] = candidate
    for row in _rows(inputs_dir / "structured_tokens.csv.gz"):
        hadm, role, kind, concept = (row[name] for name in
                                     ("hadm_id", "section_role", "token_type", "concept"))
        key = hadm, role, kind, concept
        if key in token_seen or (hadm, role) not in slot_seen:
            raise ValueError("duplicate token or unknown slot")
        token_seen.add(key)
        if row["subject_id"] != cohort[hadm]["subject_id"]:
            raise ValueError("token subject mismatch")
        if concept in {"event_type", "time_days"}:
            raise ValueError("outcome leakage")
        values = [float(row[f"value{index}"]) for index in range(5)]
        masks = [int(row[f"mask{index}"]) for index in range(5)]
        if not all(math.isfinite(value) for value in values) or not set(masks) <= {0, 1}:
            raise ValueError("invalid value or mask")
        if any(not mask and value != 0 for value, mask in zip(values, masks)):
            raise ValueError("nonzero masked value")
        by_slot[(hadm, role)] += 1
        types[kind] += 1
        if kind == "laboratory":
            old = old_labs.pop((hadm, concept), None)
            if old is None or role != "pertinent_results":
                raise ValueError("unmatched laboratory token")
            old_values = [Decimal(old[name]) for name in
                          ("first_value", "last_value", "min_value", "max_value")]
            if any(abs(Decimal(str(new)) - original) > Decimal("0.000001")
                   for new, original in zip(values[:4], old_values)):
                raise ValueError("laboratory summary mismatch")
            if masks[4] != int(int(old["measurement_count"]) >= 2):
                raise ValueError("laboratory trend mask mismatch")
            if masks[4] and abs(Decimal(str(values[4])) - (old_values[1] - old_values[0])) > Decimal("0.000001"):
                raise ValueError("laboratory delta mismatch")
            labs_seen += 1
        elif kind == "administered_medication":
            times = old_medications.pop((hadm, concept), None)
            if times is None or role != "hospital_course":
                raise ValueError("unmatched medication token")
            admission = base[hadm]["admittime"]
            expected_days = (Decimal(1), _days(times[0], admission),
                             _days(times[1], admission), _days(times[1], times[0]))
            if any(abs(Decimal(str(new)) - target) > Decimal("0.000001")
                   for new, target in zip(values[:4], expected_days)):
                raise ValueError("medication span mismatch")
            meds_seen += 1
        elif kind == "chronic_comorbidity":
            if role != "past_medical_history":
                raise ValueError("chronic comorbidity misplaced")
            if values[0] == 0 and values[1] == 0:
                raise ValueError("uncoded chronic category became a token")
            subject = cohort[hadm]["subject_id"]
            if (cohort[hadm]["admittime"], hadm) == first_by_patient[subject] and values[1] != 0:
                raise ValueError("first eligible admission has fabricated prior history")
        elif kind == "acute_diagnosis":
            if role != "discharge_diagnosis" or values[0] != 1:
                raise ValueError("uncoded or misplaced acute diagnosis")
        elif kind == "disposition" and role != "discharge_disposition":
            raise ValueError("disposition misplaced")
    if old_labs or old_medications:
        raise ValueError("missing lab or medication tokens")
    for row in _rows(inputs_dir / "section_slots.csv.gz"):
        if int(row["structured_token_count"]) != by_slot[(row["hadm_id"], row["section_role"])]:
            raise ValueError("slot structured-token count mismatch")
    if len(token_seen) != expected["structured_tokens"] or dict(types) != expected["token_types"]:
        raise ValueError("token count drift")
    result: dict[str, object] = {
        "status": "section_inputs_validated", "admissions": len(cohort),
        "slots": slot_count, "tokens": len(token_seen),
        "laboratory_tokens_reconciled": labs_seen,
        "medication_tokens_reconciled": meds_seen,
        "text_sections_referenced": sum(len(json.loads(row["text_section_ids"]))
                                        for row in _rows(inputs_dir / "section_slots.csv.gz")),
        "output_sha256": {name: _sha256(inputs_dir / name) for name in
                          ("structured_tokens.csv.gz", "section_slots.csv.gz", "schema.json")},
        "clinical_content_validated": False,
    }
    (inputs_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.aligned_dir, args.inputs_dir, args.old_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
