"""对候选章节做可复现的内容风险筛查；规则仅供复核，不自动删改原文。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from training.v5.section_alignment_data.prepare_section_text import PILOT_ROLES


_ADMIN_HEADER = re.compile(
    r"(?im)^\s*(?:PCP|Primary Care (?:Physician|Provider)|Contact|Phone|Fax)\s*:"
)
_DRUG_LIST_CUE = re.compile(
    r"(?i)(?:\bSig\s*:|\bPO\s+(?:DAILY|BID|TID)\b|\b\d+(?:\.\d+)?\s*(?:mg|mcg)\b)"
)
_SHORT_CONDITION = re.compile(
    r"(?is)^\s*(?:stable|good|fair|improved|guarded|satisfactory|unchanged)\.?\s*$"
)
_FUNCTION_CUE = re.compile(
    r"(?i)\b(?:activity status|ambulatory|ambulates?|mental status|"
    r"level of consciousness|alert|independent|assistance|oxygen|room air)\b"
)
_COURSE_SUBHEADING = re.compile(r"(?m)^\s*#{1,3}\s*\S")


def screen_flags(role: str, text: str, timestamp_relation: str) -> tuple[str, ...]:
    """返回待复核风险标记；它们既不是临床标签，也不是编码过滤条件。"""

    flags: list[str] = []
    if timestamp_relation == "after_discharge":
        flags.append("stored_after_discharge")
    if len(text) > 2500:
        flags.append("long_over_2500_chars")
    if role == "history_of_present_illness" and _ADMIN_HEADER.search(text[:500]):
        flags.append("administrative_header_first_500_chars")
    if role in {"admission_medications", "discharge_medications"} and _DRUG_LIST_CUE.search(text):
        flags.append("drug_list_format")
    if role == "hospital_course" and _COURSE_SUBHEADING.search(text):
        flags.append("problem_subheadings")
    if role == "discharge_condition":
        if len(text) > 500:
            flags.append("condition_over_500_chars_review_boundary")
        if _SHORT_CONDITION.fullmatch(text):
            flags.append("generic_condition_only")
        if _FUNCTION_CUE.search(text):
            flags.append("function_or_respiratory_cue")
    return tuple(flags)


def screen(aligned_dir: Path, output_dir: Path) -> dict[str, object]:
    """核对来源指纹并保存无正文计数及分层复核索引。"""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    qa = json.loads((aligned_dir / "qa_validation.json").read_text(encoding="utf-8"))
    source = aligned_dir / "note_sections.csv.gz"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if qa["status"] != "structural_checks_passed" or digest.hexdigest() != qa["output_sha256"][source.name]:
        raise ValueError("source section dataset not validated")

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    # 固定哈希顺序，短／中／长篇每类两例；只保存源键，正文仍留在私有原始清单。
    examples: dict[tuple[str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["section_role"]
            if role not in PILOT_ROLES or row["parse_status"] == "uncertain" or row["content_status"] != "informative":
                continue
            text = row["text"]
            counter = counts[role]
            counter["sections"] += 1
            for flag in screen_flags(role, text, row["timestamp_relation"]):
                counter[flag] += 1
            bucket = "short" if len(text) < 500 else "long" if len(text) > 2500 else "medium"
            score = hashlib.sha256(("sample-v1-" + row["section_id"]).encode("utf-8")).hexdigest()
            examples[(role, bucket)].append((score, row))

    output_dir.mkdir(parents=True)
    with (output_dir / "sample_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("section_role", "length_stratum", "section_id", "hadm_id", "note_id",
                         "start_char", "end_char", "timestamp_relation", "text_length"))
        for role in PILOT_ROLES:
            for bucket in ("short", "medium", "long"):
                for _, row in sorted(examples[(role, bucket)], key=lambda item: item[0])[:2]:
                    writer.writerow((role, bucket, row["section_id"], row["hadm_id"], row["note_id"],
                                     row["start_char"], row["end_char"], row["timestamp_relation"], len(row["text"])))
    result: dict[str, object] = {
        "status": "screening_flags_not_clinical_truth",
        "source_note_sections_sha256": digest.hexdigest(),
        "counts": {role: dict(counts[role]) for role in PILOT_ROLES},
        "sampling": "two SHA-256-ranked sections per role and length stratum; first 800 characters manually inspected after redaction",
        "automatic_exclusions": 0,
    }
    (output_dir / "screening.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(screen(args.aligned_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
