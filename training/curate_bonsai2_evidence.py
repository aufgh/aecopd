"""对 Bonsai 候选证据做保守的字段、时点与断行校验；原始库保持不变。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from training.clinical_text_rules import split_sections


DISCHARGE_SECTIONS = {
    "DISCHARGE CONDITION", "DISCHARGE EXAM", "DISCHARGE PHYSICAL EXAM",
    "DISCHARGE INSTRUCTIONS", "DISCHARGE DISPOSITION", "TRANSITIONAL ISSUES",
}
EARLY_SECTIONS = {"HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "PHYSICAL EXAM"}
COURSE = "BRIEF HOSPITAL COURSE"


@dataclass(frozen=True)
class CurationResult:
    quote: str | None
    reason: str


def _matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def _short_sentence(lines: list[str], line: int, sections: dict[int, str]) -> str:
    """只连接同章节、明显被物理换行打断的相邻原文行。"""

    parts = [lines[line - 1].strip()]
    section = sections.get(line)
    if line > 1 and sections.get(line - 1) == section:
        previous = lines[line - 2].strip()
        if previous and (parts[0][:1].islower() or previous.endswith((",", ";", ":"))
                         or (not previous.endswith((".", "!", "?")) and len(previous) > 35)):
            parts.insert(0, previous)
    if line < len(lines) and sections.get(line + 1) == section:
        following = lines[line].strip()
        if following and not parts[-1].endswith((".", "!", "?")):
            parts.append(following)
    return " ".join(" ".join(parts).split())[:700]


def curate_span(field: str, line: int, original: str) -> CurationResult:
    """宁可保留未知，也不把入院发现当作出院残余状态。"""

    lines = original.splitlines()
    if not 1 <= line <= len(lines):
        return CurationResult(None, "invalid_original_line")
    sections = {number: name for name, entries in split_sections(original).items()
                for number, _ in entries}
    section = sections.get(line, "UNHEADED")
    text = _short_sentence(lines, line, sections)
    if not text:
        return CurationResult(None, "blank_evidence")
    if _matches(r"\b(?:readmitted|subsequent admission|died after discharge)\b", text):
        return CurationResult(None, "possible_future_event")

    discharge_context = section in DISCHARGE_SECTIONS
    course_at_discharge = section == COURSE and _matches(
        r"\b(?:at (?:the time of )?discharge|on (?:the )?day of discharge|"
        r"upon discharge|prior to discharge|discharged|will be discharged)\b", text
    )
    if field in {"residual_dyspnea", "residual_cough_sputum"}:
        topic = (r"\b(?:dyspnea|shortness of breath|\bsob\b|breathless|breathing)\b"
                 if field == "residual_dyspnea" else r"\b(?:cough\w*|sputum|phlegm)\b")
        if not _matches(topic, text):
            return CurationResult(None, "field_topic_absent")
        if not (section in {"DISCHARGE CONDITION", "DISCHARGE EXAM", "DISCHARGE PHYSICAL EXAM",
                            "DISCHARGE INSTRUCTIONS"} or course_at_discharge):
            return CurationResult(None, "not_discharge_state")
    elif field == "oxygen_or_niv_at_discharge":
        if not _matches(r"\b(?:oxygen|O2|BiPAP|CPAP|NIV|room air|nasal cannula)\b", text):
            return CurationResult(None, "field_topic_absent")
        if not (discharge_context or section == "DISCHARGE MEDICATIONS" or course_at_discharge):
            return CurationResult(None, "not_discharge_state")
    elif field == "mobility_at_discharge":
        if not _matches(r"\b(?:walk\w*|ambulat\w*|mobil\w*|activity|rehab|"
                        r"weight.?bear\w*|WBAT|TDWB|bedbound|wheelchair|assist\w*)\b", text):
            return CurationResult(None, "field_topic_absent")
        if not (discharge_context or course_at_discharge):
            return CurationResult(None, "not_discharge_state")
    elif field == "inhaler_use_or_adherence":
        if section in {"DISCHARGE MEDICATIONS", "PERTINENT RESULTS"}:
            return CurationResult(None, "medication_list_not_adherence")
        if not _matches(r"\b(?:adheren\w*|non.?complian\w*|complian\w*|"
                        r"technique|taught|teach|educat\w*|access|not taking|"
                        r"not using|discontinued|stopped|only uses|only using|"
                        r"unable to use|incorrect\w* use)\b", text):
            return CurationResult(None, "behavior_not_documented")
    elif field == "social_support":
        if not _matches(r"\b(?:lives? with|lives? alone|living alone|assisted living|"
                        r"home with service|home services|home nursing|caregiver|"
                        r"family support|social support|son will stay|daughter will stay|"
                        r"rehab(?:ilitation)? facility)\b", text):
            return CurationResult(None, "support_not_documented")
    elif field == "steroid_or_antibiotic_discharge_plan":
        if not (section in {"DISCHARGE MEDICATIONS", "DISCHARGE INSTRUCTIONS",
                            "TRANSITIONAL ISSUES"} or course_at_discharge):
            return CurationResult(None, "not_discharge_plan")
        if not _matches(r"\b(?:predni\w*|steroid\w*|methylpred\w*|dexamethasone|"
                        r"antibiot\w*|azithro\w*|levo\w*|doxy\w*|amoxi\w*|"
                        r"cefpodoxime|ceftriaxone|cephalexin|ciprofloxacin|"
                        r"metronidazole|vancomycin|rifaximin)\b", text):
            return CurationResult(None, "drug_name_absent")
        if _matches(r"\b(?:C\.\s?diff|sternal erythema|hepatic encephalopathy)\b", text):
            return CurationResult(None, "nonrespiratory_indication")
    elif field == "unresolved_infection_or_imaging":
        if section not in {COURSE, "TRANSITIONAL ISSUES", "DISCHARGE INSTRUCTIONS",
                           "DISCHARGE CONDITION"}:
            return CurationResult(None, "raw_imaging_not_discharge_status")
        if not _matches(r"\b(?:pneumoni\w*|pulmon\w*|lung\w*|chest|consolidation|"
                        r"opacit\w*|infiltrat\w*|pleural|nodule\w*|infection)\b", text):
            return CurationResult(None, "respiratory_topic_absent")
        if not _matches(r"\b(?:persistent|persist\w*|unresolved|remaining|unchanged|"
                        r"stable|follow.?up|repeat (?:CT|CXR|imaging)|at discharge|"
                        r"upon discharge|discharged)\b", text):
            return CurationResult(None, "persistence_at_discharge_unproven")
    else:
        raise ValueError(f"unknown field: {field}")
    if section in EARLY_SECTIONS and field != "inhaler_use_or_adherence":
        return CurationResult(None, "early_section")
    return CurationResult(text, "accepted")


def curate(source: Path, database: Path, output: Path) -> dict[str, object]:
    """新建独立 SQLite，保留原始数据库不变；输出只供再次审计。"""

    if output.exists():
        raise FileExistsError(output)
    source_connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        documents = source_connection.execute("SELECT * FROM documents WHERE state='complete'").fetchall()
        rows = source_connection.execute("""
            SELECT s.hadm_id,s.field,s.quote,s.evidence_line
            FROM evidence_spans s JOIN documents d USING(hadm_id)
            WHERE d.state='complete' ORDER BY s.hadm_id,s.field
        """).fetchall()
    finally:
        source_connection.close()
    by_doc: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for hadm, field, quote, line in rows:
        by_doc[int(hadm)].append((str(field), str(quote), int(line)))
    output.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(output)
    counts: Counter[str] = Counter()
    try:
        target.executescript("""
            CREATE TABLE documents (
                hadm_id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL,
                source_sha256 TEXT NOT NULL, prompt_version TEXT NOT NULL,
                state TEXT NOT NULL, error_code TEXT, possible_future_fact INTEGER NOT NULL,
                processed_at_utc TEXT NOT NULL
            );
            CREATE TABLE evidence_spans (
                hadm_id INTEGER NOT NULL, field TEXT NOT NULL, quote TEXT NOT NULL,
                evidence_line INTEGER NOT NULL, PRIMARY KEY(hadm_id,field)
            );
            CREATE TABLE curation_decisions (
                hadm_id INTEGER NOT NULL, field TEXT NOT NULL, accepted INTEGER NOT NULL,
                reason TEXT NOT NULL, PRIMARY KEY(hadm_id,field)
            );
        """)
        target.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)", documents)
        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
            for note in csv.DictReader(handle):
                hadm = int(note["hadm_id"])
                for field, quote, line in by_doc.get(hadm, []):
                    original = note["document_text"]
                    if original.splitlines()[line - 1][:700] != quote:
                        raise ValueError("candidate quote/source mismatch")
                    result = curate_span(field, line, original)
                    counts[result.reason] += 1
                    target.execute("INSERT INTO curation_decisions VALUES (?,?,?,?)",
                                   (hadm, field, int(result.quote is not None), result.reason))
                    if result.quote is not None:
                        target.execute("INSERT INTO evidence_spans VALUES (?,?,?,?)",
                                       (hadm, field, result.quote, line))
        target.commit()
    finally:
        target.close()
    summary: dict[str, object] = {
        "input_documents_complete": len(documents), "candidate_spans": len(rows),
        "accepted_spans": counts["accepted"], "reason_counts": dict(counts),
        "requires_manual_validation": True, "encoder_ready": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(curate(args.source, args.database, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
