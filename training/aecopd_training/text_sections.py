"""Deterministic section parsing for selected clinical documents.

This module deliberately treats note-internal dates as ordinary text.  Temporal
alignment remains defined by the document-level charttime/storetime fields in
the source dataset.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSection:
    """One non-empty clinical text section in source order."""

    section_role: str
    section_order: int
    section_instance: int
    text: str


@dataclass(frozen=True)
class ParsedClinicalDocument:
    """A parsed document with a coarse modality and ordered sections."""

    modality: str
    sections: tuple[TextSection, ...]


_DISCHARGE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chief_complaint", ("chief complaint",)),
    ("history_of_present_illness", ("history of present illness", "hpi")),
    ("past_medical_history", ("past medical history",)),
    ("pertinent_results", ("pertinent results", "major results", "important results")),
    ("hospital_course", ("hospital course", "brief hospital course")),
    ("discharge_diagnosis", ("discharge diagnosis", "discharge diagnoses")),
    ("discharge_medications", ("discharge medication", "discharge medications")),
    ("discharge_condition", ("discharge condition",)),
    ("follow_up", ("follow-up", "follow up", "followup", "followup instruction", "followup instructions")),
)

_RADIOLOGY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("examination", ("examination", "exam")),
    ("indication", ("indication", "reason for exam")),
    ("comparison", ("comparison",)),
    ("technique", ("technique",)),
    ("findings", ("finding", "findings")),
    ("impression", ("impression",)),
)

_DISCHARGE_OTHER_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "other",
        (
            "admission date",
            "discharge date",
            "date of birth",
            "birth date",
            "sex",
            "service",
            "allergy",
            "allergies",
            "attending",
            "major surgical or invasive procedure",
            "major procedures",
            "procedure",
            "procedures",
            "past surgical history",
            "social history",
            "family history",
            "physical exam",
            "medications on admission",
            "admission medications",
            "home medications",
            "discharge disposition",
            "disposition",
            "discharge instructions",
            "instructions",
            "code status",
            "consults",
            "consultations",
            "operations",
        ),
    ),
)

_RADIOLOGY_OTHER_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "other",
        (
            "history",
            "clinical history",
            "clinical information",
            "reason",
            "recommendation",
            "recommendations",
            "conclusion",
            "addendum",
            "notification",
            "wet read",
            "preliminary report",
            "final report",
            "procedure",
            "observations",
            "measurements",
        ),
    ),
)


def _compile_header_patterns(
    aliases_by_role: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for role, aliases in aliases_by_role:
        alias_expression = "|".join(
            re.escape(alias).replace(r"\ ", r"[ \t]+") for alias in aliases
        )
        patterns.append(
            (
                role,
                re.compile(
                    rf"^[ \t]*(?:\[(?:{alias_expression})\]|(?:{alias_expression}))"
                    rf"(?:[ \t]*:[ \t]*(?P<inline>.*)|[ \t]*:?[ \t]*)$",
                    flags=re.IGNORECASE,
                ),
            )
        )
    return tuple(patterns)


_DISCHARGE_HEADERS = _compile_header_patterns(_DISCHARGE_ALIASES)
_RADIOLOGY_HEADERS = _compile_header_patterns(_RADIOLOGY_ALIASES)
_DISCHARGE_OTHER_HEADERS = _compile_header_patterns(_DISCHARGE_OTHER_ALIASES)
_RADIOLOGY_OTHER_HEADERS = _compile_header_patterns(_RADIOLOGY_OTHER_ALIASES)

_MODALITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ct", re.compile(r"\b(?:ct|cta|computed tomography)\b", re.IGNORECASE)),
    ("mri", re.compile(r"\b(?:mri|mr imaging|magnetic resonance)\b", re.IGNORECASE)),
    ("ultrasound", re.compile(r"\b(?:ultrasound|sonograph\w*|ultrasonograph\w*)\b", re.IGNORECASE)),
    ("nuclear_medicine", re.compile(r"\b(?:nuclear medicine|pet[ /-]?ct|scintigraph\w*)\b", re.IGNORECASE)),
    ("xray", re.compile(r"\b(?:x[ -]?ray|radiograph\w*|portable ap|chest ap|chest pa)\b", re.IGNORECASE)),
)


def _document_family(document_role: str) -> str:
    if document_role in {"current_discharge", "prior_discharge"}:
        return "discharge"
    if document_role in {"current_radiology", "prior_radiology"}:
        return "radiology"
    if document_role in {"current_clinical_note", "prior_clinical_note"}:
        return "clinical_note"
    raise ValueError(f"Unsupported document role: {document_role!r}")


def _detect_modality(family: str, note_type: str, text: str) -> str:
    if family == "discharge":
        return "discharge_summary"
    if family == "clinical_note":
        return "clinical_note"
    search_text = f"{note_type}\n{text[:2000]}"
    for modality, pattern in _MODALITY_PATTERNS:
        if pattern.search(search_text):
            return modality
    return "other"


def _match_header(
    line: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, str] | None:
    for role, pattern in patterns:
        match = pattern.fullmatch(line)
        if match is not None:
            return role, (match.groupdict().get("inline") or "").strip()
    return None


def split_clinical_document(
    document_role: str,
    note_type: str,
    text: str,
) -> ParsedClinicalDocument:
    """Split one selected document into non-empty canonical sections.

    Section boundaries are recognized only at full physical lines.  This avoids
    treating words such as "findings" in narrative prose as headings.  Unknown
    preamble or header-free text is retained as ``other``.
    """

    family = _document_family(document_role)
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    modality = _detect_modality(family, note_type, normalized_text)
    if not normalized_text:
        return ParsedClinicalDocument(modality=modality, sections=())
    if family == "clinical_note":
        return ParsedClinicalDocument(
            modality=modality,
            sections=(TextSection("other", 0, 1, normalized_text),),
        )

    patterns = _DISCHARGE_HEADERS if family == "discharge" else _RADIOLOGY_HEADERS
    other_patterns = _DISCHARGE_OTHER_HEADERS if family == "discharge" else _RADIOLOGY_OTHER_HEADERS
    raw_sections: list[tuple[str, str]] = []
    active_role = "other"
    active_lines: list[str] = []

    def flush() -> None:
        section_text = "\n".join(active_lines).strip()
        if section_text:
            raw_sections.append((active_role, section_text))

    for line in normalized_text.split("\n"):
        header = _match_header(line, patterns)
        if header is None and _match_header(line, other_patterns) is not None:
            header = ("other", line.strip())
        if header is None:
            active_lines.append(line)
            continue
        flush()
        active_role, inline_text = header
        active_lines = [inline_text] if inline_text else []
    flush()

    instances: Counter[str] = Counter()
    sections: list[TextSection] = []
    for section_order, (section_role, section_text) in enumerate(raw_sections):
        instances[section_role] += 1
        sections.append(
            TextSection(
                section_role=section_role,
                section_order=section_order,
                section_instance=instances[section_role],
                text=section_text,
            )
        )
    return ParsedClinicalDocument(modality=modality, sections=tuple(sections))


def build_section_embedding_text(
    document_role: str,
    modality: str,
    section: TextSection,
    note_type: str = "",
) -> str:
    """Build the exact Qwen input, including semantic context for deduplication."""

    _document_family(document_role)
    note_context = f"Note type: {note_type.strip()}\n" if _document_family(document_role) == "clinical_note" else ""
    return (
        f"Document role: {document_role}\n"
        f"Modality: {modality}\n"
        f"{note_context}"
        f"Section: {section.section_role}\n"
        f"Section instance: {section.section_instance}\n\n"
        f"{section.text}"
    )
