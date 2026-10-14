"""保留原文偏移的出院小结主章节切分。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedSection:
    """同一小结中的一段原文；偏移为 Python 字符索引，右端不包含。"""

    ordinal: int
    role: str
    heading: str
    start_char: int
    end_char: int
    text: str
    parse_status: str
    content_status: str


_ROLES: dict[str, str] = {
    "allergies": "allergies",
    "chief complaint": "chief_complaint",
    "major surgical or invasive procedure": "major_procedure",
    "history of present illness": "history_of_present_illness",
    "hpi": "history_of_present_illness",
    "past medical history": "past_medical_history",
    "social history": "social_history",
    "family history": "family_history",
    "physical exam": "physical_exam",
    "pertinent results": "pertinent_results",
    "brief hospital course": "hospital_course",
    "hospital course": "hospital_course",
    "medications on admission": "admission_medications",
    "discharge medications": "discharge_medications",
    "discharge disposition": "discharge_disposition",
    "discharge diagnosis": "discharge_diagnosis",
    "discharge diagnoses": "discharge_diagnosis",
    "discharge condition": "discharge_condition",
    "discharge instructions": "discharge_instructions",
    "followup instructions": "followup_instructions",
    "follow-up instructions": "followup_instructions",
    "transitional issues": "transitional_issues",
    "admission physical exam": "admission_physical_exam",
    "admission exam": "admission_physical_exam",
    "discharge physical exam": "discharge_physical_exam",
    "discharge exam": "discharge_physical_exam",
    "admission labs": "admission_labs",
    "discharge labs": "discharge_labs",
    "imaging": "imaging",
    "microbiology": "microbiology",
    "review of systems": "review_of_systems",
    "attending": "metadata",
    "service": "metadata",
}

_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z /&()'\-]{1,64})\s*:\s*(?P<inline>.*)$")
_NON_PLACEHOLDER = re.compile(r"[A-Za-z0-9]")


def _normalize_heading(value: str) -> str:
    return " ".join(value.lower().split())


def _header_role(heading: str, previous_role: str | None) -> tuple[str, str] | None:
    role = _ROLES.get(heading)
    if role is not None:
        return role, "recognized"
    if heading in {"diagnosis", "primary diagnosis", "secondary diagnosis", "primary diagnoses",
                   "secondary diagnoses", "primary", "secondary"}:
        if previous_role in {"discharge_disposition", "discharge_diagnosis"}:
            if previous_role == "discharge_diagnosis":
                return None  # 诊断节内部的分层标签，不另开章节。
            return "discharge_diagnosis", "contextual_alias"
        return None
    if heading == "history" and previous_role == "social_history":
        return "unclassified", "uncertain"
    return None


def parse_sections(original: str) -> tuple[ParsedSection, ...]:
    """切分主章节；未知标题保留在原文，歧义分界不伪装成社会史。"""

    sections: list[ParsedSection] = []
    current: tuple[str, str, str, int] | None = None

    def flush(boundary: int) -> None:
        if current is None:
            return
        role, heading, parse_status, start = current
        raw = original[start:boundary]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start_char = start + left
        end_char = max(start_char, start + right)
        body = original[start_char:end_char]
        sections.append(ParsedSection(
            ordinal=len(sections), role=role, heading=heading,
            start_char=start_char, end_char=end_char, text=body,
            parse_status=parse_status,
            content_status="informative" if _NON_PLACEHOLDER.search(body) else "placeholder",
        ))

    offset = 0
    for line in original.splitlines(keepends=True):
        match = _HEADER.fullmatch(line.rstrip("\r\n"))
        if match is not None:
            heading = _normalize_heading(match.group(1))
            result = _header_role(heading, current[0] if current else None)
            if result is not None:
                flush(offset)
                role, parse_status = result
                inline = match.group("inline")
                start = offset + match.start("inline") if inline.strip() else offset + len(line)
                current = (role, heading, parse_status, start)
        offset += len(line)
    flush(len(original))
    return tuple(sections)
