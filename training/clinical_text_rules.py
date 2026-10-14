"""高精度临床文本规则基线；未知不编码为阴性，不输出原文证据。"""

from __future__ import annotations

import re
from dataclasses import dataclass


FIELD_NAMES = (
    "discharge_oxygen_support",
    "new_or_escalated_home_oxygen",
    "discharge_mobility",
    "residual_wheeze",
    "discharge_steroid_taper",
    "persistent_consolidation",
    "pulmonary_edema",
)


@dataclass(frozen=True)
class RuleFinding:
    value: str
    section: str
    line: int | None


UNKNOWN = RuleFinding("unknown", "", None)

_SECTION = re.compile(
    r"(?im)^\s*(DISCHARGE CONDITION|DISCHARGE PHYSICAL EXAM|DISCHARGE EXAM|"
    r"DISCHARGE MEDICATIONS|DISCHARGE INSTRUCTIONS|DISCHARGE DISPOSITION|"
    r"TRANSITIONAL ISSUES|HISTORY OF PRESENT ILLNESS|PAST MEDICAL HISTORY|"
    r"BRIEF HOSPITAL COURSE|PHYSICAL EXAM|PERTINENT RESULTS|"
    r"SOCIAL HISTORY|IMPRESSION|FINDINGS)\s*:\s*"
)
_FUTURE_FACT = re.compile(
    r"(?i)\b(?:post[- ]discharge|after discharge|subsequent admission|"
    r"readmitted after discharge|addendum)\b"
)


def split_sections(text: str) -> dict[str, list[tuple[int, str]]]:
    """识别常见标题；保留行号，避免将入院查体误认为出院查体。"""

    sections: dict[str, list[tuple[int, str]]] = {}
    current = "UNHEADED"
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _SECTION.match(line)
        if match:
            current = match.group(1).upper()
            remainder = line[match.end():].strip()
            if remainder:
                sections.setdefault(current, []).append((line_number, remainder))
        else:
            sections.setdefault(current, []).append((line_number, line.strip()))
    return sections


def _find(
    sections: dict[str, list[tuple[int, str]]],
    names: tuple[str, ...],
    positive: re.Pattern[str],
    negative: re.Pattern[str] | None = None,
) -> RuleFinding:
    for name in names:
        for line, content in sections.get(name, []):
            if _FUTURE_FACT.search(content):
                continue
            if negative is not None and negative.search(content):
                return RuleFinding("0", name, line)
            if positive.search(content):
                return RuleFinding("1", name, line)
    return UNKNOWN


def extract_discharge(text: str) -> dict[str, RuleFinding]:
    """仅对出院锚定章节下结论；复杂自由叙述留给人工或 LLM。"""

    sections = split_sections(text)
    result = {name: UNKNOWN for name in FIELD_NAMES}
    result["discharge_oxygen_support"] = _find(
        sections,
        ("DISCHARGE CONDITION", "DISCHARGE INSTRUCTIONS", "TRANSITIONAL ISSUES"),
        re.compile(r"(?i)\b(?:home oxygen|home o2|discharged? (?:with|on) (?:oxygen|o2)|on\s*\d+(?:\.\d+)?\s*(?:l|liters?)\s*(?:nc|nasal cannula|oxygen|o2)?)\b"),
        re.compile(r"(?i)\b(?:no (?:home )?(?:oxygen|o2)|room air|off oxygen|weaned off (?:oxygen|o2))\b"),
    )
    result["new_or_escalated_home_oxygen"] = _find(
        sections,
        ("DISCHARGE CONDITION", "DISCHARGE INSTRUCTIONS", "TRANSITIONAL ISSUES"),
        re.compile(r"(?i)\b(?:new(?:ly)?|increas(?:ed|ing)|higher)\b.{0,50}\b(?:home )?(?:oxygen|o2)\b|\b(?:home )?(?:oxygen|o2)\b.{0,50}\b(?:new(?:ly)?|increas(?:ed|ing)|higher)\b"),
        re.compile(r"(?i)\b(?:no (?:home )?(?:oxygen|o2)|room air|back to baseline (?:oxygen|o2)|baseline (?:oxygen|o2) requirement)\b"),
    )
    mobility_names = ("DISCHARGE CONDITION", "DISCHARGE INSTRUCTIONS")
    for name in mobility_names:
        for line, content in sections.get(name, []):
            if _FUTURE_FACT.search(content):
                continue
            if re.search(r"(?i)\b(?:bedbound|bedridden|wheelchair bound)\b", content):
                result["discharge_mobility"] = RuleFinding("bedbound", name, line)
                break
            if re.search(r"(?i)\b(?:requires assistance|with assistance|walker|walking aid|cane)\b", content):
                result["discharge_mobility"] = RuleFinding("assisted", name, line)
                break
            if re.search(r"(?i)\b(?:ambulatory\s*[-–]\s*independent|independently ambulat\w*|walking independently)\b", content):
                result["discharge_mobility"] = RuleFinding("independent", name, line)
                break
        if result["discharge_mobility"] != UNKNOWN:
            break
    result["residual_wheeze"] = _find(
        sections,
        ("DISCHARGE EXAM", "DISCHARGE PHYSICAL EXAM", "DISCHARGE CONDITION"),
        re.compile(r"(?i)\b(?:wheez\w*|rhonchi)\b"),
        re.compile(r"(?i)\b(?:no|without|resolved)\s+(?:\w+\s+){0,2}(?:wheez\w*|rhonchi)\b|\blungs? clear\b"),
    )
    result["discharge_steroid_taper"] = _find(
        sections,
        ("DISCHARGE MEDICATIONS", "DISCHARGE INSTRUCTIONS", "TRANSITIONAL ISSUES"),
        re.compile(r"(?i)\b(?:prednisone|prednisolone|methylprednisolone|steroid)\b.{0,60}\b(?:taper|decreas\w*|wean\w*)\b|\b(?:taper|wean)\b.{0,30}\b(?:prednisone|steroid)\b"),
        re.compile(r"(?i)\b(?:no|without)\s+(?:steroid|prednisone)\s+taper\b|\b(?:steroid|prednisone)\s+(?:course|taper)\s+completed\b"),
    )
    return result


def extract_radiology(text: str) -> dict[str, RuleFinding]:
    """影像仅识别明确的阳性或阴性，未提及维持 unknown。"""

    sections = split_sections(text)
    result = {name: UNKNOWN for name in FIELD_NAMES}
    names = ("IMPRESSION", "FINDINGS", "UNHEADED")
    result["persistent_consolidation"] = _find(
        sections, names,
        re.compile(r"(?i)\b(?:persistent|persisting|unresolved)\b.{0,45}\bconsolidation\b|\bconsolidation\b.{0,45}\b(?:persists?|unresolved)\b"),
        re.compile(r"(?i)\b(?:no|without)\s+(?:\w+\s+){0,2}consolidation\b|\bconsolidation\s+(?:has )?resolved\b"),
    )
    result["pulmonary_edema"] = _find(
        sections, names,
        re.compile(r"(?i)\b(?:pulmonary|interstitial)\s+edema\b"),
        re.compile(r"(?i)\b(?:no|without|resolved)\s+(?:\w+\s+){0,2}(?:pulmonary|interstitial)\s+edema\b"),
    )
    return result


def has_possible_post_discharge_fact(text: str) -> bool:
    """仅提示人工检查，不执行删除或整篇截断。"""

    return _FUTURE_FACT.search(text) is not None
