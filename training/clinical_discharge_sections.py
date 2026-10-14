"""按高覆盖主标题切分 MIMIC-IV 出院小结，供试点限制证据来源。"""

from __future__ import annotations

from dataclasses import dataclass

from training.audit_discharge_headings import HEADING, normalize_title
from training.clinical_discharge_rules_v2 import FIELDS


# 影像报告的 Impression/Findings、问题列表的 COPD 等不作为顶层分区边界。
MAJOR_HEADINGS = frozenset({
    "allergies", "chief complaint", "major surgical or invasive procedure",
    "history of present illness", "past medical history", "social history",
    "family history", "physical exam", "pertinent results",
    "brief hospital course", "medications on admission", "discharge medications",
    "discharge disposition", "discharge diagnosis", "discharge condition",
    "discharge instructions", "followup instructions", "transitional issues",
    "admission physical exam", "discharge physical exam", "discharge exam",
    "admission exam", "admission labs", "discharge labs", "imaging",
    "microbiology", "review of systems",
})

ALLOWED_HEADINGS: dict[str, frozenset[str]] = {
    "residual_dyspnea": frozenset({
        "discharge condition", "discharge physical exam", "discharge exam",
        "brief hospital course", "discharge instructions",
    }),
    "residual_cough_sputum": frozenset({
        "discharge condition", "discharge physical exam", "discharge exam",
        "brief hospital course", "discharge instructions",
    }),
    "oxygen_or_niv_at_discharge": frozenset({
        "discharge condition", "discharge physical exam", "discharge exam",
        "brief hospital course", "discharge instructions", "transitional issues",
    }),
    "mobility_at_discharge": frozenset({
        "discharge condition", "discharge physical exam", "discharge exam",
        "brief hospital course", "discharge instructions",
    }),
    "inhaler_use_or_adherence": frozenset({
        "brief hospital course", "discharge instructions", "transitional issues",
        "social history", "discharge condition",
    }),
    "social_support": frozenset({
        "social history", "brief hospital course", "discharge disposition",
        "discharge instructions", "transitional issues",
    }),
    "steroid_or_antibiotic_discharge_plan": frozenset({
        "discharge medications", "discharge instructions", "brief hospital course",
        "transitional issues",
    }),
    "unresolved_infection_or_imaging": frozenset({
        "pertinent results", "imaging", "brief hospital course",
        "transitional issues", "followup instructions", "discharge condition",
    }),
}
assert set(ALLOWED_HEADINGS) == set(FIELDS)


@dataclass(frozen=True)
class NoteSection:
    title: str
    body: str


def split_major_sections(original: str) -> list[NoteSection]:
    """内部问题名保留在父章节正文中，不让它改写主章节。"""

    sections: list[NoteSection] = []
    current_title: str | None = None
    body: list[str] = []
    for line in original.splitlines():
        match = HEADING.fullmatch(line)
        title = normalize_title(match.group(1)) if match else None
        if title in MAJOR_HEADINGS:
            if current_title is not None:
                sections.append(NoteSection(current_title, "\n".join(body)))
            current_title = title
            body = []
        elif current_title is not None:
            body.append(line)
    if current_title is not None:
        sections.append(NoteSection(current_title, "\n".join(body)))
    return sections


def field_contexts(original: str) -> tuple[str, dict[str, tuple[str, ...]]]:
    """只传可能含出院证据的章节，并保留字段级允许章节集合。"""

    sections = split_major_sections(original)
    needed = set().union(*ALLOWED_HEADINGS.values())
    relevant = [section for section in sections if section.title in needed]
    if not relevant:
        raise ValueError("no recognized discharge sections")
    context = "\n\n".join(
        f"=== {section.title.upper()} ===\n{section.body}" for section in relevant
    )
    allowed = {
        field: tuple(section.body for section in relevant
                     if section.title in headings and section.body.strip())
        for field, headings in ALLOWED_HEADINGS.items()
    }
    return context, allowed
