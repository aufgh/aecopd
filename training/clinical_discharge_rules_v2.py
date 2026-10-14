"""MIMIC-IV 出院小结的全文规则候选抽取；候选不等于可训练的临床事实。"""

from __future__ import annotations

import re
from dataclasses import dataclass


FIELDS: tuple[str, ...] = (
    "residual_dyspnea", "residual_cough_sputum", "oxygen_or_niv_at_discharge",
    "mobility_at_discharge", "inhaler_use_or_adherence", "social_support",
    "steroid_or_antibiotic_discharge_plan", "unresolved_infection_or_imaging",
)
RULE_VERSION = "fulltext_rules_v4_corpus_expanded"

# 从 4,150 篇本地小结的标题频次建立；内层 Activity Status 等不切断顶层章节。
SECTION_HEADERS = (
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY",
    "FAMILY HISTORY", "SOCIAL HISTORY", "MEDICATIONS ON ADMISSION", "ALLERGIES",
    "PHYSICAL EXAM", "PERTINENT RESULTS", "MAJOR SURGICAL OR INVASIVE PROCEDURE",
    "BRIEF HOSPITAL COURSE", "DISCHARGE CONDITION", "DISCHARGE PHYSICAL EXAM",
    "DISCHARGE EXAM", "DISCHARGE DIAGNOSIS", "DISCHARGE MEDICATIONS",
    "DISCHARGE INSTRUCTIONS", "FOLLOWUP INSTRUCTIONS", "DISCHARGE DISPOSITION",
    "TRANSITIONAL ISSUES", "IMPRESSION", "FINDINGS",
)
_HEAD = re.compile(
    r"^\s*(" + "|".join(re.escape(name) for name in sorted(SECTION_HEADERS, key=len, reverse=True))
    + r")\s*:\s*(.*)$", re.I
)
_SUBHEADING = re.compile(
    r"^\s*(?:Mental Status|Level of Consciousness|Activity Status|"
    r"Condition|Dispo|Lungs|Respiratory|Neuro|Extremities)\s*:", re.I
)
_BULLET = re.compile(r"^\s*(?:[-*#]|\d{1,2}[.)])\s+")
_DISCHARGE_CUE = re.compile(
    r"\b(?:at (?:the time of )?discharge|(?:on|by) (?:the )?day of discharge|"
    r"prior to discharge|upon discharge|discharged|will be discharged|"
    r"by discharge)\b", re.I
)
_FUTURE_OBSERVED = re.compile(
    r"\b(?:subsequently readmitted|readmitted after discharge|"
    r"died after discharge|post-discharge admission|after discharge (?:was|were) admitted)\b", re.I
)
_PERSISTENCE = re.compile(
    r"\b(?:persistent|persisting|unresolved|remaining|unchanged|"
    r"follow.?up|repeat (?:CT|CXR|chest|imaging)|still present|"
    r"residual|not resolved)\b", re.I
)

# 宽松的候选词表：不命中仍为未知，绝不编码为阴性。
PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "residual_dyspnea": (
        ("dyspnea", re.compile(r"\bdyspn\w*\b", re.I)),
        ("shortness_of_breath", re.compile(r"\bshortness\s+of\s+breath\b", re.I)),
        ("sob", re.compile(r"\bS\.?O\.?B\.?\b", re.I)),
        ("breathing_status", re.compile(r"\b(?:breathless\w*|difficulty breathing|breathing (?:comfortably|better|improved|at baseline))\b", re.I)),
        ("doe_distress", re.compile(r"\b(?:DOE|air hunger|respiratory distress|trouble breathing|"
                                    r"labou?red breathing|non.?labou?red breathing)\b", re.I)),
    ),
    "residual_cough_sputum": (
        ("cough", re.compile(r"\bcough\w*\b", re.I)),
        ("sputum", re.compile(r"\bsputum\b", re.I)),
        ("phlegm_secretions", re.compile(r"\b(?:phlegm|respiratory secretions|productive of mucus)\b", re.I)),
    ),
    "oxygen_or_niv_at_discharge": (
        ("oxygen", re.compile(r"\b(?:oxygen|O2|O₂|nasal cannula|\d+(?:\.\d+)?\s*(?:L|liters?)\s*(?:NC|oxygen))\b", re.I)),
        ("room_air", re.compile(r"\broom air\b|\bon RA\b", re.I)),
        ("noninvasive_ventilation", re.compile(r"\b(?:BiPAP|CPAP|NIV|non.?invasive ventilation|home ventilator)\b", re.I)),
    ),
    "mobility_at_discharge": (
        ("walking", re.compile(r"\b(?:walk\w*|ambulat\w*|gait|bedbound|wheelchair|walker|cane)\b", re.I)),
        ("function", re.compile(r"\b(?:activity status|functional status|ADLs?|decondition\w*|frail\w*|requires? assist\w*|independent mobil\w*)\b", re.I)),
        ("rehab", re.compile(r"\b(?:rehab(?:ilitation)?|physical therapy|occupational therapy|weight.?bearing|WBAT|TDWB)\b", re.I)),
    ),
    "inhaler_use_or_adherence": (
        ("device", re.compile(r"\b(?:inhaler\w*|MDI|DPI|nebuliz\w*|Advair|Spiriva|Symbicort|Trelegy)\b", re.I)),
        ("behavior", re.compile(r"\b(?:adheren\w*|non.?complian\w*|not taking|not using|incorrect\w* use|"
                                r"inhaler technique|taught|teach|education|unable to use|only uses|stopped using|"
                                r"discontinued (?:his|her|the) inhaler)\b", re.I)),
    ),
    "social_support": (
        ("living", re.compile(r"\b(?:lives? alone|living alone|lives? with|assisted living|homeless|"
                              r"group home|extended care|skilled nursing facility|\bSNF\b|nursing home)\b", re.I)),
        ("care_services", re.compile(r"\b(?:caregiver|home with service\w*|home health|home nursing|"
                                     r"visiting nurse|\bVNA\b|home health aide|family support|"
                                     r"supportive family|family (?:very )?involved|brother (?:very )?involved|"
                                     r"sister (?:very )?involved|son will stay|daughter will stay)\b", re.I)),
        ("access", re.compile(r"\b(?:transportation barrier|financial barrier|cannot afford|"
                              r"lack of support|no family support|medication access)\b", re.I)),
    ),
    "steroid_or_antibiotic_discharge_plan": (
        ("steroid", re.compile(r"\b(?:predni\w*|methylpred\w*|dexamethasone|hydrocortisone|"
                              r"Medrol|Solu.?Medrol|Decadron|systemic steroid\w*|steroid taper)\b", re.I)),
        ("antibiotic", re.compile(r"\b(?:antibiot\w*|azithro\w*|clarithro\w*|doxy\w*|"
                                 r"levo\w*|moxi\w*|amoxi\w*|augmentin|cef\w*|"
                                 r"cipro\w*|vancomycin|metronidazole|rifaximin|Flagyl|"
                                 r"Bactrim|TMP.?SMX|piperacillin|tazobactam|Zosyn|"
                                 r"ampicillin|sulbactam|Unasyn|meropenem|ertapenem)\b", re.I)),
        ("course_plan", re.compile(r"\b(?:taper|complete (?:the|a) course|\d+[- ]day course|"
                                   r"continue for \d+ (?:more )?days?|last dose)\b", re.I)),
    ),
    "unresolved_infection_or_imaging": (
        ("infection", re.compile(r"\b(?:pneumoni\w*|pulmonary infection|respiratory infection|"
                                 r"infectious infiltrate)\b", re.I)),
        ("pulmonary_imaging", re.compile(r"\b(?:consolidat\w*|opacit\w*|infiltrat\w*|"
                                         r"pulmonary nodule\w*|lung nodule\w*|pleural effusion|"
                                         r"pulmonary edema|ground.?glass|bronchial wall thickening|"
                                         r"atelecta\w*|bronchiecta\w*|airspace disease|"
                                         r"pulmonary vascular congestion|pleural thickening|"
                                         r"interstitial markings|cavitary (?:lung )?lesion|lung mass)\b", re.I)),
        ("followup_imaging", re.compile(r"\b(?:repeat (?:chest )?(?:CT|CXR|X.?ray)|"
                                        r"follow.?up (?:chest )?(?:CT|CXR|imaging))\b", re.I)),
    ),
}


@dataclass(frozen=True)
class TextUnit:
    section: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class RuleMention:
    field: str
    section: str
    start_line: int
    end_line: int
    text: str
    pattern_ids: tuple[str, ...]
    temporality: str
    assertion: str
    eligible: bool
    exclusion_reason: str


def text_units(note: str) -> list[TextUnit]:
    """扫描全篇；将换行句重组成短单元，保留原文行范围和顶层章节。"""

    units: list[TextUnit] = []
    section = "UNHEADED"
    parts: list[str] = []
    start = 0
    end = 0

    def flush() -> None:
        nonlocal parts, start, end
        if parts:
            joined = " ".join(parts)
            # 同一物理行中常有多句话；不能让末句的“出院”追认首句的入院症状。
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z#\d])", joined):
                if sentence.strip():
                    units.append(TextUnit(section, start, end, sentence.strip()))
            parts = []

    for line_number, raw in enumerate(note.splitlines(), 1):
        heading = _HEAD.match(raw)
        if heading:
            flush()
            section = heading.group(1).upper()
            content = heading.group(2).strip()
        else:
            content = raw.strip()
        if not content:
            flush()
            continue
        if parts and (_SUBHEADING.match(content) or _BULLET.match(content)
                      or sum(len(part) for part in parts) + len(content) > 700):
            flush()
        if not parts:
            start = line_number
        parts.append(content)
        end = line_number
        if re.search(r"[.!?]\s*$", content):
            flush()
    flush()
    return units


def _temporality(unit: TextUnit) -> str:
    if _FUTURE_OBSERVED.search(unit.text):
        return "post_discharge_observed"
    section = unit.section
    if section in {"DISCHARGE CONDITION", "DISCHARGE PHYSICAL EXAM", "DISCHARGE EXAM"}:
        return "discharge_state"
    if section in {"DISCHARGE INSTRUCTIONS", "FOLLOWUP INSTRUCTIONS",
                   "DISCHARGE MEDICATIONS", "TRANSITIONAL ISSUES"}:
        return "discharge_plan"
    if section == "DISCHARGE DISPOSITION":
        return "discharge_state"
    if section == "BRIEF HOSPITAL COURSE":
        return "discharge_state" if _DISCHARGE_CUE.search(unit.text) else "hospital_course"
    if section in {"CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM",
                   "MEDICATIONS ON ADMISSION"}:
        return "admission"
    if section in {"PAST MEDICAL HISTORY", "FAMILY HISTORY", "SOCIAL HISTORY"}:
        return "history"
    if section in {"PERTINENT RESULTS", "IMPRESSION", "FINDINGS"}:
        return "test_result"
    return "unknown"


def _assertion(field: str, text: str) -> str:
    negative = {
        "residual_dyspnea": r"\b(?:denies? (?:dyspnea|shortness of breath|SOB)|"
                            r"no (?:dyspnea|shortness of breath)|breathing comfortably|dyspnea resolved)\b",
        "residual_cough_sputum": r"\b(?:denies? (?:cough|sputum)|no (?:cough|sputum)|cough resolved)\b",
        "oxygen_or_niv_at_discharge": r"\b(?:room air|off oxygen|weaned off oxygen|no home (?:O2|oxygen))\b",
        "unresolved_infection_or_imaging": r"\b(?:no (?:consolidation|pneumonia|infiltrate|opacity)|"
                                            r"resolved (?:consolidation|pneumonia|opacity))\b",
    }.get(field)
    if negative and re.search(negative, text, re.I):
        return "explicit_negative_or_resolved"
    if re.search(r"\b(?:possible|suspected|may represent|cannot exclude|rule out)\b", text, re.I):
        return "uncertain"
    return "unclassified"


def _eligibility(field: str, unit: TextUnit, matches: tuple[str, ...], time: str) -> tuple[bool, str]:
    if time == "post_discharge_observed":
        return False, "post_discharge_observed"
    if field in {"residual_dyspnea", "residual_cough_sputum", "oxygen_or_niv_at_discharge",
                 "mobility_at_discharge"}:
        if field in {"residual_dyspnea", "residual_cough_sputum"} and (
            unit.section == "DISCHARGE INSTRUCTIONS" and not re.search(
                r"\b(?:still|persistent|remain\w*|improved|resolved|now|currently|at discharge)\b",
                unit.text, re.I,
            )
        ):
            return False, "instruction_recounts_history"
        if field == "residual_cough_sputum" and _matches_only_sputum_test(unit.text):
            return False, "sputum_test_not_symptom"
        if field == "oxygen_or_niv_at_discharge" and not re.search(
            r"\b(?:home (?:oxygen|O2)|supplemental (?:oxygen|O2)|"
            r"oxygen (?:requirement|required|needed)|on (?:home )?oxygen|"
            r"off oxygen|weaned off oxygen|room air|on RA|BiPAP|CPAP|NIV|"
            r"nasal cannula|\d+(?:\.\d+)?\s*(?:L|liters?)\s*(?:NC|oxygen))\b",
            unit.text, re.I,
        ):
            return False, "measurement_not_oxygen_support"
        if time != "discharge_state" and not (
            field == "oxygen_or_niv_at_discharge" and time == "discharge_plan"
        ):
            return False, "not_discharge_state"
    elif field == "inhaler_use_or_adherence":
        if "behavior" not in matches:
            return False, "drug_or_device_mention_only"
        if time in {"test_result", "unknown"}:
            return False, "temporality_unknown"
    elif field == "social_support":
        if time not in {"discharge_state", "discharge_plan", "hospital_course", "history"}:
            return False, "support_context_unknown"
    elif field == "steroid_or_antibiotic_discharge_plan":
        if time != "discharge_plan" and not (
            time == "discharge_state" and unit.section == "BRIEF HOSPITAL COURSE"
        ):
            return False, "not_discharge_plan"
        if not ({"steroid", "antibiotic"} & set(matches)):
            return False, "course_without_drug"
    elif field == "unresolved_infection_or_imaging":
        if time not in {"discharge_state", "discharge_plan"}:
            return False, "not_discharge_status"
        if not _PERSISTENCE.search(unit.text):
            return False, "persistence_unproven"
        if not ({"infection", "pulmonary_imaging"} & set(matches)):
            return False, "imaging_topic_absent"
    return True, ""


def _matches_only_sputum_test(text: str) -> bool:
    if re.search(r"\bcough\w*\b", text, re.I):
        return False
    return re.search(r"\bsputum\s+(?:culture|cultures|sample|specimen|Cx)\b", text, re.I) is not None


def extract_mentions(note: str) -> list[RuleMention]:
    """保存每个规则候选；任何未命中字段保持未知，不解释为阴性。"""

    result: list[RuleMention] = []
    for unit in text_units(note):
        time = _temporality(unit)
        for field in FIELDS:
            matches = tuple(name for name, pattern in PATTERNS[field] if pattern.search(unit.text))
            if not matches:
                continue
            eligible, reason = _eligibility(field, unit, matches, time)
            result.append(RuleMention(
                field, unit.section, unit.start_line, unit.end_line, unit.text,
                matches, time, _assertion(field, unit.text), eligible, reason,
            ))
    return result
