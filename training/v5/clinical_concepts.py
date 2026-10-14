"""V5 共用的轻量临床概念映射；不依赖训练框架。"""

from __future__ import annotations

import re


LABS: dict[int, tuple[str, int]] = {
    50820: ("ph", 3), 50818: ("paco2", 3), 50821: ("pao2", 3),
    50882: ("bicarbonate", 4), 51301: ("wbc", 4),
    51222: ("hemoglobin", 4), 50912: ("creatinine", 4), 50862: ("albumin", 4),
}
MED_NAMES = ("saba", "sama", "laba", "lama", "inhaled_steroid", "systemic_steroid",
             "macrolide", "beta_lactam", "fluoroquinolone", "other_antibiotic",
             "loop_diuretic", "other_diuretic")

_MED_PATTERNS = {
    "saba": r"\b(?:albuterol|salbutamol|levalbuterol)\b",
    "sama": r"\bipratropium\b",
    "laba": r"\b(?:salmeterol|formoterol|arformoterol|indacaterol|olodaterol|vilanterol)\b",
    "lama": r"\b(?:tiotropium|umeclidinium|glycopyrrolate|aclidinium|revefenacin)\b",
    "inhaled_steroid": r"\b(?:budesonide|fluticasone|beclomethasone|mometasone|ciclesonide)\b",
    "systemic_steroid": r"\b(?:prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone)\b",
    "macrolide": r"\b(?:azithromycin|clarithromycin|erythromycin)\b",
    "beta_lactam": r"\b(?:cef\w+|cephalexin|amoxicillin|ampicillin|piperacillin|tazobactam|meropenem|ertapenem|imipenem|aztreonam|penicillin)\b",
    "fluoroquinolone": r"\b(?:levofloxacin|ciprofloxacin|moxifloxacin|ofloxacin)\b",
    "other_antibiotic": r"\b(?:vancomycin|doxycycline|tetracycline|linezolid|clindamycin|metronidazole|trimethoprim|sulfamethoxazole|gentamicin|tobramycin)\b",
    "loop_diuretic": r"\b(?:furosemide|torsemide|bumetanide)\b",
    "other_diuretic": r"\b(?:spironolactone|hydrochlorothiazide|chlorothiazide|metolazone|amiloride|eplerenone)\b",
}
_COMPILED_MED = {name: re.compile(pattern, re.I) for name, pattern in _MED_PATTERNS.items()}
_BRANDS = {
    "symbicort": {"laba", "inhaled_steroid"},
    "advair": {"laba", "inhaled_steroid"},
    "breo ellipta": {"laba", "inhaled_steroid"},
    "trelegy ellipta": {"laba", "lama", "inhaled_steroid"},
    "spiriva": {"lama"}, "incruse ellipta": {"lama"},
    "anoro ellipta": {"laba", "lama"}, "stiolto respimat": {"laba", "lama"},
    "duoneb": {"saba", "sama"}, "combivent": {"saba", "sama"},
    "pulmicort": {"inhaled_steroid"}, "ventolin": {"saba"},
    "proair": {"saba"}, "xopenex": {"saba"}, "atrovent": {"sama"},
}
_LOCAL_FORM = re.compile(r"ophth|\botic\b|nasal|topical|\bcream\b|\boint\b|\bointment\b|\blotion\b|"
                         r"\beye\s*(?:drop|gel)|\bear\s*drop|\bvaginal\b|\brectal\b|"
                         r"\benema\b|\bsuppositor(?:y|ies)\b|\bnepafenac\b|"
                         r"\bclindamycin\s*1%\s*solution\b", re.I)
_INHALED_CLASSES = {"saba", "sama", "laba", "lama", "inhaled_steroid"}
_ORAL_ROUTES = {"PO", "NG", "GT", "PEG", "OGT"}
_TETRACYCLINE = re.compile(r"\b(?:doxycycline|tetracycline|minocycline)\b", re.I)
_OTHER_ANTIBIOTIC_NON_TETRACYCLINE = re.compile(
    r"\b(?:vancomycin|linezolid|clindamycin|metronidazole|trimethoprim|sulfamethoxazole|gentamicin|tobramycin)\b", re.I)


def medication_classes(name: str, *, product_descriptions: tuple[str, ...] = (),
                       routes: tuple[str, ...] = (), mapping_version: str = "legacy") -> set[str]:
    """同一复方可贡献多个类别；给药详情只用于剂型判定，不作为模型特征。"""

    lower = name.lower()
    if mapping_version not in ("legacy", "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("unknown medication mapping version")
    result = {key for key, pattern in _COMPILED_MED.items() if pattern.search(lower)}
    if mapping_version != "legacy" and _TETRACYCLINE.search(lower):
        result.add("tetracycline")
        if not _OTHER_ANTIBIOTIC_NON_TETRACYCLINE.search(lower):
            result.discard("other_antibiotic")
    if mapping_version == "clinical_reviewed_2026_09":
        # 这些记录不是可解释为本次系统治疗的完整给药。
        if re.search(r"\block\b|graded challenge|study drug|w/ lidocaine", lower):
            return set()
        if "umeclidin" in lower:
            result.add("lama")
        if "vilanter" in lower:
            result.add("laba")
    for brand, classes in _BRANDS.items():
        if brand in lower:
            result.update(classes)
    descriptions = " ".join(product_descriptions).lower()
    if _LOCAL_FORM.search(lower) or _LOCAL_FORM.search(descriptions):
        return set()
    if mapping_version == "clinical_reviewed_2026_09":
        if (lower.strip() in {"mometasone", "fluticasone"}
                and not re.search(r"inhal|aerosol|diskus|ellipta|asmanex|flovent", descriptions)):
            result.discard("inhaled_steroid")
        if "prednisolone acetate" in lower and not re.search(r"\boral\b|\bpo\b", descriptions):
            result.discard("systemic_steroid")
    # 单独的 budesonide 在本底账同时有雾化液和口服肠溶胶囊；仅凭药名不可判为 ICS。
    if lower.strip() == "budesonide" and not re.search(r"neb|inhal|aerosol|respule", descriptions):
        result.discard("inhaled_steroid")
    # 本底账的裸名 glycopyrrolate 多为片剂或麻醉逆转注射，不能推断为吸入 LAMA。
    if "glycopyrrolate" in lower and not re.search(r"neb|inhal|lonhala|seebri", lower + " " + descriptions):
        result.discard("lama")
    if "albuterol" in lower and "extended release" in lower:
        result.discard("saba")
    if routes and set(routes).issubset(_ORAL_ROUTES):
        result.difference_update(_INHALED_CLASSES)
    return result


def diagnosis_groups(code: str, version: int) -> set[str]:
    """保守 ICD 家族映射；本次出院编码不自动等于既往史。"""

    code = code.strip().upper().replace(".", "")
    result: set[str] = set()
    if version == 10:
        if code.startswith(("I50", "I110", "I130", "I132")):
            result.add("heart_failure")
        if code.startswith(("I20", "I21", "I22", "I23", "I24", "I25", "I48")):
            result.add("cardiac_other")
        if code.startswith("N18"):
            result.add("ckd")
        if code.startswith(("E08", "E09", "E10", "E11", "E13")):
            result.add("diabetes")
        if re.match(r"^C\d\d", code):
            result.add("malignancy")
        if code.startswith(("J961", "J962", "G4733")):
            result.add("chronic_respiratory")
        if code.startswith(("J960", "J962")):
            result.add("acute_resp_failure")
        if re.match(r"^J(?:1[2-8]|69)", code):
            result.add("pneumonia_or_aspiration")
        if code.startswith(("A40", "A41", "R652")):
            result.add("sepsis")
        if code.startswith("N17"):
            result.add("aki")
    elif version == 9:
        if code.startswith("428"):
            result.add("heart_failure")
        if code.startswith(("410", "411", "412", "413", "414", "42731")):
            result.add("cardiac_other")
        if code.startswith("585"):
            result.add("ckd")
        if code.startswith("250"):
            result.add("diabetes")
        if len(code) >= 3 and code[:3].isdigit() and 140 <= int(code[:3]) <= 208:
            result.add("malignancy")
        if code.startswith(("51883", "51884", "32723")):
            result.add("chronic_respiratory")
        if code.startswith(("51881", "51884")):
            result.add("acute_resp_failure")
        if len(code) >= 3 and code[:3].isdigit() and (480 <= int(code[:3]) <= 486 or code[:3] == "507"):
            result.add("pneumonia_or_aspiration")
        if code.startswith(("038", "99591", "99592")):
            result.add("sepsis")
        if code.startswith("584"):
            result.add("aki")
    else:
        raise ValueError(f"unsupported ICD version: {version}")
    return result
