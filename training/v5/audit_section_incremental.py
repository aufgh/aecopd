"""聚合新增小结章节的正文覆盖与模板空栏；不导出患者级文本。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.v5.extend_field_manifest import EXTRA_LABELS


def audit(path: Path) -> dict[str, object]:
    roles = (*EXTRA_LABELS, "followup_instructions")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    admissions: dict[str, set[str]] = defaultdict(set)
    lengths: dict[str, list[int]] = defaultdict(list)
    hashes: dict[str, Counter[str]] = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["section_role"]
            if role not in roles:
                continue
            counts[role]["sections"] += 1
            counts[role][row["content_status"]] += 1
            if row["content_status"] == "informative":
                admissions[role].add(row["hadm_id"])
                content = row["text"].strip()
                lengths[role].append(len(content))
                hashes[role][hashlib.sha256(content.encode("utf-8")).hexdigest()] += 1
    return {
        "status": "aggregate_section_content_audit",
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "roles": {role: {
            "sections": counts[role]["sections"],
            "informative_sections": counts[role]["informative"],
            "placeholder_sections": counts[role]["placeholder"],
            "admissions_with_informative_section": len(admissions[role]),
            "median_informative_chars": float(np.median(lengths[role])) if lengths[role] else None,
            "p90_informative_chars": float(np.quantile(lengths[role], .9)) if lengths[role] else None,
            "unique_informative_texts": len(hashes[role]),
            "max_identical_text_repetitions": max(hashes[role].values(), default=0),
        } for role in roles},
        "limitations": [
            "Informative means only that text contains a letter or digit; it is not clinical adjudication.",
            "Distinct section text is not proof of new information beyond structured data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["roles"], ensure_ascii=False))


if __name__ == "__main__":
    main()
