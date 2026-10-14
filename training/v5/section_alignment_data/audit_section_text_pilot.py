"""统计候选章节文本的长度、重复和数字密度，不保存原文。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def audit(manifest_dir: Path) -> dict[str, object]:
    with gzip.open(manifest_dir / "shared_text.csv.gz", "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    chunks_by_section: Counter[str] = Counter()
    after_discharge: Counter[str] = Counter()
    normalized: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_heavy: Counter[str] = Counter()
    for row in rows:
        role = row["section_role"]
        by_role[role].append(row)
        chunks_by_section[row["section_id"]] += 1
        after_discharge[role] += int(row["timestamp_relation"] == "after_discharge")
        canonical = re.sub(r"\d", "#", " ".join(row["chunk_text"].lower().split()))
        normalized[role][hashlib.sha256(canonical.encode("utf-8")).hexdigest()] += 1
        letters = sum(character.isalpha() for character in row["chunk_text"])
        digits = sum(character.isdigit() for character in row["chunk_text"])
        numeric_heavy[role] += int(digits > letters)
    result: dict[str, object] = {"status": "content_statistics_not_clinical_truth", "roles": {}}
    for role, items in by_role.items():
        sections = {item["section_id"] for item in items}
        section_chunks = [chunks_by_section[sid] for sid in sections]
        bert_lengths = [int(item["bert_token_count"]) for item in items]
        duplicates = sum(count for count in normalized[role].values() if count >= 5)
        result["roles"][role] = {
            "sections": len(sections), "chunks": len(items),
            "median_bert_tokens_per_chunk": median(bert_lengths),
            "max_chunks_per_section": max(section_chunks),
            "sections_over_4_chunks": sum(count > 4 for count in section_chunks),
            "chunks_in_repeated_normalized_text_5plus": duplicates,
            "numeric_heavy_chunks": numeric_heavy[role],
            "stored_after_discharge_chunks": after_discharge[role],
        }
    (manifest_dir / "content_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
