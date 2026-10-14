"""统计 V5 出院小结的独立成行小标题，不读取结局标签。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path


# 只计独立成行、以冒号收尾的短标题；排除编号问题、去标识符及病历正文。
HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&()'\-]{1,64})\s*:\s*$")


def normalize_title(title: str) -> str:
    """合并大小写与多余空白，不擅自合并不同语义的章节。"""

    return " ".join(title.lower().split())


def audit(source: Path) -> dict[str, object]:
    documents = 0
    heading_docs: Counter[str] = Counter()
    heading_occurrences: Counter[str] = Counter()
    variants: dict[str, Counter[str]] = {}
    sequence_docs: Counter[tuple[str, ...]] = Counter()
    headings_per_note: Counter[int] = Counter()
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            documents += 1
            headings: list[str] = []
            for line in row["document_text"].splitlines():
                match = HEADING.fullmatch(line)
                if not match:
                    continue
                title = match.group(1).strip()
                normalized = normalize_title(title)
                headings.append(normalized)
                heading_occurrences[normalized] += 1
                variants.setdefault(normalized, Counter())[title] += 1
            heading_docs.update(set(headings))
            headings_per_note[len(headings)] += 1
            # 低频问题列表不能决定模板；保留全序列以供计算稳定性。
            sequence_docs[tuple(headings)] += 1
    if not documents:
        raise ValueError("source contains no discharge notes")
    visible = sorted(
        (title for title, count in heading_docs.items() if count >= 5),
        key=lambda title: (-heading_docs[title], title),
    )
    common = [title for title in visible if heading_docs[title] >= documents * 0.5]
    core = [title for title in visible if heading_docs[title] >= documents * 0.8]
    common_order: Counter[tuple[str, ...]] = Counter()
    common_set = set(common)
    core_set = set(core)
    core_order: Counter[tuple[str, ...]] = Counter()
    all_core_documents = 0
    for sequence, count in sequence_docs.items():
        common_order[tuple(title for title in sequence if title in common_set)] += count
        # 同一标题在病程内重复时只计首次位置。
        first_core = tuple(dict.fromkeys(title for title in sequence if title in core_set))
        core_order[first_core] += count
        if len(first_core) == len(core):
            all_core_documents += count
    modal_core_order = core_order.most_common(1)[0][0]
    modal_core_index = {title: index for index, title in enumerate(modal_core_order)}
    order_consistent_documents = sum(
        count for sequence, count in core_order.items()
        if all(modal_core_index.get(left, -1) < modal_core_index.get(right, -1)
               for left, right in zip(sequence, sequence[1:]))
    )
    report: dict[str, object] = {
        "source": str(source.resolve()),
        "documents": documents,
        "standalone_colon_heading_definition": HEADING.pattern,
        "distinct_normalized_titles": len(heading_docs),
        "titles_in_at_least_five_documents": len(visible),
        "headings_per_note": dict(sorted(headings_per_note.items())),
        "headings": [
            {
                "title": title,
                "documents": heading_docs[title],
                "mentions": heading_occurrences[title],
                "percent_documents": round(100 * heading_docs[title] / documents, 2),
                "variants": dict(variants[title].most_common()),
            }
            for title in visible
        ],
        "titles_in_at_least_half_of_documents": common,
        "core_titles_in_at_least_eighty_percent_of_documents": core,
        "documents_with_all_core_titles": all_core_documents,
        "modal_core_title_order": modal_core_order,
        "documents_with_modal_core_title_order": core_order.most_common(1)[0][1],
        "documents_with_core_titles_in_modal_relative_order": order_consistent_documents,
        "most_common_full_sequence_count": sequence_docs.most_common(1)[0][1],
        "most_common_common_title_order_count": common_order.most_common(1)[0][1],
        "most_common_common_title_order": common_order.most_common(1)[0][0],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "documents": report["documents"],
        "distinct_normalized_titles": report["distinct_normalized_titles"],
        "titles_in_at_least_five_documents": report["titles_in_at_least_five_documents"],
        "most_common_full_sequence_count": report["most_common_full_sequence_count"],
        "most_common_common_title_order_count": report["most_common_common_title_order_count"],
        "documents_with_all_core_titles": report["documents_with_all_core_titles"],
        "documents_with_modal_core_title_order": report["documents_with_modal_core_title_order"],
        "documents_with_core_titles_in_modal_relative_order": report["documents_with_core_titles_in_modal_relative_order"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
