"""将额外小结章节追加到冻结四字段清单；旧行与向量保持不变。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from training.v5.field_text_baseline import FIELD_LABELS, FIELDS


EXTRA_LABELS = {
    "discharge_medications": "Discharge Medications",
    "discharge_instructions": "Discharge Instructions",
    "social_history": "Social History",
    "pertinent_results": "Pertinent Results",
    "discharge_physical_exam": "Discharge Physical Exam",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def prepare(base_dir: Path, candidate_dir: Path, output_dir: Path,
            bert_tokenizer: object, qwen_tokenizer: object) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base_path = base_dir / "shared_text.csv.gz"
    source_path = candidate_dir / "shared_text.csv.gz"
    base_qa = json.loads((base_dir / "validation.json").read_text(encoding="utf-8"))
    candidate_qa = json.loads((candidate_dir / "validation.json").read_text(encoding="utf-8"))
    if (base_qa["status"] != "field_text_manifest_validated"
            or base_qa["manifest_sha256"] != _sha(base_path)
            or candidate_qa["status"] != "manifest_structurally_validated"
            or candidate_qa["manifest_sha256"] != _sha(source_path)):
        raise ValueError("source manifests are not validated or have changed")
    candidate_summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    if tuple(candidate_summary["roles"]) != tuple(EXTRA_LABELS):
        raise ValueError("unexpected candidate role vocabulary")
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidate_text.csv.gz"
    combined_path = output_dir / "shared_text.csv.gz"
    counts: Counter[str] = Counter()
    admissions: set[str] = set()
    with (gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as candidate_handle,
          gzip.open(combined_path, "wt", encoding="utf-8", newline="") as combined_handle):
        candidate_writer = csv.writer(candidate_handle)
        combined_writer = csv.writer(combined_handle)
        candidate_writer.writerow(FIELDS)
        combined_writer.writerow(FIELDS)
        for index, row in enumerate(_rows(base_path)):
            if int(row["embedding_row"]) != index or row["field"] not in FIELD_LABELS:
                raise ValueError("baseline row/order changed")
            combined_writer.writerow([row[name] for name in FIELDS])
            counts["baseline"] += 1
            admissions.add(row["hadm_id"])
        for source_index, source in enumerate(_rows(source_path)):
            role = source["section_role"]
            if role not in EXTRA_LABELS or int(source["embedding_row"]) != source_index:
                raise ValueError("candidate row/order changed")
            raw = source["chunk_text"]
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != source["text_sha256"]:
                raise ValueError("candidate content hash mismatch")
            text = f"Clinical field: {EXTRA_LABELS[role]}\n{raw}"
            bert_count = len(bert_tokenizer(text, add_special_tokens=True)["input_ids"])
            qwen_count = len(qwen_tokenizer(text, add_special_tokens=False)["input_ids"])
            if not 1 <= bert_count <= 512 or not 1 <= qwen_count <= 8192:
                raise ValueError("candidate text exceeds encoder context")
            common = (
                source["subject_id"], source["hadm_id"], role, source["section_id"],
                source["chunk_index"], source_index, source["source_start_char"],
                source["source_end_char"], source["source_section_sha256"],
                source["timestamp_relation"], bert_count, qwen_count,
                source["text_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
            )
            candidate_writer.writerow((source_index, *common))
            combined_writer.writerow((counts["baseline"] + source_index, *common))
            counts[role] += 1
            admissions.add(source["hadm_id"])
    if sum(counts[role] for role in EXTRA_LABELS) != candidate_qa["chunks"]:
        raise ValueError("candidate row count changed")
    summary = {
        "status": "screened_candidate_not_encoded",
        "fields": [*FIELD_LABELS, *EXTRA_LABELS],
        "baseline_rows": counts["baseline"],
        "candidate_rows": candidate_qa["chunks"],
        "chunks": counts["baseline"] + candidate_qa["chunks"],
        "admissions_with_text": len(admissions),
        "chunks_by_extra_field": {role: counts[role] for role in EXTRA_LABELS},
        "baseline_manifest_sha256": _sha(base_path),
        "source_candidate_manifest_sha256": _sha(source_path),
        "candidate_manifest_sha256": _sha(candidate_path),
        "manifest_sha256": _sha(combined_path),
        "clinical_semantics_validated": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def combine_vectors(base_dir: Path, extension_dir: Path, encoded_dir: Path) -> dict[str, object]:
    summary = json.loads((extension_dir / "summary.json").read_text(encoding="utf-8"))
    candidate_path = extension_dir / "candidate_text.csv.gz"
    combined_path = extension_dir / "shared_text.csv.gz"
    candidate_state = json.loads((encoded_dir / "state.json").read_text(encoding="utf-8"))
    if (_sha(candidate_path) != summary["candidate_manifest_sha256"]
            or _sha(combined_path) != summary["manifest_sha256"]
            or candidate_state["status"] != "complete"
            or candidate_state["manifest_sha256"] != summary["candidate_manifest_sha256"]):
        raise ValueError("candidate encoding or manifest incomplete")
    combined_rows = _rows(combined_path)
    for original, extended in zip(_rows(base_dir / "shared_text.csv.gz"), combined_rows):
        if original != extended:
            raise ValueError("baseline text row changed in extension")
    old = np.load(base_dir / "bert/bioclinicalbert.float16.npy", mmap_mode="r")
    new = np.load(encoded_dir / "bioclinicalbert.float16.npy", mmap_mode="r")
    if old.shape != (summary["baseline_rows"], 768) or new.shape != (summary["candidate_rows"], 768):
        raise ValueError("embedding shape mismatch")
    target_dir = extension_dir / "bert"
    if target_dir.exists():
        raise FileExistsError(target_dir)
    target_dir.mkdir()
    target = np.lib.format.open_memmap(target_dir / "bioclinicalbert.float16.npy", mode="w+",
                                       dtype=np.float16, shape=(summary["chunks"], 768))
    target[:len(old)] = old
    target[len(old):] = new
    target.flush()
    if not np.array_equal(target[:len(old)], old) or not np.array_equal(target[len(old):], new):
        raise ValueError("combined vectors differ from sources")
    state = {"status": "complete", "model": "Bio_ClinicalBERT", "device": "cuda",
             "rows": summary["chunks"], "completed_rows": summary["chunks"],
             "dimension": 768, "manifest_sha256": summary["manifest_sha256"],
             "baseline_vector_sha256": _sha(base_dir / "bert/bioclinicalbert.float16.npy"),
             "candidate_vector_sha256": _sha(encoded_dir / "bioclinicalbert.float16.npy")}
    (target_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    validation = {"status": "field_text_manifest_validated",
                  "manifest_sha256": summary["manifest_sha256"],
                  "chunks": summary["chunks"], "admissions": summary["admissions_with_text"],
                  "baseline_rows_unchanged": True, "clinical_semantics_validated": False}
    (extension_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "combine-vectors"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--encoded-dir", type=Path)
    parser.add_argument("--bert-model-dir", type=Path)
    parser.add_argument("--qwen-model-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        from transformers import AutoTokenizer
        if args.candidate_dir is None or args.bert_model_dir is None or args.qwen_model_dir is None:
            parser.error("prepare requires candidate and both tokenizer directories")
        bert = AutoTokenizer.from_pretrained(args.bert_model_dir, local_files_only=True)
        qwen = AutoTokenizer.from_pretrained(args.qwen_model_dir, local_files_only=True)
        result = prepare(args.base_dir, args.candidate_dir, args.extension_dir, bert, qwen)
    else:
        if args.encoded_dir is None:
            parser.error("combine-vectors requires encoded directory")
        result = combine_vectors(args.base_dir, args.extension_dir, args.encoded_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
