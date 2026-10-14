from __future__ import annotations

import argparse
import json
from pathlib import Path

from aecopd_training.patient_sequences import build_patient_sequence_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal patient admission ordering and patient-equal loss weights."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_patient_sequence_index(
        args.cohort,
        args.output_dir / "patient_sequence_index.csv.gz",
        args.output_dir / "patient_sequence_summary.json",
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
