from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "model_loader"))

from aecopd_loader import AecopdDataBundle  # noqa: E402


def main() -> int:
    # 在训练折上拟合静态标准化与词表，写出 timeline_loader_preprocessing.json。
    # 复现被移除的原生成入口，并执行当前受控临床特征合同。
    parser = argparse.ArgumentParser(description="Fit train-only preprocessing artifacts.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, default=None,
                        help="Reuse section text index from another dir (e.g. model_v2). "
                             "fit_preprocessing never reads text, but the bundle validates its path.")
    args = parser.parse_args()

    if args.text_dir is not None:
        bundle = AecopdDataBundle(args.data_dir, text_metadata_only=True, text_dir=args.text_dir)
    else:
        bundle = AecopdDataBundle(args.data_dir)
    artifacts = bundle.fit_preprocessing(args.output)
    print(json.dumps({
        "schema_version": artifacts.get("schema_version"),
        "fit_split": artifacts.get("fit_split"),
        "train_admissions": artifacts.get("train_admissions"),
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
