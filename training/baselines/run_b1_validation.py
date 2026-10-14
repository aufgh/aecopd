"""B1 端到端驱动：验证集上跑 HAIM 风格 XGBoost 基线并存患者聚类 bootstrap CI。

与 B2 存储格式一致：outputs/<sub>/private/baseline_results/b1_validation.json，
键 b1_validation -> {metrics(bootstrap per-metric point/CI), n_patients, n_admissions, notes}。
在 aecopd-baselines:v1 容器内运行；测试集保持冻结（默认 eval_split=validation）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/training")
sys.path.insert(0, "/workspace/model_loader")

from baselines import b1_haim_xgboost as b1
from baselines.shared_contract import SplitContract, bootstrap_with_project_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/workspace/outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base = Path(args.base_dir)
    v3 = base / "model_v3"
    v2 = base / "model_v2"
    out_dir = base / "baseline_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = b1.HaimXgbConfig(
        section_embedding_dir=v2,
        events_path=v3 / "events.csv.gz",
        cohort_csv_gz_path=v3 / "cohort_labels_static.csv.gz",
        split=SplitContract(patient_sequence_index_path=v3 / "patient_sequence_index.csv.gz"),
    )

    result, preds = b1.run_b1(config, eval_split=args.eval_split)

    boot = bootstrap_with_project_metrics(
        preds["interval_probabilities"],
        preds["interval_labels"],
        preds["interval_mask"],
        preds["tail_probabilities"],
        preds["tail_labels"],
        preds["tail_mask"],
        preds["subject_ids"],
        prefix=args.eval_split,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    n_patients = int(np.unique(preds["subject_ids"]).shape[0])
    n_admissions = int(preds["hadm_ids"].shape[0])
    payload = {
        f"b1_{args.eval_split}": {
            "metrics": boot.to_json(),
            "n_patients": n_patients,
            "n_admissions": n_admissions,
            "notes": result.notes,
        }
    }
    out_path = out_dir / f"b1_{args.eval_split}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 控制台仅打印聚合指标点估计（不含任何真实 id）。
    m = boot.to_json()
    def pt(k):
        v = m.get(k)
        return round(v["point"], 4) if isinstance(v, dict) else v
    print(f"[B1] eval_split={args.eval_split} n_patients={n_patients} n_admissions={n_admissions}")
    for iv in ("days_0_30", "days_31_90", "days_91_365"):
        print(f"  AECOPD AUC {iv}: {pt(f'{args.eval_split}_aecopd_auc_{iv}')}  "
              f"death AUC: {pt(f'{args.eval_split}_death_auc_{iv}')}")
    print(f"  tail accuracy: {pt(f'{args.eval_split}_tail_accuracy')}")
    print(f"[B1] saved -> {out_path}")


if __name__ == "__main__":
    main()
