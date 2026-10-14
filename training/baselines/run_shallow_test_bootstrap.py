"""在冻结测试集上评估已定型的浅层基线 B1/B2 并做患者聚类 bootstrap。

B1/B2 的所有超参与拟合都只用训练折（开发对比阶段用 validation），此处对测试集只做
一次纯预测评估以并入四模型对照表——非模型选择，符合测试集冻结原则。

输出 b1_test.json / b2_test.json，格式与 m1_test.json / b3_test.json 一致，供统一对照表。
受控数据：不打印真实 subject_id/hadm_id。
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
from baselines import b2_cause_specific_cox as b2
from baselines.shared_contract import SplitContract
from aecopd_training.conditional_outcomes import ConditionalOutcomeStore
from aecopd_training.patient_bootstrap import BootstrapConfig, patient_clustered_bootstrap


def _save(payload_key, boot, n_patients, n_admissions, notes, out_path):
    payload = {
        payload_key: {
            "metrics": boot.to_json(),
            "n_patients": int(n_patients),
            "n_admissions": int(n_admissions),
            "note": "pure eval on frozen test set; model fixed on train, not selected on test",
            "notes": notes,
        }
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    m = boot.to_json()["metrics"]
    def pt(k):
        v = m.get(k)
        return round(v["point"], 4) if isinstance(v, dict) else v
    print(f"[{payload_key}] n_patients={n_patients} n_admissions={n_admissions}")
    for iv in ("days_0_30", "days_31_90", "days_91_365"):
        print(f"  AECOPD AUC {iv}: {pt(f'test_aecopd_auc_{iv}')}  death AUC: {pt(f'test_death_auc_{iv}')}")
    print(f"  tail acc: {pt('test_tail_accuracy')}  saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/workspace/outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base = Path(args.base_dir)
    v3 = base / "model_v3"
    v2 = base / "model_v2"
    out_dir = base / "baseline_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    split = SplitContract(patient_sequence_index_path=v3 / "patient_sequence_index.csv.gz")
    cohort = v3 / "cohort_labels_static.csv.gz"
    store = ConditionalOutcomeStore.from_cohort(cohort)
    bcfg = BootstrapConfig(n_resamples=args.n_resamples, seed=args.seed)

    # ---------------- B2 (cause-specific Cox) on test ----------------
    b2cfg = b2.CoxBaselineConfig(cohort_csv_gz_path=cohort, split=split)
    train_rows = [r for r in b2._read_cohort_rows(cohort) if r["data_split"] == "train"]
    stats = b2._fit_standardizer(train_rows)
    train_frame = b2.assemble_survival_frame(b2cfg, "train", stats)
    aecopd_model, death_model = b2.fit_cause_specific_cox(train_frame, b2cfg)
    tail_subset = train_frame.event_type == 0
    train_tail_labels = np.array(
        [store[int(h)].tail_label for h in train_frame.hadm_ids[tail_subset]], dtype=np.int64
    )
    tail_model = b2.fit_tail_multinomial(train_frame.features[tail_subset], train_tail_labels, b2cfg)
    preds = b2.predict_split(b2cfg, stats, aecopd_model, death_model, tail_model, "test")
    il, im, tl, tm = b2._aligned_labels(store, preds.hadm_ids)
    boot_b2 = patient_clustered_bootstrap(
        preds.interval_probabilities, il, im, preds.tail_probabilities, tl, tm,
        preds.subject_ids, prefix="test", config=bcfg,
    )
    _save("b2_test", boot_b2, np.unique(preds.subject_ids).shape[0], preds.hadm_ids.shape[0],
          f"cause-specific Cox; aecopd_converged={aecopd_model.converged}, death_converged={death_model.converged}",
          out_dir / "b2_test.json")

    # ---------------- B1 (HAIM XGBoost) on test ----------------
    b1cfg = b1.HaimXgbConfig(
        section_embedding_dir=v2,
        events_path=v3 / "events.csv.gz",
        cohort_csv_gz_path=cohort,
        split=split,
    )
    result_b1, p = b1.run_b1(b1cfg, eval_split="test")
    boot_b1 = patient_clustered_bootstrap(
        p["interval_probabilities"], p["interval_labels"], p["interval_mask"],
        p["tail_probabilities"], p["tail_labels"], p["tail_mask"],
        p["subject_ids"], prefix="test", config=bcfg,
    )
    _save("b1_test", boot_b1, np.unique(p["subject_ids"]).shape[0], p["hadm_ids"].shape[0],
          result_b1.notes, out_dir / "b1_test.json")


if __name__ == "__main__":
    main()
