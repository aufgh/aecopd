"""汇总异常分支多种子配对试验，不把重复查看的历史测试集用于模型选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path, output: Path, seeds: tuple[int, ...]) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid seed list")
    rows = []
    reference: dict[str, object] | None = None
    for seed in seeds:
        reports = {
            mode: json.loads((root / f"{mode}_seed{seed}" / "report.json").read_text(encoding="utf-8"))
            for mode in ("masked", "core9", "measurement_only", "active")
        }
        control = reports["masked"]
        if reference is None:
            reference = control
        else:
            for key in ("cohort_patients", "cohort_admissions", "label_sha256",
                        "field_manifest_sha256", "raw_source_sha256", "parameter_count",
                        "text_vector_width", "event_encoder", "clinical_recovery_manifest_sha256"):
                if control[key] != reference[key]:
                    raise ValueError(f"{key} differs between seeds")
        for mode, report in reports.items():
            for key in ("cohort_patients", "cohort_admissions", "label_sha256",
                        "field_manifest_sha256", "raw_source_sha256", "parameter_count",
                        "text_vector_width", "event_encoder", "clinical_recovery_manifest_sha256"):
                if report[key] != control[key]:
                    raise ValueError(f"{key} differs for {mode}, seed {seed}")
            if (report["status"] != "training_completed" or report["seed"] != seed
                    or report["event_encoder"] != "abnormal" or report["raw_events_enter_model"]
                    or report["clinical_recovery_mode"] != mode):
                raise ValueError(f"invalid experiment arm: {mode}, seed {seed}")
            readmission = report["validation"]["temporal_full"]
            death = report["validation"]["death_temporal_full"]
            rows.append({"seed": seed, "mode": mode, "best_epoch": report["best_epoch"],
                         "readmission_ibs": readmission["integrated_brier_7_365"],
                         "readmission_auc": readmission["mean_auc_7_365"],
                         "death_ibs": death["integrated_brier_7_365"],
                         "death_auc": death["mean_auc_7_365"]})
    result = {"status": "exploratory_multi_seed", "seeds": seeds, "arms": len(rows),
              "cohort_patients": control["cohort_patients"],
              "cohort_admissions": control["cohort_admissions"],
              "parameter_count_each_arm": control["parameter_count"],
              "validation_only_for_design": True,
              "notes": ["All four arms share architecture, source, patient split, labels and seed within comparison.",
                        "A 200-draw patient-cluster interval is reported separately for each seed.",
                        "The historical test split is not blind and is not used to select axes."],
              "validation_rows": rows}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = parser.parse_args()
    print(json.dumps({"status": summarize(args.run_root, args.output, tuple(args.seeds))["status"]}))
