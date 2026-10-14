"""同一已核验来源上运行完整22轴逐轴消融；给药只有二元暴露。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from patient_data.expanded_abnormality_features import AXES
from training.v5.medication_effect_data import attach_medication_effects
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train_event_sequence import run

PRIVATE = Path("outputs/01a08b96-2988-7001-af1d-b40d901f3400/private")
RUN_ROOT = Path("training_runs/v5_strict_all22_binary_meds_one_seed_2026-09-30")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--worker-name", default="batch")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.axes)) != len(args.axes):
        raise ValueError("duplicate seed or axis")
    paths = dict(
        aligned_dir=PRIVATE / "v5_section_aligned/dataset_v3",
        structured_dir=PRIVATE / "v5_section_aligned/section_inputs_sparse_2026_09_26",
        base_csv=PRIVATE / "v5_mimiciv_2026-09-25/structured/base.csv",
        field_dir=PRIVATE / "v5_section_aligned/field_text_baseline_2026_09_27",
        vectors=PRIVATE / "v5_section_aligned/field_text_baseline_2026_09_27/bert/bioclinicalbert.float16.npy",
        labels_csv=Path("patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz"),
        processed_event_dir=Path("patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29"),
        raw_ledger_dir=Path("patient_data/mimiciv_copd/snapshot_2026-09-27"),
    )
    started = time.perf_counter()
    print("VERIFY common source once; no new text encoding", flush=True)
    source = load_processed_event_data(
        paths["aligned_dir"], paths["structured_dir"], paths["base_csv"],
        paths["processed_event_dir"], paths["raw_ledger_dir"],
        labels_csv=paths["labels_csv"], field_dir=paths["field_dir"], vectors_path=paths["vectors"],
    )
    exposures = attach_medication_effects(source[0], paths["processed_event_dir"],
                                         mode="exposure_only").medication_by_admission
    exposures.setflags(write=False)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    completed = []
    for seed in args.seeds:
        for axis in args.axes if args.skip_baseline else (None, *args.axes):
            name = f"full_seed{seed}" if axis is None else f"without_axis_{axis}_seed{seed}"
            mode = "active" if axis is None else f"without_axis_{axis}"
            output = RUN_ROOT / name
            if output.exists():
                report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                status = json.loads((output / "status.json").read_text(encoding="utf-8"))
                if (status["status"] != "training_completed" or report["status"] != "training_completed"
                        or report["primary_event"] != STRICT_PRIMARY_EVENT or report["seed"] != seed
                        or report["clinical_recovery_mode"] != mode
                        or report["medication_effect_mode"] != "exposure_only"
                        or report["label_sha256"] != source[0].base.label_sha256
                        or report["field_manifest_sha256"] != source[0].base.manifest_sha256
                        or report["max_epochs"] != 35 or report["raw_events_enter_model"]):
                    raise ValueError(f"partial or inconsistent run: {output}")
                print(f"SKIP {name}", flush=True)
            else:
                print(f"START {name}", flush=True)
                report = run(
                    **paths, raw_dir=None, output_dir=output, process_only=False,
                    max_epochs=35, batch_size=12, event_encoder="abnormal_medication",
                    clinical_recovery_dir=Path("patient_data/processed/abnormality_clinical_reviewed_2026-09-29"),
                    clinical_recovery_mode=mode, abnormality_scale_mode="training_positive_p95",
                    medication_effect_mode="exposure_only", seed=seed,
                    prepared_source=source, prepared_exposure_only=exposures,
                )
                print(f"DONE {name} epochs={report['epochs_run']} best={report['best_epoch']} "
                      f"IBS={report['validation']['temporal_full']['integrated_brier_7_365']:.6f}", flush=True)
            completed.append(name)
            (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
                "status": "running", "completed": completed,
                "planned": len(args.seeds) * (len(args.axes) + (not args.skip_baseline)),
                "elapsed_seconds": time.perf_counter() - started,
            }, indent=2), encoding="utf-8")
    (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
        "status": "training_completed", "completed": completed, "planned": len(completed),
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")
    if tuple(args.axes) == tuple(AXES) and not args.skip_baseline:
        from training.v5.compare_all_lab_axes import compare
        compare(RUN_ROOT, RUN_ROOT / "paired_analysis", seeds=tuple(args.seeds), draws=100)
        (RUN_ROOT / f"{args.worker_name}_status.json").write_text(json.dumps({
            "status": "training_and_analysis_completed", "completed": completed, "planned": len(completed),
            "elapsed_seconds": time.perf_counter() - started,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
