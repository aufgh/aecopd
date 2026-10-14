"""以原始住院内事件流取代检验／给药摘要的 V5 竞争风险实验。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import replace
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from training.v5.event_data import EVENT_NAMES, EventData, collate_events, load_event_data
from training.v5.event_model import TemporalFusionModel
from training.v5.event_set_model import EventSetFusionModel
from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.medication_effect_model import MedicationEffectFusionModel
from training.v5.medication_effect_data import EFFECT_ABLATIONS, EFFECT_GROUPS, attach_medication_effects
from training.v5.lab_recovery_data import ABNORMALITY_ABLATIONS, attach_abnormalities, attach_recovery
from training.v5.independent_data import IndependentData, split_sequences
from training.v5.model import competing_nll, predict_cif
from training.v5.processed_event_data import load_processed_event_data
from training.v5.train import GRID, _public_metrics, _save_curves, metrics

PRIOR_COUNT_CONCEPTS = ("context:prior_coded_admissions",
                        "context:prior_strict_aecopd_admissions")
# 整组非 context 静态概念；慢性共病 token 同时含本次和历史编码，不能解释为纯本次字段。
CURRENT_STATIC_TYPES = frozenset({"chronic_comorbidity", "aecopd_index_code",
                                  "acute_diagnosis", "disposition"})


def select_active_text_fields(base: IndependentData, active_fields: tuple[str, ...]) -> IndependentData:
    """保持字段词表和参数形状，只遮蔽未启用字段的片段引用。"""

    if not active_fields or len(set(active_fields)) != len(active_fields):
        raise ValueError("invalid active text field selection")
    unknown = set(active_fields) - set(base.field_roles)
    if unknown:
        raise ValueError("unknown active text fields: " + ", ".join(sorted(unknown)))
    active_ids = {base.field_roles.index(role) for role in active_fields}
    return replace(base, text_by_admission=tuple(
        tuple(ref for ref in refs if ref[0] in active_ids)
        for refs in base.text_by_admission))


def ablate_event_inputs(data: EventData, *, disable_medications: bool = False,
                        lab_process: str = "full",
                        excluded_concepts: tuple[str, ...] = ()) -> EventData:
    """保留事件词表；单次检验过程对照按概念合为一条平均值，时间设零。"""
    if lab_process not in {"full", "no_time", "one_per_concept"}:
        raise ValueError("unknown laboratory process ablation")
    if len(set(excluded_concepts)) != len(excluded_concepts) or set(excluded_concepts) - set(data.concept_names):
        raise ValueError("invalid excluded event concepts")
    excluded_ids = np.array([data.concept_names.index(name) + 1 for name in excluded_concepts], dtype=np.int64)
    values = data.event_values.copy()
    refs_by_admission = []
    for refs in data.events_by_admission:
        refs = refs[data.event_is_lab[refs]] if disable_medications else refs.copy()
        if len(excluded_ids):
            refs = refs[~np.isin(data.event_concepts[refs], excluded_ids)]
        if lab_process == "one_per_concept":
            keep = []
            for concept in np.unique(data.event_concepts[refs[data.event_is_lab[refs]]]):
                members = refs[(data.event_concepts[refs] == concept) & data.event_is_lab[refs]]
                observed = (members if data.event_numeric_mask is None else
                            members[data.event_numeric_mask[members]])
                first = int(observed[0] if len(observed) else members[0])
                values[first, 0] = float(values[observed, 0].mean()) if len(observed) else 0.0
                values[first, 1] = 0.0
                keep.append(first)
            keep.extend(int(ref) for ref in refs if not data.event_is_lab[ref])
            refs = np.asarray(keep, dtype=np.int64)
        elif lab_process == "no_time":
            values[refs[data.event_is_lab[refs]], 1] = 0.0
        refs_by_admission.append(refs)
    return replace(data, event_values=values, events_by_admission=tuple(refs_by_admission))


def _batches(sequences: list[np.ndarray], size: int) -> list[list[np.ndarray]]:
    return [sequences[start:start + size] for start in range(0, len(sequences), size)]


def deterministic_inference(function):
    """固定 CUDA 分段加和的推理结果；不改变训练阶段的算法开关。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            return function(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    return wrapped


@deterministic_inference
@torch.no_grad()
def predict(model: TemporalFusionModel, data: EventData, sequences: list[np.ndarray], *,
            process_only: bool, batch_size: int, device: torch.device,
            shuffle_lab_order_seed: int | None = None,
            return_all_causes: bool = False) -> tuple[np.ndarray, float]:
    model.eval()
    selected = np.concatenate(sequences)
    logits = np.zeros((len(data.base.hadm_ids), 2), dtype=np.float32)
    filled = np.zeros(len(data.base.hadm_ids), dtype=bool)
    loss_sum, count_sum = 0.0, 0
    for members in _batches(sequences, batch_size):
        batch = collate_events(data, members, include_text=True,
                               process_only=process_only,
                               shuffle_lab_order_seed=shuffle_lab_order_seed).to(device)
        output = model(batch)
        count = int(batch.base.valid.sum().item())
        loss_sum += float(competing_nll(output, batch.base).item()) * count
        count_sum += count
        scores = output.log_amplitude.cpu().numpy()
        for position, sequence in enumerate(members):
            logits[sequence] = scores[position, :len(sequence)]
            filled[sequence] = True
    if int(filled.sum()) != len(selected) or not filled[selected].all():
        raise ValueError("prediction row alignment failed")
    shape = torch.nn.functional.softplus(model.raw_shape).detach().cpu().numpy() + 0.1
    cumulative = predict_cif(logits[selected], shape, GRID)
    return (cumulative if return_all_causes else cumulative[:, :, 1]), loss_sum / count_sum


def _build_model_with_matched_text_initialization(model_class: type[TemporalFusionModel], *,
                                                  concept_count: int, type_count: int,
                                                  field_count: int, text_width: int,
                                                  shared_text_init_width: int | None,
                                                  event_arguments: dict[str, int],
                                                  device: torch.device) -> TemporalFusionModel:
    model = model_class(
        concept_count=concept_count, type_count=type_count,
        field_count=field_count,
        text_width=text_width if shared_text_init_width is None else shared_text_init_width,
        **event_arguments,
    )
    if shared_text_init_width is not None and shared_text_init_width != text_width:
        # Subclasses initialize event/lab/medication layers after the text projection.
        # Build them with the BERT-sized projection first, then replace only that layer.
        model.text_projection = torch.nn.Linear(text_width,
                                                model.text_projection.out_features)
    return model.to(device)


def fit(data: EventData, train: list[np.ndarray], validation: list[np.ndarray], *,
        process_only: bool, max_epochs: int, batch_size: int,
        device: torch.device, event_encoder: str = "gru", seed: int = 2026,
        shared_text_init_width: int | None = None,
        history_path: Path | None = None) -> tuple[TemporalFusionModel, list[dict[str, float | int]], int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_class = {"set": EventSetFusionModel, "gru": TemporalFusionModel,
                   "abnormal": AbnormalityFusionModel,
                   "abnormal_medication": MedicationEffectFusionModel}[event_encoder]
    event_arguments = {"event_concept_count": len(data.concept_names)} if event_encoder == "set" and data.concept_names != EVENT_NAMES else {}
    if data.recovery_by_admission is not None:
        if event_encoder == "set":
            event_arguments["clinical_recovery_width"] = data.recovery_by_admission.shape[1]
        elif event_encoder in ("abnormal", "abnormal_medication"):
            event_arguments["abnormality_width"] = data.recovery_by_admission.shape[1]
        else:
            raise ValueError("clinical features require set or abnormal encoder")
    elif event_encoder in ("abnormal", "abnormal_medication"):
        raise ValueError("abnormality encoder requires derived features")
    model = _build_model_with_matched_text_initialization(
        model_class,
        concept_count=len(data.base.concept_names), type_count=len(data.base.type_names),
        field_count=len(data.base.field_roles), text_width=data.base.text_width,
        shared_text_init_width=shared_text_init_width,
        event_arguments=event_arguments, device=device,
    )
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    rng = np.random.default_rng(seed)
    validation_indices = np.concatenate(validation)
    best_state = copy.deepcopy(model.state_dict())
    best_risk: np.ndarray | None = None
    best_epoch, best_score, stale = 0, math.inf, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = rng.permutation(len(train))
        loss_sum, count_sum = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            members = [train[int(index)] for index in shuffled[start:start + batch_size]]
            batch = collate_events(data, members, include_text=True,
                                   process_only=process_only).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = competing_nll(model(batch), batch.base)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite temporal training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch.base.valid.sum().item())
            loss_sum += float(loss.detach().item()) * count
            count_sum += count
        risk, val_nll = predict(model, data, validation, process_only=process_only,
                                batch_size=batch_size, device=device)
        evaluated = metrics(data.base.times[validation_indices],
                            data.base.events[validation_indices], risk)
        score = float(evaluated["integrated_brier_7_365"])
        record: dict[str, float | int] = {
            "epoch": epoch, "train_nll": loss_sum / count_sum,
            "validation_nll": val_nll, "validation_integrated_brier": score,
            "validation_mean_auc": float(evaluated["mean_auc_7_365"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if history_path is not None:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({"process_only": process_only, **record}), flush=True)
        if score < best_score - 1e-5:
            best_state, best_risk = copy.deepcopy(model.state_dict()), risk.copy()
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    assert best_risk is not None
    return model, history, best_epoch, best_risk


def run(aligned_dir: Path, structured_dir: Path, base_csv: Path, raw_dir: Path | None,
        field_dir: Path, vectors: Path, labels_csv: Path, output_dir: Path, *,
        process_only: bool, max_epochs: int, batch_size: int,
        event_encoder: str = "gru", processed_event_dir: Path | None = None,
        raw_ledger_dir: Path | None = None,
        exclude_prior_admission_counts: bool = False,
        disable_text: bool = False, disable_events: bool = False,
        disable_current_static: bool = False,
        disable_medications: bool = False,
        lab_process: str = "full",
        excluded_event_concepts: tuple[str, ...] = (),
        active_text_fields: tuple[str, ...] | None = None,
        clinical_recovery_dir: Path | None = None,
        clinical_recovery_mode: str = "off",
        abnormality_scale_mode: str = "none",
        medication_effect_mode: str = "off",
        shared_text_init_width: int | None = None,
        prepared_source: tuple[EventData, dict[str, object]] | None = None,
        prepared_exposure_only: np.ndarray | None = None,
        seed: int = 2026) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("full temporal experiment requires CUDA")
    if (processed_event_dir is None) == (raw_dir is None):
        raise ValueError("choose exactly one legacy raw source or processed event source")
    if processed_event_dir is not None:
        if raw_ledger_dir is None or event_encoder not in ("set", "abnormal", "abnormal_medication"):
            raise ValueError("processed events require the raw ledger and set/abnormal encoder")
        data, preprocessing = prepared_source if prepared_source is not None else load_processed_event_data(
            aligned_dir, structured_dir, base_csv, processed_event_dir, raw_ledger_dir,
            labels_csv=labels_csv, field_dir=field_dir, vectors_path=vectors,
        )
    else:
        if prepared_source is not None or prepared_exposure_only is not None:
            raise ValueError("prepared batch inputs require the checked processed source")
        data, preprocessing = load_event_data(
            aligned_dir, structured_dir, base_csv, raw_dir, labels_csv=labels_csv,
            field_dir=field_dir, vectors_path=vectors,
        )
    if clinical_recovery_mode not in ("off", "masked", "active", "measurement_only", "core9", "core_risk",
                                      *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)):
        raise ValueError("unknown clinical recovery mode")
    if clinical_recovery_mode != "off":
        if clinical_recovery_dir is None or processed_event_dir is None or event_encoder not in ("set", "abnormal", "abnormal_medication"):
            raise ValueError("clinical recovery requires derived features and processed events")
        if clinical_recovery_mode in ("measurement_only", "core9", "core_risk") and event_encoder not in ("abnormal", "abnormal_medication"):
            raise ValueError("control is specific to abnormality encoder")
        data = (attach_abnormalities(data, clinical_recovery_dir, mode=clinical_recovery_mode,
                                     scale_mode=abnormality_scale_mode)
                if event_encoder in ("abnormal", "abnormal_medication") else
                attach_recovery(data, clinical_recovery_dir, mode=clinical_recovery_mode))
    elif clinical_recovery_dir is not None:
        raise ValueError("clinical recovery directory supplied while mode is off")
    if event_encoder == "abnormal_medication":
        if processed_event_dir is None or medication_effect_mode == "off":
            raise ValueError("medication-effect encoder requires accepted eMAR and explicit mode")
        if prepared_exposure_only is None:
            data = attach_medication_effects(data, processed_event_dir, mode=medication_effect_mode)
        else:
            # 批量消融只复用已核验的二元给药；不得复用含频次／时间的缓存。
            exposure = prepared_exposure_only.reshape(len(data.base.hadm_ids), len(EFFECT_GROUPS), 6)
            if (medication_effect_mode != "exposure_only" or
                    not np.isin(exposure[:, :, 0], (0., 1.)).all() or
                    np.any(exposure[:, :, 1:] != 0.)):
                raise ValueError("cached medication input must contain only binary exposures")
            data = replace(data, medication_by_admission=prepared_exposure_only)
    elif medication_effect_mode != "off":
        raise ValueError("medication-effect mode requires abnormal_medication encoder")
    if event_encoder in ("abnormal", "abnormal_medication"):
        if clinical_recovery_mode == "off" or process_only or disable_events or disable_medications or lab_process != "full" or excluded_event_concepts:
            raise ValueError("abnormality-only encoder has no raw-event ablations and requires active/masked features")
        # 原始事件仍经来源契约校验，但绝不组成模型输入或梯度路径。
        data = replace(data, events_by_admission=tuple(
            np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if data.base.label_sha256 is None:
        raise ValueError("explicit validated outcome labels are required")
    if exclude_prior_admission_counts:
        if not set(PRIOR_COUNT_CONCEPTS).issubset(data.base.concept_names):
            raise ValueError("prior admission count concepts missing")
        data = replace(data, base=replace(data.base, disabled_concepts=PRIOR_COUNT_CONCEPTS))
    if disable_current_static:
        types = {i + 1 for i, name in enumerate(data.base.type_names)
                 if name in CURRENT_STATIC_TYPES}
        disabled = tuple(name for i, name in enumerate(data.base.concept_names, start=1)
                         if np.any((data.base.token_concepts == i) &
                                   np.isin(data.base.token_types, list(types))))
        if not disabled or not types:
            raise ValueError("current static concepts missing")
        data = replace(data, base=replace(data.base,
                                         disabled_concepts=data.base.disabled_concepts + disabled))
    # 保持完整模型及参数形状相同，只把指定模态的本次住院输入变为空。
    # 训练和评估均使用此数据，因此与冻结检查点遮蔽是不同实验。
    if disable_text:
        data = replace(data, base=replace(
            data.base, text_by_admission=tuple(() for _ in data.base.hadm_ids)))
    if active_text_fields is not None:
        if disable_text:
            raise ValueError("cannot select text fields while text is disabled")
        data = replace(data, base=select_active_text_fields(data.base, active_text_fields))
    if disable_events:
        data = replace(data, events_by_admission=tuple(
            np.empty(0, dtype=np.int64) for _ in data.base.hadm_ids))
    if disable_medications or lab_process != "full" or excluded_event_concepts:
        if disable_events or (process_only and lab_process != "full"):
            raise ValueError("overlapping event ablations")
        data = ablate_event_inputs(data, disable_medications=disable_medications,
                                   lab_process=lab_process,
                                   excluded_concepts=excluded_event_concepts)
    train = split_sequences(data.base, "train")
    validation = split_sequences(data.base, "validation")
    test = split_sequences(data.base, "test")
    device = torch.device("cuda")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    (output_dir / "status.json").write_text(json.dumps({"status": "training", "max_epochs": max_epochs,
                                                        "disable_text": disable_text, "disable_events": disable_events}), encoding="utf-8")
    if "contract_audit" in preprocessing:
        (output_dir / "model_data_audit.json").write_text(json.dumps(preprocessing["contract_audit"], indent=2), encoding="utf-8")
    model, history, best_epoch, val_risk = fit(
        data, train, validation, process_only=process_only,
        max_epochs=max_epochs, batch_size=batch_size, device=device,
        event_encoder=event_encoder, seed=seed,
        shared_text_init_width=shared_text_init_width,
        history_path=output_dir / "history.json",
    )
    curves: dict[str, dict[str, dict[str, object]]] = {}
    death_scores: dict[str, dict[str, object]] = {}
    for split, sequences in (("validation", validation), ("historical_test", test)):
        indices = np.concatenate(sequences)
        all_causes, _ = predict(
            model, data, sequences, process_only=process_only,
            batch_size=batch_size, device=device, return_all_causes=True)
        risk = all_causes[:, :, 1]
        if split == "validation":
            maximum_difference = float(np.max(np.abs(risk - val_risk)))
            (output_dir / "checkpoint_prediction_check.json").write_text(json.dumps({
                "maximum_absolute_difference": maximum_difference,
                "absolute_tolerance": 1e-6, "deterministic_inference": True,
                "passed": maximum_difference <= 1e-6,
            }, indent=2), encoding="utf-8")
            if maximum_difference > 1e-6:
                torch.save(model.state_dict(), output_dir / "diagnostic_best_model.pt")
                np.savez_compressed(output_dir / "diagnostic_validation_risks.npz",
                                    selected=val_risk, reproduced=risk)
                raise ValueError(f"best checkpoint risk mismatch: maximum difference={maximum_difference}")
        curves[split] = {"temporal_process_only" if process_only else "temporal_full":
                         metrics(data.base.times[indices], data.base.events[indices], risk)}
        swapped_events = np.where(data.base.events[indices] == 1, 2,
                                  np.where(data.base.events[indices] == 2, 1, 0))
        death_scores[split] = metrics(data.base.times[indices], swapped_events, all_causes[:, :, 2])
        # 留在本地供同患者配对比较，不向聊天输出患者级预测。
        np.savez_compressed(output_dir / f"{split}_predictions.npz",
                            subject_ids=data.base.subject_ids[indices],
                            hadm_ids=data.base.hadm_ids[indices],
                            times=data.base.times[indices], events=data.base.events[indices],
                            risk=risk, competing_cif=all_causes)
    report: dict[str, object] = {
        "status": "training_completed",
        "model": ("V5 static/text + abnormality and medication-effect summaries; causal patient Transformer; two-cause Weibull"
                  if event_encoder == "abnormal_medication" else
                  "V5 static/text + abnormality-only admission encoder; causal patient Transformer; two-cause Weibull"
                  if event_encoder == "abnormal" else
                  "V5 independent static and text branches + " + event_encoder + " event encoder; causal patient Transformer; two-cause Weibull"),
        "event_encoder": event_encoder,
        "self_supervised_pretraining": False,
        "primary_event": data.base.label_name,
        "ablation": (f"medication effect {medication_effect_mode}; abnormality {clinical_recovery_mode}"
                     if event_encoder == "abnormal_medication" else
                     "abnormality features masked" if event_encoder == "abnormal" and clinical_recovery_mode == "masked" else
                     "original nine axes active; thirteen added axes masked"
                     if event_encoder == "abnormal" and clinical_recovery_mode == "core9" else
                     "abnormality severities/recovery masked; measurement presence/span retained"
                     if event_encoder == "abnormal" and clinical_recovery_mode == "measurement_only" else
                     "abnormality features active; raw event sequence absent" if event_encoder == "abnormal" else
                     "event type/time/frequency without lab numeric values" if process_only else
                     "no event inputs" if disable_events else "full raw event sequence"),
        "disabled_modalities": [name for name, disabled in (("text", disable_text), ("events", disable_events)) if disabled],
        "disabled_static_concepts": data.base.disabled_concepts,
        "disabled_current_static": disable_current_static,
        "current_static_ablation_scope": (
            "All non-context static concepts, including both current and historical channels of chronic comorbidity; "
            "age, sex, explicit prior counts, gap, events and text retained."
            if disable_current_static else None),
        "disabled_medications": disable_medications,
        "excluded_event_concepts": excluded_event_concepts,
        "lab_process": lab_process,
        "ablation_scope": ("Explicit counts fully masked; patient history, gaps, diagnoses and text unchanged."
                           if exclude_prior_admission_counts else
                           "Explicit prior admission counts retained; all other inputs unchanged."),
        "cohort_admissions": int(len(data.base.hadm_ids)),
        "cohort_patients": int(len(data.base.sequences)),
        "field_manifest_sha256": data.base.manifest_sha256,
        "field_roles": data.base.field_roles,
        "active_text_fields": (() if disable_text else
                               data.base.field_roles if active_text_fields is None else active_text_fields),
        "label_sha256": data.base.label_sha256,
        "raw_source_sha256": data.source_hashes,
        "text_vector_width": data.base.text_width,
        "shared_text_init_width": (data.base.text_width if shared_text_init_width is None
                                   else shared_text_init_width),
        "deterministic_inference": True,
        "event_concept_count": len(data.concept_names),
        "event_concepts": data.concept_names,
        "raw_events_enter_model": event_encoder not in ("abnormal", "abnormal_medication") and not disable_events,
        "abnormality_feature_count": (data.recovery_by_admission.shape[1]
                                      if event_encoder in ("abnormal", "abnormal_medication") else None),
        "medication_effect_feature_count": (data.medication_by_admission.shape[1]
                                            if event_encoder == "abnormal_medication" else None),
        "preprocessed_events": processed_event_dir is not None,
        "clinical_recovery_mode": clinical_recovery_mode,
        "abnormality_scale_mode": abnormality_scale_mode,
        "abnormality_scaling": data.abnormality_scaling,
        "medication_effect_mode": medication_effect_mode,
        "medication_nonconstant_input_count": (len(EFFECT_GROUPS) if medication_effect_mode == "exposure_only"
                                                else None),
        "medication_effect_groups": tuple(EFFECT_GROUPS) if event_encoder == "abnormal_medication" else (),
        "clinical_recovery_manifest_sha256": (None if clinical_recovery_dir is None else
                                             hashlib.sha256((clinical_recovery_dir / "manifest.json").read_bytes()).hexdigest()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(device),
        "epochs_run": len(history),
        "max_epochs": max_epochs,
        "patience": 7,
        "seed": seed,
        "learning_rate": 8e-4,
        "weight_decay": 0.02,
        "patient_batch_size": batch_size,
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "selection": f"validation 7-365d integrated Brier; {max_epochs}-epoch cap and seven-epoch patience",
        "validation": {**{name: _public_metrics(detail) for name, detail in curves["validation"].items()},
                       "death_temporal_full": _public_metrics(death_scores["validation"])},
        "historical_test": {**{name: _public_metrics(detail) for name, detail in curves["historical_test"].items()},
                            "death_temporal_full": _public_metrics(death_scores["historical_test"])},
        "limitations": [
            "This is a within-MIMIC design experiment, not external validation.",
            "A qualifying COPD code denotes coded co-occurrence, not adjudicated COPD cause.",
            "Diagnosis and readmission outside MIMIC hospitals are not fully observed.",
            "The historical test split was previously examined and is not blind.",
            "365-day administrative censoring is the existing observation contract, not complete lifetime or out-of-hospital follow-up.",
        ],
    }
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_curves(output_dir, curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_nll"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_nll"] for row in history], label="validation")
    axes[0].set(ylabel="Negative log likelihood")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_integrated_brier"] for row in history])
    axes[1].set(ylabel="Validation integrated Brier")
    axes[2].plot(epochs, [row["validation_mean_auc"] for row in history])
    axes[2].set(ylabel="Validation mean AUC(t)")
    for axis in axes:
        axis.set(xlabel="Epoch")
        axis.axvline(best_epoch, linestyle="--", color="grey")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(figure)
    (output_dir / "status.json").write_text(json.dumps({"status": "training_completed"}), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned_dir", "structured_dir", "base_csv", "field_dir",
                 "vectors", "labels_csv", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--processed-event-dir", type=Path)
    parser.add_argument("--raw-ledger-dir", type=Path)
    parser.add_argument("--clinical-recovery-dir", type=Path)
    parser.add_argument("--clinical-recovery-mode", choices=("off", "masked", "active", "measurement_only", "core9", "core_risk",
                                                    *(f"without_{group}" for group in ABNORMALITY_ABLATIONS)), default="off")
    parser.add_argument("--abnormality-scale-mode", choices=("none", "training_positive_p95"), default="none")
    parser.add_argument("--medication-effect-mode", choices=("off", "active", "masked", "exposure_only", "core_acute", *(f"without_{group}" for group in EFFECT_ABLATIONS)), default="off")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--exclude-prior-admission-counts", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--active-text-fields", help="Comma-separated roles; fixed full field vocabulary is retained")
    parser.add_argument("--disable-events", action="store_true")
    parser.add_argument("--disable-current-static", action="store_true")
    parser.add_argument("--disable-medications", action="store_true")
    parser.add_argument("--exclude-event-concepts", help="Comma-separated existing event concepts to omit without changing vocabulary")
    parser.add_argument("--lab-process", choices=("full", "no_time", "one_per_concept"), default="full")
    parser.add_argument("--event-encoder", choices=("gru", "set", "abnormal", "abnormal_medication"), default="gru")
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--shared-text-init-width", type=int,
                        help="Match shared-layer initialization to this reference text width")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size, args.seed) < 1:
        raise ValueError("epochs, batch size, and seed must be positive")
    if args.shared_text_init_width is not None and args.shared_text_init_width < 1:
        raise ValueError("shared text initialization width must be positive")
    report = run(args.aligned_dir, args.structured_dir, args.base_csv, args.raw_dir,
                 args.field_dir, args.vectors, args.labels_csv, args.output_dir,
                 process_only=args.process_only, max_epochs=args.max_epochs,
                 batch_size=args.batch_size, event_encoder=args.event_encoder,
                 processed_event_dir=args.processed_event_dir, raw_ledger_dir=args.raw_ledger_dir,
                 exclude_prior_admission_counts=args.exclude_prior_admission_counts,
                 disable_text=args.disable_text, disable_events=args.disable_events,
                 disable_current_static=args.disable_current_static,
                 disable_medications=args.disable_medications, lab_process=args.lab_process,
                 clinical_recovery_dir=args.clinical_recovery_dir,
                 clinical_recovery_mode=args.clinical_recovery_mode,
                 abnormality_scale_mode=args.abnormality_scale_mode,
                 medication_effect_mode=args.medication_effect_mode,
                 excluded_event_concepts=(tuple(args.exclude_event_concepts.split(","))
                                          if args.exclude_event_concepts else ()),
                 active_text_fields=(tuple(args.active_text_fields.split(","))
                                     if args.active_text_fields is not None else None),
                 shared_text_init_width=args.shared_text_init_width,
                 seed=args.seed)
    print(json.dumps({"validation": report["validation"],
                      "historical_test": report["historical_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
