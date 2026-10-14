"""固定扩展文本批次的单种子四臂比较；不修改原模型或正式队列。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from training.bonsai_broad_fragments import atomic_json
from training.v5.broad_fragment_data import (
    attach_broad_text, checked_selection, old_or_no_text, preserve_selected_histories, sha,
)
from training.v5.compare_screened_group_ablations import _summary_for_cause
from training.v5.independent_data import split_sequences
from training.v5.lab_recovery_data import attach_abnormalities
from training.v5.medication_effect_data import EFFECT_GROUPS, attach_medication_effects
from training.v5.medication_effect_model import MedicationEffectFusionModel
from training.v5.processed_event_data import load_processed_event_data
from training.v5.strict_endpoint_labels import STRICT_PRIMARY_EVENT
from training.v5.train import _save_curves, metrics
from training.v5.train_event_sequence import _build_model_with_matched_text_initialization, fit, predict

ARMS = ('no_text', 'old_fields_bert', 'fragments_bert', 'fragments_qwen')
PRIVATE = Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private')
SNAPSHOT = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
PROCESSED = Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
LABELS = Path('patient_data/derived/strict_aecopd_endpoint_2026-09-28/labels.csv.gz')


def build_model(data, device):
    return _build_model_with_matched_text_initialization(
        MedicationEffectFusionModel, concept_count=len(data.base.concept_names),
        type_count=len(data.base.type_names), field_count=len(data.base.field_roles),
        text_width=data.base.text_width, shared_text_init_width=768,
        event_arguments={'abnormality_width': data.recovery_by_admission.shape[1]}, device=device)


def initial_shared_hash(data, seed, device):
    torch.manual_seed(seed)
    model = build_model(data, device)
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith('text_projection.'):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    count = sum(p.numel() for p in model.parameters())
    del model
    return digest.hexdigest(), count


def train_arm(data, selection, output: Path, arm: str, seed: int, epochs: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    sequences = {s: split_sequences(data.base, s) for s in ('train', 'validation', 'test')}
    indices = {s: np.concatenate(rs) for s, rs in sequences.items()}
    initial_hash, parameter_count = initial_shared_hash(data, seed, device)
    started = time.perf_counter()
    atomic_json(output/'status.json', {'status': 'training', 'arm': arm})
    model, history, best_epoch, selected_risk = fit(
        data, sequences['train'], sequences['validation'], process_only=False,
        max_epochs=epochs, batch_size=12, device=device, event_encoder='abnormal_medication',
        seed=seed, shared_text_init_width=768, history_path=output/'history.json')
    torch.save(model.state_dict(), output/'model.pt')
    reloaded = build_model(data, device)
    reloaded.load_state_dict(torch.load(output/'model.pt', map_location=device, weights_only=True))
    reloaded.eval()
    curves, scores, note_scores = {}, {}, {}
    note_ids = {int(r['hadm_id']) for r in selection['members']}
    for split, key in (('validation', 'validation'), ('historical_test', 'test')):
        index = indices[key]
        cif, _ = predict(reloaded, data, sequences[key], process_only=False, batch_size=12,
                         device=device, return_all_causes=True)
        if split == 'validation':
            difference = float(np.max(np.abs(cif[:, :, 1]-selected_risk)))
            atomic_json(output/'checkpoint_prediction_check.json', {
                'passed': difference <= 1e-6, 'maximum_absolute_difference': difference,
                'absolute_tolerance': 1e-6, 'reloaded_from_saved_weights': True})
            if difference > 1e-6:
                raise ValueError('Reloaded best checkpoint differs from selected validation predictions')
        times, events = data.base.times[index], data.base.events[index]
        curves[split] = {arm: metrics(times, events, cif[:, :, 1])}
        scores[split] = {name: _summary_for_cause(times, events, cif, cause)
                         for cause, name in ((1, 'readmission'), (2, 'death'))}
        mask = np.isin(data.base.hadm_ids[index], list(note_ids))
        note_scores[split] = {name: _summary_for_cause(times[mask], events[mask], cif[mask], cause)
                              for cause, name in ((1, 'readmission'), (2, 'death'))}
        np.savez_compressed(output/f'{split}_predictions.npz',
            subject_ids=data.base.subject_ids[index], hadm_ids=data.base.hadm_ids[index],
            times=times, events=events, risk=cif[:, :, 1], competing_cif=cif, note_member=mask)
    _save_curves(output, curves)
    report = {'status': 'training_completed', 'arm': arm, 'seed': seed,
        'model': 'V5 static + lab abnormality + binary medication + gated text; causal patient Transformer; two-cause Weibull',
        'cohort_patients': len(data.base.sequences), 'cohort_admissions': sum(len(s) for s in data.base.sequences),
        'split_counts': {s: {'patients': len(sequences[s]), 'admissions': len(indices[s]),
            'readmissions': int((data.base.events[indices[s]] == 1).sum()),
            'deaths': int((data.base.events[indices[s]] == 2).sum())} for s in sequences},
        'primary_event': data.base.label_name, 'label_sha256': data.base.label_sha256,
        'member_sha256': selection['member_sha256'], 'field_manifest_sha256': data.base.manifest_sha256,
        'field_roles': data.base.field_roles, 'text_width': data.base.text_width,
        'text_vector_sha256': None if arm == 'no_text' else hashlib.sha256(np.asarray(data.base.text_vectors).tobytes()).hexdigest(),
        'shared_initialization_sha256': initial_hash, 'shared_text_init_width': 768,
        'parameter_count': parameter_count, 'epochs_run': len(history), 'best_epoch': best_epoch,
        'max_epochs': epochs, 'patience': 7, 'learning_rate': 8e-4, 'weight_decay': .02,
        'patient_batch_size': 12, 'device': torch.cuda.get_device_name(device),
        'selection': 'validation 7-365d integrated Brier', 'raw_events_enter_model': False,
        'clinical_recovery_axes': 22, 'medication_effect_mode': 'exposure_only; diuretic masked',
        'medication_nonconstant_inputs': 7,
        'old_text_restricted_to_same_note_members': True,
        'text_admissions_selected_histories': sum(bool(data.base.text_by_admission[i]) for ix in indices.values() for i in ix),
        'all_eligible_histories_retained': True, 'selected_note_histories_complete': True,
        'structural_scaling': 'existing full training fold static normalization and training-positive P95; identical frozen values across arms; no validation/test fitting',
        'validation': scores['validation'], 'historical_test': scores['historical_test'],
        'note_member_only': note_scores, 'seconds': time.perf_counter()-started,
        'limitations': ['Single seed; exploratory small training subset of 108 patients.',
            'Historical test previously examined; evaluation subset expanded after support audit, not a new blind test.',
            'Source comparison changes clinical content coverage and text roles; not proof of an isolated LLM causal contribution.',
            'Qwen and BERT have different frozen widths and projection parameter counts.',
            'World A observed 365-day administrative contract; no complete outside-hospital follow-up is proven.',
            'Source spans verified, not a gold-standard clinical extraction accuracy assessment.']}
    atomic_json(output/'report.json', report)
    atomic_json(output/'status.json', {'status': 'training_completed'})
    del model, reloaded
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-epochs', type=int, default=35)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    audit = json.loads((args.batch/'audit_2026-10-05/summary.json').read_text(encoding='utf-8'))
    if audit['status'] != 'source_provenance_checked_evaluation_ready':
        raise ValueError('Batch provenance/event support has not been checked')
    old = PRIVATE/'v5_section_aligned/field_text_baseline_2026_09_27'
    args.output.mkdir(parents=True)
    atomic_json(args.output/'status.json', {'status': 'checking_sources', 'planned_arms': ARMS})
    try:
        source, preprocessing = load_processed_event_data(
            PRIVATE/'v5_section_aligned/dataset_v3',
            PRIVATE/'v5_section_aligned/section_inputs_sparse_2026_09_26',
            PRIVATE/'v5_mimiciv_2026-09-25/structured/base.csv', PROCESSED, SNAPSHOT,
            labels_csv=LABELS, field_dir=old, vectors_path=old/'bert/bioclinicalbert.float16.npy')
        selection = checked_selection(source, args.batch/'selection.json')
        if (selection['member_sha256'] != audit['member_sha256'] or
                source.base.label_sha256 != audit['label_sha256'] or source.base.label_name != STRICT_PRIMARY_EVENT):
            raise ValueError('Audit membership/endpoint changed')
        source = attach_abnormalities(source, Path('patient_data/processed/abnormality_clinical_reviewed_2026-09-29'),
                                      mode='active', scale_mode='training_positive_p95')
        source = attach_medication_effects(source, PROCESSED, mode='exposure_only')
        medication = source.medication_by_admission.copy().reshape(len(source.base.hadm_ids), len(EFFECT_GROUPS), 6)
        medication[:, tuple(EFFECT_GROUPS).index('diuretic'), :] = 0
        if np.any(medication[:, :, 1:] != 0) or not np.isin(medication[:, :, 0], (0., 1.)).all():
            raise ValueError('Nonbinary/time medication input entered comparison')
        source = replace(source, medication_by_admission=medication.reshape(len(medication), -1),
                         events_by_admission=tuple(np.empty(0, dtype=np.int64) for _ in source.base.hadm_ids))
        source = preserve_selected_histories(source, selection)
        atomic_json(args.output/'preprocessing.json', preprocessing)
        structural_digest = hashlib.sha256()
        for array in (source.base.token_values, source.base.token_masks, source.base.token_concepts,
                      source.base.token_types, source.base.gap, source.recovery_by_admission,
                      source.medication_by_admission):
            structural_digest.update(np.asarray(array).tobytes())
        atomic_json(args.output/'contract.json', {'member_sha256': selection['member_sha256'],
            'label_sha256': source.base.label_sha256, 'primary_event': source.base.label_name,
            'structural_input_sha256': structural_digest.hexdigest(), 'source_hashes': source.source_hashes,
            'same_structural_object_for_all_arms': True, 'training_patients': 108,
            'field_vocabulary': (*source.base.field_roles, 'clinical_excerpt'),
            'raw_events_enter_model': False, 'medications_binary_only': True, 'diuretic_masked': True,
            'all_22_lab_axes_retained_no_unapproved_axis_deletion': True,
            'arms': ARMS, 'seed': args.seed, 'max_epochs': args.max_epochs})
        reports, expected_hash = {}, None
        for arm in ARMS:
            data = (old_or_no_text(source, selection, enabled=arm == 'old_fields_bert')
                    if arm in ('no_text', 'old_fields_bert') else
                    attach_broad_text(source, selection, args.batch/'encoding',
                                      'bert' if arm == 'fragments_bert' else 'qwen'))
            initial_hash, _ = initial_shared_hash(data, args.seed, torch.device('cuda'))
            if expected_hash is not None and initial_hash != expected_hash:
                raise ValueError('Shared non-text-projection weights differ between arms')
            expected_hash = initial_hash
            atomic_json(args.output/'status.json', {'status': 'training', 'current_arm': arm,
                                                   'completed_arms': list(reports)})
            reports[arm] = train_arm(data, selection, args.output/arm, arm, args.seed, args.max_epochs)
            print(json.dumps({'completed_arm': arm, 'validation': reports[arm]['validation']['readmission']}, ensure_ascii=False), flush=True)
        from training.v5.compare_broad_fragments import compare
        compare(args.output)
        atomic_json(args.output/'status.json', {'status': 'training_and_analysis_completed', 'completed_arms': list(reports)})
    except Exception as error:
        atomic_json(args.output/'status.json', {'status': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
