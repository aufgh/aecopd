"""七类多来源药物、五类微生物及三类呼吸支持；只读底账，不训练。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from patient_data.export_mimiciv import rows, sha256
from patient_data.eos_med_context import SYSTEMIC_ROUTES
from patient_data.resp_micro_context import available_micro, invalid_micro, micro_scope, viral_positive
from patient_data.v6_lab_state import MED_GROUPS, AXES
from training.v5.clinical_concepts import medication_classes

MED_CHANNELS = ('execution_evidence', 'order_only_evidence', 'execution_source_observed', 'order_source_observed')
MICRO_NAMES = ('common_copd_bacteria', 'pseudomonas_aeruginosa', 'other_named_respiratory_bacteria',
               'respiratory_virus_positive', 'named_bacteria_resistance')
SUPPORT_NAMES = ('recorded_invasive_support', 'recorded_noninvasive_support', 'recorded_supplemental_oxygen')
INHALED_ROUTES = {'IH', 'NEB', 'INH', 'INHALATION', 'ORAL INHALATION'}
ORDER_SYSTEMIC_ROUTES = SYSTEMIC_ROUTES | {'IV BOLUS', 'SC', 'SUBCUT', 'SUBCUTANEOUS', 'OGT'}
OTHER_BACTERIA = {
    'STAPH AUREUS COAG +', 'STAPHYLOCOCCUS AUREUS', 'ESCHERICHIA COLI', 'KLEBSIELLA PNEUMONIAE',
    'KLEBSIELLA OXYTOCA', 'SERRATIA MARCESCENS', 'ENTEROBACTER CLOACAE', 'ENTEROBACTER CLOACAE COMPLEX',
    'ENTEROBACTER AEROGENES', 'ACINETOBACTER BAUMANNII COMPLEX', 'ACINETOBACTER LWOFFII',
    'STENOTROPHOMONAS MALTOPHILIA', 'PROTEUS MIRABILIS', 'PROVIDENCIA STUARTII', 'PROVIDENCIA RETTGERI',
    'MORGANELLA MORGANII', 'HAFNIA ALVEI', 'ACHROMOBACTER SP.', 'ACHROMOBACTER SPECIES',
    'ACHROMOBACTER XYLOSOXIDANS', 'ACHROMOBACTER (ALCALIGENES) DENTRIFICANS',
    'ELIZABETHKINGIA (CHRYSEOBACTERIUM) MENIGOSEPTICA', 'PSEUDOMONAS SPECIES', 'PSEUDOMONAS FLUORESCENS',
}
OXYGEN_DEVICES = {'NASAL CANNULA', 'HIGH FLOW NASAL CANNULA', 'NON-REBREATHER', 'VENTI MASK',
                  'OXYMIZER', 'MEDIUM CONC MASK'}


def parse_time(value):
    try: return datetime.fromisoformat(value)
    except (TypeError, ValueError): return None


def order_available(row, admission, *, entered=None, status=None):
    """仅接受出院前已录入且治疗区间与住院相交的医嘱；不把起始日期当录入日期。"""
    start, end = admission
    first = parse_time(row.get('starttime'))
    stop = parse_time(row.get('stoptime'))
    entered = parse_time(row.get('entertime')) if entered is None else entered
    status = row.get('status', '') if status is None else status
    if entered is None: return False, 'entry_time_unknown'
    if entered > end: return False, 'entered_after_discharge'
    if status.strip().upper() in {'H', 'HOLD', 'HELD', 'CANCELLED', 'CANCELED', 'VOID', 'REJECTED'}:
        return False, 'held_or_cancelled_order'
    if first is None: return False, 'start_time_unknown'
    if row.get('stoptime') not in (None, '', r'\N') and stop is None: return False, 'invalid_stop_time'
    if stop is not None and stop < first: return False, 'negative_order_interval'
    if first > end or (stop is not None and stop < start): return False, 'outside_admission'
    return True, 'available_order'


def order_groups(name, route, descriptions=()):
    """剂型/途径仅上游判定；复方可映射多个药效，未知途径不伪造系统治疗。"""
    route = route.strip().upper()
    descriptions = tuple(descriptions) + (('inhalation',) if route in INHALED_ROUTES else ())
    classes = medication_classes(name, routes=(route,), product_descriptions=descriptions,
                                 mapping_version='clinical_reviewed_2026_09')
    result = set()
    for j, (_, constituents) in enumerate(MED_GROUPS.items()):
        if not classes.intersection(constituents): continue
        allowed = INHALED_ROUTES if j < 5 else ORDER_SYSTEMIC_ROUTES
        if route in allowed: result.add(j)
    return result


def bacteria_group(name):
    name = name.strip().upper()
    if (name in {'STREPTOCOCCUS PNEUMONIAE', 'MORAXELLA CATARRHALIS', 'HAEMOPHILUS INFLUENZAE'}
            or name.startswith('HAEMOPHILUS INFLUENZAE,')): return 0
    if name == 'PSEUDOMONAS AERUGINOSA': return 1
    if name in OTHER_BACTERIA: return 2
    return None


def compact_micro_scope(row):
    """真菌/分枝杆菌/军团菌专项培养，不能证明常规细菌组被检查。"""
    scope = micro_scope(row)
    test = row['test_name'].strip().upper()
    if scope == 'lower':
        if test == 'RESPIRATORY CULTURE': return 'bacteria'
        if 'RESPIRATORY VIRAL' in test or 'INFLUENZA' in test: return 'virus'
        return 'excluded_targeted_or_nonculture_test'
    return 'virus' if scope == 'resp_viral' else 'other'


def compact_support(support, support_masks, numeric, numeric_masks, device_rows):
    n = len(support)
    values = np.zeros((n, 3), np.float32)
    masks = np.zeros((n, 3), bool)
    # 气管导管是支持证据，不声称急性COPD适应证；气切单独存在不算在通气。
    values[:, 0] = np.maximum(support[:, 0], support[:, 2])
    masks[:, 0] = support_masks[:, 0] | support_masks[:, 2]
    values[:, 1] = np.maximum(support[:, 1], support[:, 3])
    masks[:, 1] = support_masks[:, 1] | support_masks[:, 3]
    # 氧流量/FIO2只用于上游证据，不产生新时间/剂量输入。
    for k in (2, 3, 4):
        masks[:, 2] |= numeric_masks[:, k, 2]
        values[:, 2] = np.maximum(values[:, 2], (numeric_masks[:, k, 2] & (numeric[:, k, 2] > 0)).astype(np.float32))
    masks[:, 2] |= numeric_masks[:, 5, 2]
    values[:, 2] = np.maximum(values[:, 2], (numeric_masks[:, 5, 2] & (numeric[:, 5, 2] > 21)).astype(np.float32))
    for i, item, raw in device_rows:
        name = raw.strip().upper()
        if item == '226732':
            if name not in ('', r'\N', 'OTHER'): masks[i, 2] = True
            if name in OXYGEN_DEVICES: values[i, 2] = 1
        if item == '229314' and name in {'NIV', 'NIV-ST'}:
            values[i, 1] = 1; masks[i, 1] = True
    # CPAP/PSV、CPAP面罩和呼吸机品牌不能独立证明急性NIV。
    return values, masks


def load_checked(directory, filename, status):
    meta = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if meta['status'] != status: raise ValueError('unfinished parent: '+str(directory))
    for name, digest in meta['output_sha256'].items():
        if sha256(directory/name) != digest: raise ValueError('parent file changed: '+name)
    with np.load(directory/filename, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, meta


def build(snapshot, supplement, state_dir, eos_dir, resp_micro_dir, output):
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter()
    state, st = load_checked(state_dir, 'clinical_state.npz', 'derived_and_roundtrip_checked_not_prediction_validated')
    eos, em = load_checked(eos_dir, 'eos_med_context.npz', 'eos_med_context_validated_not_trained')
    rm, mm = load_checked(resp_micro_dir, 'resp_micro_context.npz', 'resp_micro_context_validated_not_trained')
    sm = json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    sup = json.loads((supplement/'metadata/manifest.json').read_text(encoding='utf-8'))
    if (sm['status'] != 'validated' or sup['status'] != 'validated_restricted_source_export_not_model_validated'
            or not sup['cohort_membership_exactly_matches_parent']
            or sha256(snapshot/'metadata/manifest.json') != st['source_manifest_sha256']
            or sha256(snapshot/'metadata/manifest.json') != sup['parent_snapshot_manifest_sha256']
            or sha256(supplement/'metadata/manifest.json') != mm['parent_supplement_manifest_sha256']):
        raise ValueError('raw/state lineage mismatch')
    parent_hash = sha256(state_dir/'manifest.json')
    for arrays, meta in ((eos, em), (rm, mm)):
        if meta.get('parent_state_manifest_sha256', meta.get('parent_clinical_state_manifest_sha256')) != parent_hash:
            raise ValueError('derived lineage mismatch')
        for key in ('hadm_ids', 'subject_ids', 'patient_splits'):
            if not np.array_equal(arrays[key], state[key]): raise ValueError('member/order/split mismatch')
    if tuple(state['axis_names']) != AXES or tuple(state['med_group_names']) != tuple(MED_GROUPS):
        raise ValueError('vocabulary drift')
    if not np.array_equal(state['med_exposure'], eos['med_exposure']): raise ValueError('execution exposure drift')
    source_hashes = {}
    for name in ('raw/pharmacy', 'raw/prescriptions', 'cohort/admissions'):
        path = snapshot/(name+'.csv.gz'); digest = sha256(path)
        if digest != sm['sources'][name]['sha256']: raise ValueError('raw changed: '+name)
        source_hashes[str(path)] = digest
    for name in ('microbiologyevents', 'respiratory_chartevents'):
        path = supplement/'raw'/(name+'.csv.gz'); digest = sha256(path)
        if digest != sup['sources'][name]['sha256']: raise ValueError('supplement changed: '+name)
        source_hashes[str(path)] = digest
    hadms = [str(h) for h in state['hadm_ids']]; positions = {h: i for i, h in enumerate(hadms)}
    members = {r['hadm_id']: r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admissions = {h: (parse_time(members[h]['admittime']), parse_time(members[h]['dischtime'])) for h in hadms}
    if len(hadms) != 19401 or len(positions) != 19401: raise ValueError('cohort drift')
    for h, i in positions.items():
        if int(members[h]['subject_id']) != state['subject_ids'][i]: raise ValueError('identity drift')
    def checked_rows(path, expected_count):
        count = 0
        for row in rows(path):
            count += 1
            h = row['hadm_id']
            if h not in members or row['subject_id'] != members[h]['subject_id']: raise ValueError('raw association mismatch')
            if h in positions: yield row
        if count != expected_count: raise ValueError('row count changed: '+str(path))
    n = len(hadms); audit = Counter()
    ordered = np.zeros((n, 7), np.float32); pharma_flag = np.zeros_like(ordered)
    rx_flag = np.zeros_like(ordered); rx_unverified = np.zeros_like(ordered)
    order_source = np.zeros(n, bool); pharma = {}; unique_orders = set()
    for r in checked_rows(snapshot/'raw/pharmacy.csv.gz', sm['sources']['raw/pharmacy']['rows']):
        h = r['hadm_id']; i = positions[h]; pid = r['pharmacy_id']
        if pid in pharma: raise ValueError('duplicate pharmacy_id')
        pharma[pid] = (h, parse_time(r['entertime']), r['status'], r['route'])
        ok, reason = order_available(r, admissions[h]); audit['pharmacy:'+reason] += 1
        if not ok: continue
        order_source[i] = True
        groups = order_groups(r['medication'], r['route'])
        if not groups: audit['pharmacy:no_target_class_or_incompatible_route'] += 1
        for j in groups:
            pharma_flag[i, j] = 1; ordered[i, j] = 1
            unique_orders.add((h, pid, j))
    for r in checked_rows(snapshot/'raw/prescriptions.csv.gz', sm['sources']['raw/prescriptions']['rows']):
        h = r['hadm_id']; i = positions[h]; linked = pharma.get(r['pharmacy_id'])
        if linked is not None and linked[0] != h: raise ValueError('prescription/pharmacy association mismatch')
        descriptions = (r['prod_strength'], r['form_rx'])
        route = r['route'] if r['route'] not in ('', r'\N') else linked[3] if linked else ''
        groups = order_groups(r['drug'], route, descriptions)
        if linked is None:
            # 未关联处方没有录入时间，仅起始日期不足以证明出院前可见；仅审计，不进入主输入。
            audit['prescription:unlinked_entry_time_unknown'] += 1
            for j in groups: rx_unverified[i, j] = 1
            continue
        ok, reason = order_available(r, admissions[h], entered=linked[1], status=linked[2])
        audit['prescription:'+reason] += 1
        if not ok: continue
        order_source[i] = True
        if not groups: audit['prescription:no_target_class_or_incompatible_route'] += 1
        for j in groups:
            rx_flag[i, j] = 1; ordered[i, j] = 1
            key = (h, r['pharmacy_id'], j)
            if key in unique_orders: audit['order_class_duplicate_collapsed'] += 1
            unique_orders.add(key)
    med = np.zeros((n, 7, 4), np.float32)
    med[:, :, 0] = state['med_exposure']
    med[:, :, 1] = ordered * (1 - state['med_exposure'])
    med[:, :, 2] = eos['med_source_mask']
    med[:, :, 3] = order_source[:, None]
    micro = np.zeros((n, 5), np.float32); micro_masks = np.zeros_like(micro, bool)
    lower_available = np.zeros(n, bool); viral_available = np.zeros(n, bool)
    isolates = set(); sir_isolates = set(); micro_ids = set()
    for r in checked_rows(supplement/'raw/microbiologyevents.csv.gz', sup['sources']['microbiologyevents']['rows']):
        if r['microevent_id'] in micro_ids: raise ValueError('duplicate microevent_id')
        micro_ids.add(r['microevent_id']); scope = compact_micro_scope(r)
        if scope not in ('bacteria', 'virus'):
            audit['micro:'+scope] += 1; continue
        ok, reason = available_micro(r, *admissions[r['hadm_id']])
        audit['micro:'+reason] += 1
        if not ok: continue
        if invalid_micro(r): audit['micro:invalid_or_cancelled'] += 1; continue
        i = positions[r['hadm_id']]
        if scope == 'bacteria':
            lower_available[i] = True
            j = bacteria_group(r['org_name'])
            if j is None: audit['micro:unmapped_organism_rows'] += 1; continue
            micro[i, j] = 1
            key = tuple(r[k] for k in ('hadm_id', 'micro_specimen_id', 'test_seq', 'test_itemid', 'org_name', 'isolate_num'))
            if key in isolates: audit['micro:duplicate_isolate_rows_collapsed'] += 1
            isolates.add(key)
            if r['interpretation'] in ('S', 'I', 'R'):
                micro_masks[i, 4] = True; sir_isolates.add(key)
                if r['interpretation'] == 'R': micro[i, 4] = 1
        else:
            viral_available[i] = True
            if viral_positive(r['org_name'], r['comments']): micro[i, 3] = 1
    micro_masks[:, :3] = lower_available[:, None]; micro_masks[:, 3] = viral_available
    device_rows = []
    for r in checked_rows(supplement/'raw/respiratory_chartevents.csv.gz', sup['sources']['respiratory_chartevents']['rows']):
        h = r['hadm_id']; t = parse_time(r['charttime']); stored = parse_time(r['storetime'])
        if t is None or stored is None or not admissions[h][0] <= t <= stored <= admissions[h][1]:
            raise ValueError('respiratory source availability drift')
        if r['selection_role'] == 'state_record': device_rows.append((positions[h], r['itemid'], r['value']))
    support, support_masks = compact_support(rm['support_values'], rm['support_masks'],
                                             rm['resp_numeric_raw'], rm['resp_numeric_masks'], device_rows)
    for values, masks in ((micro, micro_masks), (support, support_masks)):
        if not np.isin(values, (0, 1)).all() or ((values > 0) & ~masks).any(): raise ValueError('invalid evidence/mask')
    if ((med[:, :, 0] > 0) & (med[:, :, 2] == 0)).any() or ((med[:, :, 1] > 0) & (med[:, :, 3] == 0)).any():
        raise ValueError('medication positive without source')
    if (med[:, :, 0] * med[:, :, 1]).any(): raise ValueError('execution and order-only overlap')
    def coverage(present):
        return {'admissions': int(present.sum()), 'patients': len(set(state['subject_ids'][present].tolist()))}
    def evidence(values, masks):
        return {'positive': coverage(values > 0), 'source_observed': coverage(masks), 'unknown_admissions': int((~masks).sum())}
    report = {'status': 'compact_multisource_validated_not_trained', 'admissions': n,
              'patients': len(set(state['subject_ids'].tolist())), 'lab_axes_kept': list(AXES),
              'parent_state_manifest_sha256': parent_hash,
              'parent_eos_manifest_sha256': sha256(eos_dir/'manifest.json'),
              'parent_resp_micro_manifest_sha256': sha256(resp_micro_dir/'manifest.json'),
              'source_sha256': source_hashes, 'code_sha256': sha256(Path(__file__)), 'audit': dict(audit),
              'med_group_names': list(MED_GROUPS), 'med_channel_names': MED_CHANNELS,
              'micro_names': MICRO_NAMES, 'support_names': SUPPORT_NAMES,
              'med_coverage': {g: {'execution': coverage(med[:, j, 0] > 0),
                                  'order_only': coverage(med[:, j, 1] > 0),
                                  'pharmacy': coverage(pharma_flag[:, j] > 0),
                                  'linked_prescription': coverage(rx_flag[:, j] > 0)} for j, g in enumerate(MED_GROUPS)},
              'med_any_coverage': {'execution': coverage(med[:, :, 0].any(axis=1)),
                                   'order_only': coverage(med[:, :, 1].any(axis=1)),
                                   'execution_or_order': coverage(med[:, :, :2].any(axis=(1, 2))),
                                   'no_emar_with_order': coverage(~eos['med_source_mask'].any(axis=1) & ordered.any(axis=1)),
                                   'order_source_observed': coverage(order_source),
                                   'unlinked_rx_audit_only': coverage(rx_unverified.any(axis=1))},
              'micro_coverage': {g: evidence(micro[:, j], micro_masks[:, j]) for j, g in enumerate(MICRO_NAMES)},
              'support_coverage': {g: evidence(support[:, j], support_masks[:, j]) for j, g in enumerate(SUPPORT_NAMES)},
              'unique_order_classes': len(unique_orders), 'unique_controlled_bacterial_isolates': len(isolates),
              'bacterial_isolates_with_SIR': len(sir_isolates),
              'limitations': ['Order evidence is not administration, dose, discharge medication or respiratory indication',
                             'Source-observed mask does not establish complete treatment coverage',
                             'Unlinked prescription without entered time retained only for audit',
                             'Controlled positive culture is not proof of active infection or colonization',
                             'Resistance means at least one R result, not multidrug resistance',
                             'Respiratory support is ICU recorded evidence; acute indication and non-ICU absence unknown',
                             'CPAP alone is not NIV; tracheostomy alone is not invasive ventilation',
                             'No medication time/count/dose or support duration enters the compact interface',
                             'All 22 lab axes retained unchanged for future group ablation; no training this run']}
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    filename = 'compact_context.npz'
    arrays = {k: state[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits', 'axis_names')}
    arrays.update(med_features=med, med_group_names=np.asarray(tuple(MED_GROUPS)), med_channel_names=np.asarray(MED_CHANNELS),
                  med_pharmacy_audit=pharma_flag, med_linked_rx_audit=rx_flag, med_unlinked_rx_audit=rx_unverified,
                  micro_values=micro, micro_masks=micro_masks, micro_names=np.asarray(MICRO_NAMES),
                  support_values=support, support_masks=support_masks, support_names=np.asarray(SUPPORT_NAMES))
    np.savez_compressed(output/filename, **arrays)
    with np.load(output/filename, allow_pickle=False) as archive:
        for key, value in arrays.items():
            if not np.array_equal(archive[key], value): raise ValueError('roundtrip failed: '+key)
    report['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    report['output_sha256'] = {filename: sha256(output/filename)}
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'admissions', 'med_any_coverage', 'micro_coverage', 'support_coverage', 'elapsed_seconds')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot', 'supplement', 'state_dir', 'eos_dir', 'resp_micro_dir', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    build(args.snapshot, args.supplement, args.state_dir, args.eos_dir, args.resp_micro_dir, args.output)
