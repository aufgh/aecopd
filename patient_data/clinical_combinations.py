"""同采样CBC的原成分/NLR/SII；只读冻结底账，不读结局，训练折定尺度。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.v6_lab_state import summarize
from training.v5.numeric_preprocessing import robust_parameters

NAMES = ('neutrophil_fraction', 'lymphocyte_fraction', 'platelets_k_per_ul', 'nlr', 'sii_k_per_ul')
ITEMS = {'51256': 'npct', '51244': 'lpct', '52075': 'nabs', '51133': 'labs', '51265': 'plt'}


def ratios(neutrophils, lymphocytes, platelets=None):
    if not all(math.isfinite(x) and x >= 0 for x in (neutrophils, lymphocytes)) or lymphocytes <= 0:
        return None, None
    ratio = neutrophils/lymphocytes
    sii = platelets*ratio if platelets is not None and math.isfinite(platelets) and platelets >= 0 else None
    if not math.isfinite(ratio): return None, None
    if sii is not None and not math.isfinite(sii): sii = None
    return ratio, sii


def scale(raw, splits):
    out = raw.copy(); parameters = {}
    for a, name in enumerate(NAMES):
        for j in range(6):
            valid = raw[:, a, 6] > 0
            if j in (3, 5): valid = valid & (raw[:, a, 7] > 0)
            value = np.log1p(raw[:, a, j]) if j < 4 else raw[:, a, j]
            train = valid & (splits == 'train')
            if train.any():
                p = robust_parameters(value[train]); parameters[f'{name}/{j}'] = p
                out[:, a, j] = np.where(valid, np.clip((value-p['median'])/p['scale'], -8, 8), 0)
            else:
                parameters[f'{name}/{j}'] = {'training_observations': 0}
                out[:, a, j] = 0
    return out.astype(np.float32), parameters


def build(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    sm = json.loads((snapshot/'metadata/manifest.json').read_text())
    if sm['status'] != 'validated': raise ValueError('unvalidated snapshot')
    hashes = {}
    for name in ('raw/labevents', 'raw/d_labitems', 'cohort/admissions'):
        path = snapshot/(name+'.csv.gz'); hashes[name] = sha256(path)
        if hashes[name] != sm['sources'][name]['sha256']: raise ValueError('raw source changed')
    manifest = json.loads((state_dir/'manifest.json').read_text())
    if sha256(state_dir/'clinical_state.npz') != manifest['output_sha256']['clinical_state.npz']:
        raise ValueError('state changed')
    with np.load(state_dir/'clinical_state.npz') as z:
        ids = {k: z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    positions = {str(h): i for i, h in enumerate(ids['hadm_ids'])}
    adm = {r['hadm_id']: (datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']), r['subject_id'])
           for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in positions}
    labs = defaultdict(lambda: defaultdict(list)); counts = Counter()
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        if r['itemid'] not in ITEMS or r['hadm_id'] not in adm: continue
        start, end, subject = adm[r['hadm_id']]
        if r['subject_id'] != subject: raise ValueError('identity mismatch')
        try: t = datetime.fromisoformat(r['charttime']); value = float(r['valuenum'])
        except (ValueError, TypeError): counts['missing_numeric_or_time'] += 1; continue
        if not start <= t <= end or not math.isfinite(value) or value < 0: counts['invalid_time_value'] += 1; continue
        if '<' in r.get('value', '') or '>' in r.get('value', ''):
            counts['quantification_bound_excluded'] += 1; continue
        unit = r['valueuom'].strip().lower().replace(' ', '').replace('μ', 'u').replace('µ', 'u')
        key = ITEMS[r['itemid']]
        if (key.endswith('pct') and (unit != '%' or value > 100)) or (not key.endswith('pct') and unit not in ('k/ul', '10^3/ul')):
            counts['invalid_unit_or_percent'] += 1; continue
        specimen = r['specimen_id'].strip()
        if specimen in ('', '\\N'): counts['missing_specimen_excluded'] += 1; continue
        labs[(r['hadm_id'], t, specimen)][key].append(value)
    samples = defaultdict(list); pairing = Counter()
    for (h, t, specimen), values in labs.items():
        v = {k: float(np.median(x)) for k, x in values.items()}
        counts['duplicate_same_item_sample'] += sum(len(x)-1 for x in values.values())
        if 'npct' in v: samples[h, 0].append(((t-adm[h][0]).total_seconds()/86400, v['npct']/100))
        if 'lpct' in v: samples[h, 1].append(((t-adm[h][0]).total_seconds()/86400, v['lpct']/100))
        if 'plt' in v: samples[h, 2].append(((t-adm[h][0]).total_seconds()/86400, v['plt']))
        if 'nabs' in v and 'labs' in v:
            n, l, source = v['nabs'], v['labs'], 'absolute'
        elif 'npct' in v and 'lpct' in v:
            n, l, source = v['npct'], v['lpct'], 'percentage'
        else: pairing['no_paired_differential'] += 1; continue
        nlr, sii = ratios(n, l, v.get('plt'))
        day = (t-adm[h][0]).total_seconds()/86400
        if nlr is not None: samples[h, 3].append((day, nlr)); pairing[source+'_valid_nlr'] += 1
        else: pairing['zero_lymphocyte_denominator_excluded'] += 1
        if sii is not None: samples[h, 4].append((day, sii)); pairing[source+'_valid_sii'] += 1
    raw = np.zeros((len(positions), 5, 9))
    for (h, a), values in samples.items():
        duration = (adm[h][1]-adm[h][0]).total_seconds()/86400
        raw[positions[h], a] = summarize(values, duration)[0]
    # 原成分对照同时包含绝对分类的N/L，避免比值臂独享绝对数来源。
    # 百分比与绝对数不能相加混作同一概念；用另行可核验的absolute ratio输入替代会混淆。
    # 因此比值仅在同标本有百分比成分时使用，绝对源只用于配对一致性统计。
    pct_samples = defaultdict(list)
    for (h, t, specimen), values in labs.items():
        v = {k: float(np.median(x)) for k, x in values.items()}
        if 'npct' not in v or 'lpct' not in v: continue
        nlr, sii = ratios(v['npct'], v['lpct'], v.get('plt'))
        day = (t-adm[h][0]).total_seconds()/86400
        if nlr is not None: pct_samples[h, 3].append((day, nlr))
        if sii is not None: pct_samples[h, 4].append((day, sii))
    raw[:, 3:] = 0
    for (h, a), values in pct_samples.items(): raw[positions[h], a] = summarize(values, (adm[h][1]-adm[h][0]).total_seconds()/86400)[0]
    scaled, parameters = scale(raw, ids['patient_splits'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'combinations.npz', **ids, values=scaled, raw=raw, names=np.asarray(NAMES))
    report = dict(status='combinations_source_checked', source_hashes=hashes,
                  state_manifest_sha256=sha256(state_dir/'manifest.json'), names=NAMES, parameters=parameters,
                  counts=dict(counts), pairing=dict(pairing),
                  coverage={name: {'admissions': int((raw[:, j, 6] > 0).sum()),
                                  'patients': len(set(ids['subject_ids'][raw[:, j, 6] > 0].tolist())),
                                  'by_fold': {f: int(((ids['patient_splits']==f)&(raw[:, j, 6] > 0)).sum()) for f in ('train','validation','test')}} for j,name in enumerate(NAMES)},
                  formula='NLR=neutrophil_percent/lymphocyte_percent; SII=platelets(K/uL)*NLR; exact charttime AND specimen_id',
                  input_contract='5 axes x 9: first/max/latest/max-minus-latest/log_last_age/log_peak_gap/observed/recheck/latest_at_max; maxima are not clinical abnormality thresholds',
                  limits=['Not causal recovery or baseline inflammation; steroids/infection affect differential',
                          'No imputation of absent ratios; ratios restricted to same percentage-source components used by control',
                          'Exact same specimen/time is conservative, not all CBCs; no cross-day division'],
                  output_sha256=sha256(output/'combinations.npz'))
    (output/'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['coverage']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'output'): p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    a = p.parse_args(); build(a.snapshot, a.state_dir, a.output)
