"""额外检验与核心异常使用共同九通道；原派生和源资料不覆盖。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.v6_lab_state import summarize, scale_on_train, AXES
from patient_data.v6_detail_extension import LABS
from patient_data.eos_med_context import EOS_ITEMS, canonical_eos, finite

ROOT = Path('patient_data/processed/v6_unified_lab_state_2026-10-14')
SNAPSHOT = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE = Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EXTENSION = Path('patient_data/processed/v6_detail_extension_2026-10-11_r2')
NAMES = tuple(LABS) + ('eos_absolute', 'eos_percent')
EXTRA_NAMES = tuple(f'{name}_{direction}' for name in NAMES for direction in ('low', 'high'))


def abnormal(value, lower, upper, direction):
    if (direction not in ('low', 'high') or not np.isfinite([value, lower, upper]).all()
            or lower >= upper):
        raise ValueError('invalid laboratory reference/value')
    return max(0., ((lower-value) if direction == 'low' else (value-upper))/(upper-lower))


def build():
    if ROOT.exists(): raise FileExistsError('unified interface exists; do not overwrite')
    sm = json.loads((SNAPSHOT/'metadata/manifest.json').read_text())
    st = json.loads((STATE/'manifest.json').read_text())
    em = json.loads((EXTENSION/'manifest.json').read_text())
    sources = {}
    for name in ('raw/labevents', 'cohort/admissions'):
        p = SNAPSHOT/(name+'.csv.gz'); sources[str(p)] = sha256(p)
        if sources[str(p)] != sm['sources'][name]['sha256']: raise ValueError('raw source changed')
    for p, h in ((STATE/'clinical_state.npz', st['output_sha256']['clinical_state.npz']),
                 (EXTENSION/'extension.npz', em['output_sha256'])):
        sources[str(p)] = sha256(p)
        if sources[str(p)] != h: raise ValueError('derived source changed')
    with np.load(STATE/'clinical_state.npz') as z:
        a = {k:z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits', 'lab_axis_raw', 'lab_axis_inputs')}
    with np.load(EXTENSION/'extension.npz') as z:
        eos = z['eos']; np.testing.assert_array_equal(z['hadm_ids'], a['hadm_ids'])
    ids = {str(h):i for i,h in enumerate(a['hadm_ids'])}
    dates = {r['hadm_id']:(datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']))
             for r in rows(SNAPSHOT/'cohort/admissions.csv.gz') if r['hadm_id'] in ids}
    lookup = {item:(j, unit, None) for j,(item, unit) in enumerate(LABS.values())}
    lookup.update({item:(5 if kind=='absolute' else 6, None, kind) for item,kind in EOS_ITEMS.items()})
    samples = defaultdict(list); audit = Counter()
    for r in rows(SNAPSHOT/'raw/labevents.csv.gz'):
        if r['hadm_id'] not in ids or r['itemid'] not in lookup: continue
        j, unit, kind = lookup[r['itemid']]; audit[NAMES[j]+':candidate'] += 1
        try: t, s = datetime.fromisoformat(r['charttime']), datetime.fromisoformat(r['storetime'])
        except ValueError: audit['invalid_time'] += 1; continue
        start, end = dates[r['hadm_id']]
        if not start <= t <= s <= end: audit['outside_visible_interval'] += 1; continue
        v, lo, hi = (finite(r[k]) for k in ('valuenum', 'ref_range_lower', 'ref_range_upper'))
        if kind:
            v, lo, hi = (canonical_eos(x, r['valueuom'], kind) for x in (v,lo,hi))
        elif r['valueuom'] != unit:
            audit['invalid_unit'] += 1; continue
        if v is None or lo is None or hi is None or lo >= hi or v < 0 or ((kind=='percent' or unit=='%') and v > 100):
            audit[NAMES[j]+':invalid_value_or_reference'] += 1; continue
        day = (t-start).total_seconds()/86400
        for direction, delta in (('low',0), ('high',1)):
            samples[ids[r['hadm_id']], 2*j+delta].append((day, abnormal(v, lo, hi, direction)))
        audit[NAMES[j]+':accepted'] += 1
    n = len(ids); raw = np.zeros((n, len(AXES)+len(EXTRA_NAMES), 9))
    raw[:, :len(AXES)] = a['lab_axis_raw']
    for (i,j), values in samples.items():
        start,end = dates[str(a['hadm_ids'][i])]
        raw[i, len(AXES)+j], _ = summarize(values, (end-start).total_seconds()/86400)
    values, valid, scales = scale_on_train(raw, a['patient_splits'])
    np.testing.assert_allclose(values[:,:22], a['lab_axis_inputs'], atol=1e-6, rtol=0)
    # 八个激素时间类别仍在检验分支，低/高方向共用，不冒充实际药效。
    context = eos[:,:,12:20].reshape(n,16).astype(np.float32)
    ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'interface.npz', **{k:a[k] for k in ('hadm_ids','subject_ids','patient_splits')},
                        raw=raw, values=values, valid=valid, eos_context=context,
                        names=np.asarray((*AXES, *EXTRA_NAMES)))
    manifest = dict(status='unified_abnormality_interface_source_checked', source_sha256=sources,
        output_sha256=sha256(ROOT/'interface.npz'), names=list((*AXES,*EXTRA_NAMES)), channels=9,
        audit=dict(audit), train_coverage={name:int(raw[a['patient_splits']=='train',22+j,6].sum()) for j,name in enumerate(EXTRA_NAMES)},
        normalization=scales, no_outcome_used=True, no_high_frequency=True,
        limits='Extra/EOS missing valid reference is unobserved; original pH/pCO2 clinical definition retained.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], train_coverage=manifest['train_coverage'], audit=dict(audit))), flush=True)


if __name__ == '__main__': build()
