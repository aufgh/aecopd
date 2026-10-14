"""固定Cox诊断聚合图及后续常规去向组织可行性，不拟合模型。"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from training.v6.admission_cox_diagnostic import ROOT, PARENT, ARMS
from training.v6.data import disposition
from training.v6.train import write_json


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proof = json.loads((ROOT/'verification.json').read_text())
    if proof['report_sha256'] != sha256(ROOT/'report.json') or proof['maximum_error'] != 0:
        raise ValueError('prediction proof stale')
    report = json.loads((ROOT/'report.json').read_text())
    values = [report['results'][k]['metrics'] for k in ARMS]
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    labels = ['H34','H34 + availability','H34 + content']
    colors = ['#777777','#3274a1','#e1812c']
    for ax, measure, title, limits in (
        (axes[0], [m['readmission']['points']['90']['auc'] for m in values],
         'Readmission AUC90 (higher better)',(.75,.82)),
        (axes[1], [m['readmission']['integrated_brier_7_365'] for m in values],
         'Readmission IBS7-365 (lower better)',(.095,.105))):
        ax.scatter(measure,range(3),s=75,c=colors)
        ax.set_yticks(range(3),labels); ax.set_ylim(2.25,-.45); ax.set_xlim(limits)
        ax.set_title(title); ax.grid(axis='x',alpha=.2)
        for i,v in enumerate(measure): ax.annotate(f'{v:.6f}',(v,i),xytext=(6,8),textcoords='offset points')
    fig.suptitle('Fixed cause-specific Cox; same training/tuning split\nMinimal H34 inputs, not full V6; no confidence intervals')
    fig.tight_layout(); fig.savefig(ROOT/'conditional_information.png',dpi=180); plt.close(fig)

    # 下一条仅核验训练折原去向被粗映射折叠的真实类别，不按结局细分。
    with np.load(PARENT/'interface.npz',allow_pickle=False) as z:
        hadms=z['hadm_ids']; subjects=z['subject_ids']; folds=z['patient_splits']
    training = {str(h):int(s) for h,s,f in zip(hadms,subjects,folds) if f=='train'}
    manifest=json.loads((PARENT/'interface_manifest.json').read_text())
    p=next(Path(k) for k in manifest['sources_sha256'] if k.endswith('/cohort/admissions.csv.gz'))
    if sha256(p)!=manifest['sources_sha256'][str(p)]: raise ValueError('cohort changed')
    counts=Counter(); people=defaultdict(set); groups=defaultdict(Counter); seen=set()
    for row in rows(p):
        h=row['hadm_id']
        if h not in training: continue
        if h in seen or int(row['subject_id'])!=training[h]: raise ValueError('training alignment differs')
        seen.add(h); name=row['discharge_location']; counts[name]+=1; people[name].add(training[h])
        values,masks=disposition(name)
        group=('home','facility','other')[int(values.argmax())] if masks.any() else 'unknown'
        groups[group][name]+=1
    if seen!=set(training): raise ValueError('missing training rows')
    write_json(ROOT/'disposition_feasibility.json',dict(
        status='train_only_raw_category_collapse_audit_not_model_input',
        source_sha256={str(p):sha256(p)}, interface_sha256=sha256(PARENT/'interface.npz'),
        training_admissions=len(seen), raw_categories=dict(counts),
        patients_by_category={k:len(v) for k,v in people.items()},
        inherited_three_class_collapse={k:dict(v) for k,v in groups.items()},
        no_outcomes_used=True, new_fits=0, new_vectors=0))


if __name__=='__main__': run()
