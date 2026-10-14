"""对相同覆盖子集补充评价；不重选模型、不改变已保存预测。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.evaluation import metrics,public_metrics

def run(args):
    report=json.loads((args.run/'report.json').read_text()); config=json.loads((args.run/'config.json').read_text())
    with np.load(Path(config['cxr_dir'])/'cxr_trajectory.npz') as f: ids=f['hadm_ids']; masks={'cxr':f['cxr_present']}
    if config['facts_dir']:
        with np.load(Path(config['facts_dir'])/'facts.npz') as f:
            if not np.array_equal(ids,f['hadm_ids']): raise ValueError('members differ')
            masks.update(ct=f['ct_present'],discharge=f['discharge_present'])
    position={int(h):i for i,h in enumerate(ids)}; results={}
    for fold in ('validation','test'):
        results[fold]={}
        for arm in report['contract']['arms']:
            with np.load(args.run/arm/(fold+'_predictions.npz')) as f:
                index=np.array([position[int(h)] for h in f['hadm_ids']]); detail={}
                for channel,mask in masks.items():
                    selected=mask[index]
                    detail[channel]={'admissions':int(selected.sum()),'patients':len(set(f['subject_ids'][selected].tolist())),
                        'readmission':public_metrics(metrics(f['times'][selected],f['events'][selected],f['curve'][selected,:,1],ordering=True)),
                        'death':public_metrics(metrics(f['times'][selected],f['events'][selected],f['curve'][selected,:,2],cause=2,ordering=True))}
                results[fold][arm]=detail
    output={'status':'common_coverage_posthoc_no_reselection','report_sha256':sha256(args.run/'report.json'),'folds':results,
            'limits':'Coverage subsets exploratory; no clinician gold standard and no new independent test'}
    (args.run/'common_coverage.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':output['status'],'channels':list(masks)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); run(p.parse_args())
