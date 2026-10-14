"""同一原V6/同初始化/同剂量映射，单独量化新增8药物而非全部细化效果。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import checked,prepare,new_model,evaluate
from training.v6.horizon_objective import restore
from training.v6.token_experiment import fit,parameters
from training.v6.token_model import fast_cif
from training.v6.evaluation import paired_interval
from training.v6.train import write_json


def run(root):
    output=root/'molecule_width_control'
    if output.exists():raise FileExistsError(output)
    report=json.loads((root/'report.json').read_text());cfg=json.loads((root/'config.json').read_text())
    proof=json.loads((root/'fresh_process_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('parent not independently verified')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,labs,med,prov,active,norm=checked(args)
    # 字典、数值尺度、药物身份纠错一致，只屏蔽新8个被选中的身份/值，不改槽位。
    med=dict(med);med['features']=med['features'].copy();med['features'][:,21:]=0
    med['present']=med['present'].copy();med['present'][:,21:]=False
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation','test')}
    cache={f:prepare(d,c,ext,labs,med,ix[f]) for f in ('train','validation')};output.mkdir()
    write_json(output/'protocol.json',dict(parent_report_sha256=sha256(root/'report.json'),code_sha256=sha256(Path(__file__)),
       comparison='parent more_meds29 minus existing21; same corrected dictionary/scales/slot architecture',
       seeds=[42,7],epochs=100,lr=1e-4,selection='validation IBS7-365',limits='Reused test, no multiplicity correction.'))
    fits={}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1);m=new_model(base,seed,'more_meds')
        fits[str(seed)]=fit(m,cache['train'],cache['validation'],d.times[ix['validation']],d.events[ix['validation']],
            output/f'seed{seed}/existing21',seed=seed,lr=1e-4,epochs=100);del m
    torch.use_deterministic_algorithms(True)
    del cache['train'];cache['test']=prepare(d,c,ext,labs,med,ix['test']);comparisons={};models={};errors={}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1);path=output/f'seed{seed}/existing21';m=new_model(base,seed,'more_meds')
        cp=torch.load(path/'best.pt',map_location='cuda',weights_only=True);m.load_state_dict(cp['state_dict'])
        if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen base')
        history=json.loads((path/'history.json').read_text())
        if len(history)!=101 or min(history,key=lambda r:r['validation_ibs'])['epoch']!=cp['epoch']:raise ValueError('selection')
        models[str(seed)]=dict(**fits[str(seed)],folds={});comparisons[str(seed)]={}
        for fold in ('validation','test'):
            detail,curve,_,_=evaluate(m,cache[fold],d,ix[fold],path/(fold+'_predictions.npz'));models[str(seed)]['folds'][fold]=detail
            fresh=new_model(base,seed,'more_meds');fresh.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            a,k,_=parameters(fresh,cache[fold]);err=float(np.abs(fast_cif(a,k,device='cuda')-curve).max());del fresh
            if err>3e-7:raise ValueError('fresh replay')
            errors[f'{seed}/{fold}']=err
            with np.load(root/f'seed{seed}/more_meds'/(fold+'_predictions.npz')) as z:
                if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix[fold]]):raise ValueError('parent member')
                comparisons[str(seed)][fold]=paired_interval(z['subject_ids'],z['times'],z['events'],z['curve'][:,:,1],curve[:,:,1],seed=seed)
        del m
    result=dict(status='completed_matched_21_vs_29_medication_control',models=models,comparisons=comparisons,reload_errors=errors,
       source_protocol_sha256=sha256(output/'protocol.json'))
    write_json(output/'report.json',result);print(json.dumps(dict(status=result['status'],test_comparisons={s:v['test'] for s,v in comparisons.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;run(a.root)
