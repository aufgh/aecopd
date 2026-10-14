"""新进程读取已保存完整结果，独立核对训练选择/冻结/每日风险重放。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import parser,checked,prepare,new_model,ARMS
from training.v6.horizon_objective import restore
from training.v6.token_experiment import parameters
from training.v6.token_model import fast_cif,daily_probabilities
from training.v6.risk import predict_cif
from training.v6.train import write_json


def verify(root):
    cfg=json.loads((root/'config.json').read_text());r=json.loads((root/'report.json').read_text());contract=r['contract']
    for name,digest in contract['code_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('training code changed')
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,lab,med,prov,active,norm=checked(args)
    if d.provenance!=contract['data_provenance'] or prov!=contract['new_sources'] or active!=contract['active']:raise ValueError('lineage drift')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('validation','test')}
    cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ix};errors={};reference_error=0.
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1)
        for arm in ARMS:
            path=root/f'seed{seed}'/arm;history=json.loads((path/'history.json').read_text());cp=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
            if len(history)!=101 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=cp['epoch']:raise ValueError('selection/epoch')
            if sha256(path/'best.pt')!=r['models'][str(seed)][arm]['checkpoint_sha256']:raise ValueError('checkpoint changed')
            m=new_model(base,seed,arm);m.load_state_dict(cp['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('frozen changed')
            for fold in ix:
                a,k,nll=parameters(m,cache[fold]);curve=fast_cif(a,k,device='cuda');daily,conditional=daily_probabilities(curve)
                with np.load(path/(fold+'_predictions.npz')) as z:
                    for name in ('hadm_ids','subject_ids'):
                        if not np.array_equal(z[name],d.arrays[name][ix[fold]]):raise ValueError('saved member differs')
                    if not np.array_equal(z['events'],d.events[ix[fold]]) or not np.array_equal(z['times'],d.times[ix[fold]]):raise ValueError('labels differ')
                    err=float(np.abs(curve-z['curve']).max())
                    if err>3e-7 or np.abs(daily-z['daily_event_probability']).max()>3e-7:raise ValueError('GPU replay differs')
                    if np.any(np.diff(curve[:,:,1:],axis=1)<-1e-10) or np.abs(curve.sum(2)-1).max()>1e-10:raise ValueError('probability')
                    if not np.array_equal(z['conditional_daily_valid'],curve[:,:-1,0]>1e-12):raise ValueError('conditional validity')
                    errors[f'{seed}/{arm}/{fold}']=err
                reference_error=max(reference_error,float(np.abs(predict_cif(a[:5],k[:5],np.arange(366))-curve[:5]).max()))
            del m
    old=args.medication/'medication_identity.npz';new=args.expanded/'medications/medication_identity.npz'
    with np.load(old) as z:oldraw=z['values_raw'];oldpresent=z['present'];names=z['drug_names']
    with np.load(new) as z:newraw=z['values_raw'][:,:21];newpresent=z['present'][:,:21]
    corrections={str(name):dict(presence_changed_admissions=int((oldpresent[:,j]!=newpresent[:,j]).sum()),
          raw_changed_admissions=int(np.any(oldraw[:,j]!=newraw[:,j],axis=1).sum())) for j,name in enumerate(names)
          if np.any(oldraw[:,j]!=newraw[:,j]) or np.any(oldpresent[:,j]!=newpresent[:,j])}
    width_errors={};width=root/'molecule_width_control'
    if (width/'report.json').exists():
        protocol=json.loads((width/'protocol.json').read_text());width_report=json.loads((width/'report.json').read_text())
        if protocol['parent_report_sha256']!=sha256(root/'report.json') or protocol['code_sha256']!=sha256(Path('training/v6/medication_width_control.py')):
            raise ValueError('width control provenance changed')
        narrow=dict(med);narrow['features']=med['features'].copy();narrow['features'][:,21:]=0
        narrow['present']=med['present'].copy();narrow['present'][:,21:]=False
        small={f:prepare(d,c,ext,lab,narrow,ix[f]) for f in ix}
        for seed in (42,7):
            base=restore(args.parent,seed,'survival_base',h,.1);p=width/f'seed{seed}/existing21'
            cp=torch.load(p/'best.pt',map_location='cuda',weights_only=True);history=json.loads((p/'history.json').read_text())
            if len(history)!=101 or min(history,key=lambda x:x['validation_ibs'])['epoch']!=cp['epoch']:raise ValueError('width selection')
            if sha256(p/'best.pt')!=width_report['models'][str(seed)]['checkpoint_sha256']:raise ValueError('width checkpoint')
            m=new_model(base,seed,'more_meds');m.load_state_dict(cp['state_dict'])
            if not all(torch.equal(v,m.baseline.state_dict()[k]) for k,v in base.state_dict().items()):raise ValueError('width frozen')
            for fold in ix:
                a,k,_=parameters(m,small[fold]);curve=fast_cif(a,k,device='cuda')
                with np.load(p/(fold+'_predictions.npz')) as z:
                    if not np.array_equal(z['hadm_ids'],d.arrays['hadm_ids'][ix[fold]]):raise ValueError('width member')
                    error=float(np.abs(curve-z['curve']).max())
                    if error>3e-7:raise ValueError('width fresh-process replay')
                    width_errors[f'{seed}/{fold}']=error
            del m
    proof=dict(status='independently_verified_original_expansion',report_sha256=sha256(root/'report.json'),
        code_sha256=sha256(Path(__file__)),fresh_process_reload_errors=errors,max_reference_integral_error=reference_error,
        train_only_scaling_errors=norm,drug_mapping_corrections=corrections,models=12,folds=24,
        medication_width_control_reload_errors=width_errors,total_models=12+len(width_errors)//2,total_folds=24+len(width_errors),
        limits=['Clinical mapping has not been physician-adjudicated in full.', 'Reused test, not blind.'])
    write_json(root/'fresh_process_verification.json',proof)
    selected=r['validation_selected_arm']
    with np.load(root/f'seed42/{selected}/test_predictions.npz') as z:
        curve=z['curve'][0];daily=z['daily_event_probability'][0];conditional=z['conditional_daily_event_probability'][0];valid=z['conditional_daily_valid'][0]
    import csv
    with (root/'daily_selected_first_test_record.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['day','survival','readmission_cif','death_cif','readmission_daily','death_daily','readmission_given_event_free','death_given_event_free'])
        writer.writerow([0,*curve[0],0,0,0,0])
        for day in range(1,366):writer.writerow([day,*curve[day],*daily[day-1],*(conditional[day-1] if valid[day-1] else ('',''))])
    print(json.dumps(dict(status=proof['status'],max_replay_error=max(errors.values()),mapping_corrections=corrections)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True);verify(a.root)
