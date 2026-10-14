"""保留不受字典修正影响的3臂，只重跑药物相关3臂×两种子。"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.expanded_original_experiment import checked,prepare,new_model
from training.v6.horizon_objective import restore
from training.v6.token_experiment import fit
from training.v6.train import write_json


def run(source,root,*,resume_setup=False):
    if root.exists():
        if not resume_setup or list(root.rglob('best.pt')) or list(root.rglob('history.json')):
            raise FileExistsError('not an empty failed copy setup: '+str(root))
        prior=root/'contract.json'
        if prior.exists():shutil.copy2(prior,root/'contract_copy_setup_failed.json')
    old=json.loads((source/'report.json').read_text());cfg=json.loads((source/'config.json').read_text());proof=json.loads((source/'fresh_process_verification.json').read_text())
    if proof['report_sha256']!=sha256(source/'report.json'):raise ValueError('parent not verified')
    cfg['expanded']='patient_data/processed/v6_expanded_candidates_2026-10-12_r3';cfg['output']=str(root)
    args=argparse.Namespace(**{k:Path(v) for k,v in cfg.items()});d,c,h,ext,lab,med,prov,active,norm=checked(args)
    if old['contract']['data_provenance']!=d.provenance:raise ValueError('cohort drift')
    # 修正仅涉及具体药物；原检验、原7类药效、BERT、基础参照不变。
    if old['contract']['new_sources']['expanded_labs.npz']!=prov['expanded_labs.npz']:raise ValueError('lab copy drift')
    root.mkdir(exist_ok=resume_setup);write_json(root/'config.json',cfg)
    contract=dict(old['contract']);contract.update(new_sources=prov,active=active,normalization_errors=norm,
       copied_unaffected_arms=['control','more_labs','interaction_only'],rerun_arms=['more_meds','both','both_interaction'],
       correction_source_report_sha256=sha256(source/'report.json'),
       correction='Existing umeclidinium/vilanterol was incorrectly rejected by new constituent matcher; corrected, affected arms retrained.')
    contract['code_sha256']=dict(contract['code_sha256'])
    for file in ('training/v6/repair_expanded_medication.py','patient_data/v6_expanded_medication_fix.py'):
        contract['code_sha256'][file]=sha256(Path(file))
    write_json(root/'contract.json',contract)
    for seed in (42,7):
        for arm in contract['copied_unaffected_arms']:
            # 只复制原权重和完整训练日志，所有折预测用修正后的共同接口重新生成。
            dst=root/f'seed{seed}/{arm}';dst.mkdir(parents=True,exist_ok=resume_setup)
            for file in ('best.pt','history.json'):shutil.copy2(source/f'seed{seed}/{arm}'/file,dst/file)
            if sha256(dst/'best.pt')!=old['models'][str(seed)][arm]['checkpoint_sha256']:raise ValueError('copied checkpoint changed')
    ix={f:np.flatnonzero(d.arrays['patient_splits']==f) for f in ('train','validation')}
    cache={f:prepare(d,c,ext,lab,med,ix[f]) for f in ix}
    for seed in (42,7):
        base=restore(args.parent,seed,'survival_base',h,.1)
        for arm in contract['rerun_arms']:
            m=new_model(base,seed,arm);fit(m,cache['train'],cache['validation'],d.times[ix['validation']],d.events[ix['validation']],
                 root/f'seed{seed}/{arm}',seed=seed,lr=1e-4,epochs=100);del m
    write_json(root/'status.json',dict(status='all_corrected_checkpoints_selected_before_test'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12'))
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_original_interaction_2026-10-12_r3'))
    p.add_argument('--resume-copy-setup',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;run(a.source,a.output,resume_setup=a.resume_copy_setup)
