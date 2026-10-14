"""完整官方读出层补正；旧骨干/旧记录不改，复用确实未受影响的训练。"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_model import PublicModel,ARMS,initial_bias,ensemble_curve
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT as FIRST,SEEDS,LRS,setup,tensors,fit,parameters,render
from training.v6.audited_experiment import OUTPUT as AUDITED,save_curve,summary
from training.v6.evaluation import paired_interval
from training.v6.train import write_json

ROOT=FIRST/'official_readout'


class CorrectedModel(PublicModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.readout=(nn.Sequential(nn.BatchNorm1d(256),nn.ReLU()) if self.arm.startswith('resnet') else
            nn.Sequential(nn.LayerNorm(96),nn.ReLU()) if self.arm=='ft_transformer' else nn.Identity())

    def forward(self,x_num,x_binary):
        if self.arm in ('ft_transformer','tabm_ple'):
            h=self.backbone(x_num,x_binary.long() if x_binary.shape[1] else None)
        else:
            x=self.embedding(x_num).flatten(1) if self.embedding is not None else x_num
            h=self.backbone(torch.cat((x,x_binary),1))
        with torch.autocast(h.device.type,enabled=False):
            raw=self.head(self.readout(h.float()))
        if raw.ndim==2:raw=raw[:,None]
        return raw[...,:2].clamp(-10.,6.),torch.nn.functional.softplus(raw[...,2:])+.1


def model_for(arm,seed,data,prep):
    torch.manual_seed(seed)
    model=CorrectedModel(arm,data['x_num'].shape[1],data['x_binary'].shape[1],
        [torch.as_tensor(b) for b in prep['bins']],initial_bias(data['events'][data['patient_splits']=='train'])).cuda()
    torch.manual_seed(seed);return model


def run():
    setup();start=time.perf_counter()
    if ROOT.exists():raise FileExistsError(ROOT)
    original=json.loads((FIRST/'report.json').read_text())
    contract=original['contract']
    for p,digest in contract['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('first-pass source changed')
    ROOT.mkdir()
    for file in ('matrix.npz','preprocessing.joblib','data_contract.json'):
        shutil.copy2(FIRST/file,ROOT/file)
    d,prep=read(ROOT);_,train=tensors(d,'train');vix,val=tensors(d,'validation')
    contract=dict(contract,phase='complete_official_readout',new_trainings=12,reused_trainings=8,
        first_report_sha256=sha256(FIRST/'report.json'),
        correction_protocol_sha256=sha256(Path('docs/V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md')),
        source_sha256={**contract['source_sha256'],str(Path(__file__)):sha256(Path(__file__))})
    write_json(ROOT/'contract.json',contract);fits={}
    for arm in ARMS:
        fits[arm]={}
        for lr in LRS:
            fits[arm][str(lr)]={}
            for seed in SEEDS:
                path=ROOT/arm/f'lr{lr}'/f'seed{seed}'
                if arm in ('mlp','tabm_ple'):
                    oldpath=FIRST/arm/f'lr{lr}'/f'seed{seed}'
                    result=json.loads((oldpath/'training_result.json').read_text())
                    if result['completed_epochs']!=100 or sha256(oldpath/'best.pt')!=result['checkpoint_sha256']:
                        raise ValueError('unaffected checkpoint mismatch')
                    path.mkdir(parents=True)
                    for file in ('best.pt','history.json','training_result.json'):shutil.copy2(oldpath/file,path/file)
                    result=dict(result,reused_from=str(oldpath),new_training=False)
                    print(json.dumps(dict(reused=arm,seed=seed,lr=lr)),flush=True)
                else:
                    model=model_for(arm,seed,d,prep);result=fit(model,train,val,d,vix,path,seed,lr,100)
                    result['new_training']=True;del model;torch.cuda.empty_cache()
                fits[arm][str(lr)][str(seed)]=result
    choices={arm:min(LRS,key=lambda lr:np.mean([fits[arm][str(lr)][str(s)]['validation_ibs'] for s in SEEDS])) for arm in ARMS}
    write_json(ROOT/'selected_before_test.json',dict(fits=fits,learning_rates=choices,all_training_complete=True))
    del train;torch.cuda.empty_cache();tix,test=tensors(d,'test');models={};comparisons={}
    from types import SimpleNamespace
    holder=SimpleNamespace(times=d['times'],events=d['events'],arrays=d)
    for arm in ARMS:
        models[arm]={};comparisons[arm]={}
        for seed in SEEDS:
            lr=choices[arm];path=ROOT/arm/f'lr{lr}'/f'seed{seed}';model=model_for(arm,seed,d,prep)
            model.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            models[arm][str(seed)]=dict(**fits[arm][str(lr)][str(seed)],folds={});comparisons[arm][str(seed)]={}
            for fold,ix,batch in (('validation',vix,val),('test',tix,test)):
                a,k,nll=parameters(model,batch);curve=ensemble_curve(a,k)
                save_curve(path/(fold+'_predictions.npz'),curve,holder,ix,log_amplitude_members=a,shape_members=k)
                models[arm][str(seed)]['folds'][fold]=dict(nll=nll,**summary(curve,holder,ix))
                with np.load(AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz') as z:
                    if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]):raise ValueError('XGB member mismatch')
                    comparator=z['curve']
                comparisons[arm][str(seed)][fold]=paired_interval(d['subject_ids'][ix],d['times'][ix],d['events'][ix],curve[:,:,1],comparator[:,:,1],seed=seed)
            del model
    r=dict(status='completed_training_and_prediction_pending_independent_verification',models=models,fits=fits,
        learning_rates=choices,comparisons_to_same_input_xgb=comparisons,elapsed_seconds=time.perf_counter()-start,contract=contract)
    write_json(ROOT/'report.json',r);render(ROOT,r);print(json.dumps(dict(status=r['status'],seconds=r['elapsed_seconds'])),flush=True)


def verify():
    # 公共验收逻辑不变，显式替换为补正工厂，源文件没有被修改。
    import training.v6.public_tabular_verify as verifier
    contract=json.loads((ROOT/'contract.json').read_text())
    if sha256(Path('docs/V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md'))!=contract['correction_protocol_sha256']:
        raise ValueError('correction protocol changed')
    verifier.model_for=model_for
    verifier.verify(ROOT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('train','verify','finish'),required=True);args=p.parse_args()
    if args.mode=='train':run()
    elif args.mode=='verify':verify()
    else:
        from training.v6.public_tabular_report import finish
        finish(ROOT)
