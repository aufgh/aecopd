"""补充来源全量合同与 GPU 张量验收；零优化步骤，新增候选不伪称进入风险头。"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.data import load
from training.v6.source_context_data import attach
from training.v6.model import ClinicalEncoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','state_dir','labels','eos_dir','resp_micro_dir','output'):
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU fallback')
    base=load(args.snapshot,args.state_dir,args.labels)
    data=attach(base,args.eos_dir,args.resp_micro_dir); torch.set_num_threads(1)
    encoder=ClinicalEncoder(768).cuda().eval(); result={}
    for fold in ('train','validation','test'):
        candidates=np.flatnonzero(base.arrays['patient_splits']==fold)
        ix=candidates[np.linspace(0,len(candidates)-1,64,dtype=int)]
        h,b,t,e=data.collate(ix,'cuda')
        shapes={}
        for f in fields(b):
            if f.name=='clinical': continue
            value=getattr(b,f.name)
            if value.device.type!='cuda' or not torch.isfinite(value).all(): raise ValueError('invalid GPU supplemental input')
            shapes[f.name]=list(value.shape)
        with torch.no_grad(): state,_=encoder(b.clinical)
        if not torch.isfinite(state).all(): raise ValueError('corrected medication source mask invalid')
        result[fold]={'admissions_checked':len(ix),'supplemental_shapes':shapes,'finite_clinical_encoder':True}
    report={'status':'supplemental_data_gpu_interface_verified_not_trained','admissions_aligned':len(base.history),
            'patients_aligned':len(set(base.arrays['subject_ids'].tolist())),'device':torch.cuda.get_device_name(),
            'optimizer_steps':0,'checkpoint_saved':False,'folds':result,'provenance':data.provenance,
            'code_sha256':{n:sha256(Path(__file__).parent/n) for n in ('source_context_data.py','source_context_smoke.py')},
            'limits':['New EOS/micro/resp candidates validated as tensors, not consumed by current risk network',
                      'Only corrected medication source mask tested through existing clinical encoder',
                      'No training, encoder re-encoding, model selection or predictive performance evaluation']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','admissions_aligned','patients_aligned','optimizer_steps','device')}))


if __name__=='__main__': main()
