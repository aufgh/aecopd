"""已验收V6研究模型按住院重载预测；不训练，不使用未来标签计算预测。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.compact_ablation import inputs
from training.v6.architecture_experiment import batches
from training.v6.text_encoder_comparison import new_model
from training.v6.horizon_objective import restore
from training.v6.risk import predict_cif
from training.v6.train import write_json

DAYS=np.arange(366,dtype=float)


@torch.no_grad()
def scores(model,prepared):
    """标签只随数据批存在，故意不读取t/e，不调用带标签的predict90。"""
    model.eval(); a,k=[],[]
    for _,history,batch,_,_ in prepared:
        out=model(history,batch) if hasattr(model,'encoder') else model(history)
        a.append(out.log_amplitude.cpu().numpy()); k.append(out.shape.cpu().numpy())
    return np.concatenate(a),np.concatenate(k)


def predict(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required; no CPU embedding')
    torch.set_num_threads(2)
    data,compact,hist,_=inputs(args.reference,args.compact_dir)
    parent=json.loads((args.parent/'report.json').read_text())
    report=json.loads((args.text_parent/'report.json').read_text())
    for root,r in ((args.parent,parent),(args.text_parent,report)):
        proof=json.loads((root/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(root/'report.json'): raise ValueError('unverified model source')
        for p,digest in r['contract']['code_sha256'].items():
            if sha256(Path(p))!=digest: raise ValueError('model source code changed')
    if data.provenance!=report['contract']['data_provenance'] or compact.provenance!=report['contract']['compact_provenance']:
        raise ValueError('model input source changed')
    if args.hadm_ids:
        lookup={int(h):i for i,h in enumerate(data.arrays['hadm_ids'])}
        if len(set(args.hadm_ids))!=len(args.hadm_ids) or any(h not in lookup for h in args.hadm_ids):
            raise ValueError('requested admission absent or duplicated')
        ix=np.array([lookup[h] for h in args.hadm_ids],dtype=np.int64)
    else:
        ix=np.flatnonzero(data.arrays['patient_splits']=='test')
        ix=ix[np.argsort(data.arrays['hadm_ids'][ix])][:args.first_test]
    prepared=batches(data,compact,ix,512,expanded=True,text='bert')
    # 反事实修改标签作工程核验；特征、标识、时间来源全部不动。
    altered=replace(data,times=np.zeros_like(data.times),events=np.zeros_like(data.events))
    altered_batches=batches(altered,replace(compact,base=altered),ix,512,expanded=True,text='bert')
    curves,base_curves,proofs=[],[],{}
    for seed in (42,7):
        path=args.text_parent/f'seed{seed}/bert/best.pt'
        if sha256(path)!=report['models'][str(seed)]['bert']['checkpoint_sha256']: raise ValueError('checkpoint changed')
        m=new_model(args,seed,'bert',hist,.1)
        m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state_dict'])
        a,k=scores(m,prepared); altered_a,altered_k=scores(m,altered_batches)
        if not np.array_equal(a,altered_a) or not np.array_equal(k,altered_k): raise ValueError('future label influences prediction')
        curves.append(predict_cif(a,k,DAYS))
        fine=predict_cif(a,k,DAYS,step_days=.125)
        if np.abs(fine-curves[-1]).max()>2e-4: raise ValueError('numerical integration not converged')
        base=restore(args.parent,seed,'survival_base',hist,.1)
        ba,bk=scores(base,prepared); base_curves.append(predict_cif(ba,bk,DAYS))
        proofs[str(seed)]=dict(checkpoint_sha256=sha256(path),label_invariance_exact=True,
            fine_step_max_error=float(np.abs(fine-curves[-1]).max()))
        del m,base
    stacked=np.stack(curves); mean=stacked.mean(0); reference=np.stack(base_curves).mean(0)
    if not np.allclose(mean.sum(2),1,atol=1e-9) or (np.diff(mean[:,:,1:],axis=1)<-1e-10).any():
        raise ValueError('ensemble probability contract failed')
    args.output.mkdir(parents=True)
    records=[]
    for j,p in enumerate(ix):
        field_count={f:sum(x[0]==i for x in data.text_refs[p]) for i,f in enumerate(('HPI','PMH','Hospital course','Discharge condition'))}
        records.append(dict(hadm_id=int(data.arrays['hadm_ids'][p]),subject_id=int(data.arrays['subject_ids'][p]),
            patient_split=str(data.arrays['patient_splits'][p]),
            recorded_prior_copd_stays=int(np.rint(np.expm1(data.history_raw[p,2]))),
            observed_lab_axes=int((data.arrays['lab_axis_inputs'][p,:,6]>0).sum()),text_chunks=field_count,
            predicted_cif={str(t):dict(readmission=float(mean[j,t,1]),death_before_readmission=float(mean[j,t,2]),
                neither=float(mean[j,t,0]),baseline_readmission=float(reference[j,t,1])) for t in (7,14,30,60,90,180,365)}))
    result=dict(status='verified_research_inference_only',selection='requested admissions or smallest test hadm_ids; no outcome selection',
        source_reports_sha256={str(args.parent):sha256(args.parent/'report.json'),str(args.text_parent):sha256(args.text_parent/'report.json')},
        input_provenance=data.provenance,proofs=proofs,records=records,
        limits=['Research only; no deployment/clinical action validation or complete outside follow-up.',
            'Equal-weight two-seed mixture; not a calibrated confidence interval or new performance claim.',
            'Current final discharge text allowed even if stored after discharge.',
            'Predicts observed strict readmission cumulative probability, not an exact return date.',
            'Not a private-data adapter; raw concept mapping and normalization still need validation there.'])
    write_json(args.output/'predictions.json',result)
    np.savez_compressed(args.output/'risk_curves.npz',hadm_ids=data.arrays['hadm_ids'][ix],days=DAYS,
        full_seed_curves=stacked,full_mean=mean,baseline_mean=reference)
    plot(args.output,records,mean,reference,stacked)
    write_json(args.output/'verification.json',dict(status='source_hashes_label_invariance_mass_monotonicity_and_integration_checked',
        predictions_sha256=sha256(args.output/'predictions.json'),curve_sha256=sha256(args.output/'risk_curves.npz'),
        code_sha256=sha256(Path(__file__)),max_probability_sum_error=float(np.abs(mean.sum(2)-1).max())))
    print(json.dumps(dict(status=result['status'],admissions=len(records),output=str(args.output))),flush=True)


def plot(root,records,mean,baseline,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(records),1,figsize=(11,3.5*len(records)),squeeze=False)
    for i,record in enumerate(records):
        ax=axes[i,0]
        ax.plot(DAYS,baseline[i,:,1],color='#999999',ls='--',label='Frozen history + diagnosis + disposition')
        ax.plot(DAYS,mean[i,:,1],color='#2674ae',label='Plus clinical state + BERT: strict readmission')
        ax.fill_between(DAYS,seeds[:,i,:,1].min(0),seeds[:,i,:,1].max(0),color='#2674ae',alpha=.15,label='Two-seed range, NOT confidence interval')
        ax.plot(DAYS,mean[i,:,2],color='#cf7722',label='Death before target readmission')
        ax.scatter([30,90,180,365],mean[i,[30,90,180,365],1],s=15,color='#2674ae')
        ax.set(xlabel='Days after this discharge',ylabel='Predicted cumulative probability',ylim=(0,1),
            title=f'Admission{record["hadm_id"]}; recorded prior COPD stays={record["recorded_prior_copd_stays"]}; readmission risk90={mean[i,90,1]:.1%}')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Reloaded V6 research prediction; selected by ID, not outcome; no clinical deployment claim')
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(root/'patient_risk_curves.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'))
    p.add_argument('--compact-dir',type=Path,default=Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    p.add_argument('--parent',type=Path,default=Path('training_runs/v6_90day_objective_2026-10-11'))
    p.add_argument('--text-parent',type=Path,default=Path('training_runs/v6_bert_qwen_same_fields_2026-10-11'))
    p.add_argument('--hadm-ids',type=int,nargs='+'); p.add_argument('--first-test',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('training_runs/v6_research_prediction_demo_2026-10-11'))
    args=p.parse_args()
    if not 1<=args.first_test<=12: p.error('demo size1..12; use explicit hadm-ids for more')
    predict(args)


if __name__=='__main__': main()
