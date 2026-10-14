"""真实小样本事实GPU接口验收，非训练/选模/临床性能报告。"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.clinical_facts.features import ct_vector,discharge_vector
from patient_data.clinical_facts.guards import guard
from patient_data.clinical_facts.llm import digest
from patient_data.export_mimiciv import rows
from training.v6.compact_ablation import inputs
from training.v6.clinical_facts_context import attach,FactsBatch,FactsAnchoredModel
from training.v6.risk import competing_nll

def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    data,compact,base,checkpoint=inputs(Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'),
                                       Path('patient_data/processed/v6_compact_multisource_2026-10-09'))
    facts_data=attach(compact,Path('patient_data/processed/v6_cxr_trajectory_2026-10-10_r2'))
    bundles=[('v6_bonsai_tasks_2026-10-10','v6_bonsai_ct_pilot_2026-10-10_r2'),
             ('v6_bonsai_tasks_2026-10-10_r2','v6_bonsai_discharge_pilot_2026-10-10_r3')]
    root=Path('patient_data/processed'); facts={}; sources={}; rejections=0; accepted=0
    raw_radiology={r['note_id']:r for r in rows(Path('patient_data/mimiciv_copd/snapshot_2026-09-27/raw/radiology.csv.gz'))}
    for task_dir,result_dir in bundles:
        with (root/task_dir/'tasks.jsonl').open(encoding='utf-8') as f: tasks={r['task_key']:r for r in map(json.loads,f)}
        with (root/task_dir/'sources.jsonl').open(encoding='utf-8') as f: refs={r['note_id']:r for r in map(json.loads,f)}
        with (root/result_dir/'results.jsonl').open(encoding='utf-8') as f:
            for result in map(json.loads,f):
                task=tasks[result['task_key']]; source=refs[task['note_id']]
                if source['channel']=='ct': source=dict(source,text=raw_radiology[task['note_id']]['text'])
                if digest(source['text'])!=task['original_sha256']: raise ValueError('source hash differs')
                sources.setdefault(task['hadm_id'],set()).add(source['channel'])
                fs=facts.setdefault(task['hadm_id'],[])
                for original in result['facts']:
                    fact,error=guard(original,task)
                    if error: rejections+=1; continue
                    if source['text'][fact['original_start']:fact['original_end']]!=fact['quote']: raise ValueError('original quote failed')
                    fs.append(dict(fact,channel=source['channel'],charttime=task['charttime'])); accepted+=1
    indices=np.array([i for i,h in enumerate(data.arrays['hadm_ids']) if str(h) in sources])
    h,b,t,e=facts_data.collate(indices,device='cuda'); ct=[]; ds=[]; p=b.present.clone()
    for i,position in enumerate(indices):
        hadm=str(data.arrays['hadm_ids'][position]); fs=facts.get(hadm,[])
        ct.append(ct_vector([f for f in fs if f['channel']=='ct'])); ds.append(discharge_vector([f for f in fs if f['channel']=='discharge']))
        p[i,1]='ct' in sources[hadm]; p[i,2]='discharge' in sources[hadm]
    b=FactsBatch(b.compact,(b.values[0],torch.tensor(np.stack(ct),device='cuda'),torch.tensor(np.stack(ds),device='cuda')),p)
    torch.manual_seed(42); model=FactsAnchoredModel(copy.deepcopy(base),768).cuda().eval()
    output=model(h,b); reference=model.baseline(h)
    torch.testing.assert_close(output.log_amplitude,reference.log_amplitude,rtol=0,atol=0)
    _,branches=model.encoder(b)
    # 仅验收梯度可达性；随机非零头，无optimizer、不保存预测checkpoint。
    torch.nn.init.normal_(model.delta_head[-1].weight,std=.01)
    loss=competing_nll(model(h,b),t,e); loss.backward()
    gradients={channel:float(sum(v.grad.square().sum() for v in branch.parameters() if v.grad is not None).sqrt())
               for channel,branch in zip(('cxr','ct','discharge'),model.encoder.fact_branches,strict=True)}
    if not torch.isfinite(loss) or not all(v>0 for v in gradients.values()) or any(v.grad is not None for v in model.baseline.parameters()):
        raise ValueError('GPU fact branch gradient or baseline boundary failed')
    result={'status':'real_pilot_facts_GPU_forward_backward_checked_not_trained','admissions':len(indices),
            'accepted_revalidated_facts':accepted,'additional_rejected_by_final_rules':rejections,
            'source_coverage':dict(zip(('cxr','ct','discharge'),map(int,p.sum(0).cpu().numpy()))),
            'input_widths':[v.shape[1] for v in b.values],'gradient_norms':gradients,'history_frozen':True,'zero_initial_equal_H':True,
            'limits':['Pilot outputs revalidated under final guards only for engineering smoke','No optimizer, no performance evaluation, no expert gold standard',
                      'Old pilot outputs cannot become full extraction artifacts or formal six-arm training data']}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); run(p.parse_args())
