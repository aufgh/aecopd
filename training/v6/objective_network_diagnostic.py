"""同状态全可训练临床分支的四目标梯度；只读，不训练。"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.module_surgery_experiment import batches
from training.v6.warm_residual_audit import teacher
from training.v6.horizon_objective import restore
from training.v6.objective_direction_diagnostic import NAMES,cosine,compare,losses
from training.v6.train import write_json

ROOT=Path('training_runs/v6_objective_network_2026-10-13')


def run():
    start=time.perf_counter();d,c,h,ext,_,bins,_,_,search,_=checked()
    previous=Path('training_runs/v6_objective_direction_2026-10-13')
    proof=json.loads((previous/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(previous/'diagnostic.json'):raise ValueError('head proof stale')
    before=json.loads((previous/'diagnostic.json').read_text())
    for p,digest in before['source_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('head diagnostic source changed')
    ix=np.flatnonzero(d.arrays['patient_splits']=='train');cache=batches(d,c,ext,ix);result={}
    for seed in (42,7):
        base=restore(Path(search['input_config']['parent']),seed,'survival_base',h,.1)
        model=teacher(base,seed,bins).eval()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if any(n.startswith('baseline.') for n,p in named):raise ValueError('H34 not frozen')
        params=[p for n,p in named];gradients={k:[torch.zeros_like(p,dtype=torch.float64) for p in params] for k in NAMES}
        values={k:0. for k in NAMES};last_ids={id(p) for p in model.delta_head[-1].parameters()}
        for _,history,batch,t,e in cache:
            out=model(history,batch);scored,_=losses(out,t,e)
            for j,name in enumerate(NAMES):
                values[name]+=float(scored[name].detach().sum())/len(ix)
                # 原前向TF32不动；严格梯度核验只关闭反向矩阵乘法TF32舍入。
                precision=torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32=False
                    g=torch.autograd.grad(scored[name].sum()/len(ix),params,allow_unused=True,retain_graph=j<len(NAMES)-1)
                finally:torch.backends.cuda.matmul.allow_tf32=precision
                for total,part in zip(gradients[name],g):
                    if part is not None:total.add_(part.double())
        scopes={'all_clinical':[True]*len(named),
                'encoder':[n.startswith('encoder.') for n,p in named],
                'risk_head':[n.startswith('delta_head.') for n,p in named],
                'last_projection':[id(p) in last_ids for n,p in named]}
        measures={}
        for scope,mask in scopes.items():
            vectors={k:torch.cat([g.flatten() for g,active in zip(v,mask) if active]).cpu().numpy()
                     for k,v in gradients.items()}
            if not all(np.isfinite(v).all() for v in vectors.values()):raise ValueError('nonfinite full gradients')
            measures[scope]=dict(parameters=len(vectors['nll']),
                gradient_norms={k:float(np.linalg.norm(v)) for k,v in vectors.items()},
                cosines={a:{b:cosine(vectors[a],vectors[b]) for b in NAMES} for a in NAMES})
        old=before['models'][str(seed)]
        loss_error=max(abs(values[k]-old['losses'][k]) for k in NAMES)
        cosine_error=max(abs(measures['last_projection']['cosines'][a][b]-old['gradient_cosines'][a][b])
                         for a in NAMES for b in NAMES)
        if loss_error>1e-8 or cosine_error>1e-5:
            ROOT.mkdir(exist_ok=True)
            write_json(ROOT/'failure.json',dict(seed=seed,loss_error=loss_error,cosine_error=cosine_error,
                scopes=measures,losses=values,source_sha256=sha256(Path(__file__))))
            raise ValueError(f'last-projection/training objective mismatch: loss={loss_error}, cosine={cosine_error}')
        result[str(seed)]=dict(losses=values,scopes=measures,last_projection_cosine_max_error=cosine_error,
            prior_loss_max_error=loss_error,teacher_sha256=sha256(REFERENCE/f'seed{seed}/control/best.pt'))
        print(json.dumps(dict(seed=seed,scopes=measures,seconds=round(time.perf_counter()-start,1))),flush=True)
        del model,gradients
    source=('training/v6/objective_network_diagnostic.py','docs/V6_OBJECTIVE_NETWORK_PROTOCOL_2026-10-13.md',
            'training/v6/curve_score.py','training/v6/objective_gradient_precision.py')
    report=dict(models=result,admissions=len(ix),training_index_sha256=before['training_index_sha256'],
        label_sha256=before['label_sha256'],head_report_sha256=sha256(previous/'diagnostic.json'),
        source_sha256={p:sha256(Path(p)) for p in source},goal_achieved=False,
        scope='Fixed eval; all trainable clinical parameters; frozen H34; no optimization or validation selection.')
    ROOT.mkdir(exist_ok=True);path=ROOT/'diagnostic.json'
    if path.exists():
        error=compare(json.loads(path.read_text()),report)
        write_json(ROOT/'independent_verification.json',dict(status='independently_replayed',maximum_error=error,
            report_sha256=sha256(path),seconds=time.perf_counter()-start))
    else:write_json(path,report)
    print(json.dumps(dict(status='completed_full_clinical_gradient',seconds=round(time.perf_counter()-start,1))),flush=True)


if __name__=='__main__':run()
