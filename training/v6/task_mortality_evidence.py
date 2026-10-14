"""补核第二批死亡方向；只读锁定确认，不改变再入院/选模口径。"""
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.task_optimization_experiment import ROOT,SEEDS
from training.v6.task_optimization_audit import inputs
from training.v6.evaluation import paired_interval,outcome_matrix,metrics
from training.v6.train import write_json


def death_pair(subjects,times,events,candidate,reference,*,seed):
    # 仅在评价副本互换两个事件号以复用配对代码；不改训练/标签/预测。
    swapped=np.where(events==1,2,np.where(events==2,1,0))
    np.testing.assert_array_equal(outcome_matrix(times,swapped),outcome_matrix(times,events,cause=2))
    p=paired_interval(subjects,times,swapped,candidate,reference,replicates=200,seed=seed)
    expected=metrics(times,events,candidate,cause=2)['integrated_brier_7_365']-metrics(times,events,reference,cause=2)['integrated_brier_7_365']
    np.testing.assert_allclose(p['integrated_brier_difference'],expected,atol=1e-12,rtol=0)
    return p


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text());proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['report_sha256']!=sha256(ROOT/'confirmation_report.json'):raise ValueError('confirmation proof stale')
    (d,_,_,_,_,_,_,ix,_,_),_=inputs();chosen=r['lock']['selected'];pairs={}
    for seed in SEEDS:
        risks={}
        for name in (chosen,'control','official_V6'):
            with np.load(ROOT/f'seed{seed}'/name/'confirmation_predictions.npz') as z:
                for key,expected in (('hadm_ids',d.arrays['hadm_ids'][ix]),('subject_ids',d.arrays['subject_ids'][ix]),('times',d.times[ix]),('events',d.events[ix])):
                    np.testing.assert_array_equal(z[key],expected)
                risks[name]=z['curve'][:,:,2].copy()
            observed=metrics(d.times[ix],d.events[ix],risks[name],cause=2)
            if observed['integrated_brier_7_365']!=r['models'][str(seed)][name]['metrics']['death']['integrated_brier_7_365']:
                raise ValueError('death metric differs')
        pairs[str(seed)]={ref:death_pair(d.arrays['subject_ids'][ix],d.times[ix],d.events[ix],risks[chosen],risks[ref],seed=seed)
                         for ref in ('control','official_V6')}
    result=dict(candidate=chosen,comparisons=pairs,report_sha256=sha256(ROOT/'confirmation_report.json'),
        code_sha256=sha256(Path(__file__)),goal_achieved=False,test_not_evaluated=True,
        limits=['Post-hoc secondary endpoint analysis; no reselection.','Three clinical seeds reuse two base sources.',
        'Patient intervals conditional on fitted model, no multiplicity adjustment.','Not evidence of readmission improvement.'])
    write_json(ROOT/'mortality_followup.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    labels=[]
    for j,(seed,ref) in enumerate((s,ref) for s in SEEDS for ref in ('control','official_V6')):
        p=pairs[str(seed)][ref];labels.append(f'Seed {seed}: vs {ref}')
        for ax,key,ci in zip(axes,('integrated_brier_difference','auc90_difference'),('integrated_brier_95ci','auc90_95ci')):
            ax.hlines(j,*p[ci],color=f'C{SEEDS.index(seed)}',lw=2);ax.scatter(p[key],j,color=f'C{SEEDS.index(seed)}',marker='D')
    for ax in axes:ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.25);ax.set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis();axes[0].set_xlabel('Death IBS7-365 difference (negative better)')
    axes[1].set_xlabel('Death AUC90 difference (positive better)')
    fig.suptitle('Locked independent clinical branches: secondary mortality follow-up\n'
                 '200 patient-cluster bootstrap; exploratory 95% intervals; no multiplicity adjustment')
    fig.tight_layout();fig.savefig(ROOT/'mortality_increment.png',dpi=180);plt.close(fig)
    lines=['# 第二批死亡方向：补充配对核验','',
           '只读已锁定候选，不更换候选或训练标签。这里是死亡概率，不是再入院成功证据。',
           '', '| 种子 | 参照 | 死亡IBS差值↓ | 95%患者区间 | 死亡AUC90差值↑ | 95%患者区间 |',
           '|---|---|---:|---|---:|---|']
    for seed in SEEDS:
        for ref,p in pairs[str(seed)].items():
            lines.append(f'| {seed} | {ref} | {p["integrated_brier_difference"]:.6f} | {p["integrated_brier_95ci"]} | {p["auc90_difference"]:.6f} | {p["auc90_95ci"]} |')
    lines+=['','![死亡方向配对区间](mortality_increment.png)','',
            '原始住院ID/患者ID/时间/事件与确认数据逐行一致，死亡IBS重算与主报告一致。',
            '评价副本交换事件号仅复用已审计配对函数，直接cause=2指标和结局矩阵均核对一致；训练和原标签未改。',
            '属于事后次要终点探索，不替代五种子稳定确认，不用于当前第三批选模。']
    (ROOT/'MORTALITY_FOLLOWUP.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='secondary_mortality_verified',comparisons=pairs,goal_achieved=False)),flush=True)


if __name__=='__main__':run()
