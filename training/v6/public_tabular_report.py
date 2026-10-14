"""正式结果完成后补充原始来源/树重放、汇总和图表；不重新挑模型。"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import xgboost as xgb
from patient_data.export_mimiciv import sha256
from training.v6.public_tabular_data import read
from training.v6.public_tabular_experiment import ROOT, setup
from training.v6.audited_experiment import load, OUTPUT as AUDITED
from training.v6.audited_simple_controls import tree_curve
from training.v6.finalize_audited_controls import NativeTree
from training.v6.evaluation import paired_interval, metrics
from training.v6.train import write_json

ORIGINAL=Path('training_runs/v6_original_interaction_2026-10-12_r3')


def mixture_nll(a,k,times,events):
    """最终混合联合分布NLL；区别于TabM训练用的成员平均NLL。"""
    from scipy.special import logsumexp
    scaled=np.maximum(times,1e-3)[:,None,None]/365.
    cumulative=np.exp(a)*np.power(scaled,k)
    log_hazard=a+np.log(k)-np.log(365.)+(k-1)*np.log(scaled)
    selected=log_hazard[np.arange(len(a))[:,None],np.arange(a.shape[1])[None,:],np.maximum(events-1,0)[:,None]]
    loss=cumulative.sum(2)-np.where(events[:,None]>0,selected,0.)
    return float((-logsumexp(-loss,axis=1)+np.log(a.shape[1])).mean())


def finish(root=ROOT):
    setup();r=json.loads((root/'report.json').read_text());verification=json.loads((root/'independent_verification.json').read_text())
    if verification['status']!='independently_verified' or verification['report_sha256']!=sha256(root/'report.json'):
        raise ValueError('verify before final report')
    d,prep=read(root);cfg,original,c,h,a,audit=load()
    for key in ('hadm_ids','subject_ids','patient_splits'):
        if not np.array_equal(d[key],original.arrays[key]):raise ValueError('original member mismatch')
    if not np.array_equal(d['events'],original.events) or not np.array_equal(d['times'],original.times):
        raise ValueError('original strict labels mismatch')
    if sha256(Path('docs/V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md'))!=r['contract']['protocol_sha256']:
        raise ValueError('predeclared protocol changed')
    manifest=json.loads((cfg.snapshot/'metadata/manifest.json').read_text())
    raw_hashes={}
    for key,value in manifest['sources'].items():
        file=cfg.snapshot/(key+'.csv.gz')
        digest=value['sha256']
        actual=sha256(file)
        if actual!=digest:raise ValueError('raw snapshot changed: '+key)
        raw_hashes[key]=actual
    simple=json.loads((AUDITED/'simple_controls/report.json').read_text())
    old=json.loads((ORIGINAL/'report.json').read_text())
    tree_errors={};comparisons={};references={};predictive_nll={}
    for seed in (42,7):
        depth=simple['xgb_selected_depth'][str(seed)];models=[]
        for cause in (1,2):
            booster=xgb.Booster();booster.load_model(AUDITED/'simple_controls'/f'native_seed{seed}_depth{depth}_cause{cause}.ubj')
            models.append(NativeTree(booster))
        with np.load(AUDITED/'simple_controls'/f'xgb{seed}_baseline.npz') as z:
            hz=[(z[f'{j}_time'],z[f'{j}_increment']) for j in range(2)]
        for fold in ('validation','test'):
            ix=np.flatnonzero(d['patient_splits']==fold)
            curve=tree_curve(models,hz,d['x_xgb'][ix])
            with np.load(AUDITED/'simple_controls'/f'xgb{seed}_{fold}_predictions.npz') as z:
                if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]):raise ValueError('tree members mismatch')
                err=float(np.max(np.abs(curve-z['curve'])))
            if err>1e-10:raise ValueError('matrix is not same XGBoost source representation')
            tree_errors[f'{seed}/{fold}']=err
        references[str(seed)]=dict(xgb=simple['folds'][f'xgb{seed}'],original_v6=old['models'][str(seed)]['control']['folds'])
    for arm,models in r['models'].items():
        comparisons[arm]={};predictive_nll[arm]={}
        for seed in (42,7):
            path=root/arm/f"lr{r['learning_rates'][arm]}"/f'seed{seed}'
            predictive_nll[arm][str(seed)]={}
            for fold in ('validation','test'):
                with np.load(path/(fold+'_predictions.npz')) as z:
                    predictive_nll[arm][str(seed)][fold]=mixture_nll(z['log_amplitude_members'].astype(float),
                        z['shape_members'].astype(float),z['times'],z['events'])
            ix=np.flatnonzero(d['patient_splits']=='test')
            with np.load(path/'test_predictions.npz') as z:curve=z['curve']
            with np.load(ORIGINAL/f'seed{seed}'/'control/test_predictions.npz') as z:
                if not np.array_equal(z['hadm_ids'],d['hadm_ids'][ix]) or not np.array_equal(z['events'],d['events'][ix]):
                    raise ValueError('original reference members/endpoint mismatch')
                reference=z['curve']
            comparisons[arm][str(seed)]=paired_interval(d['subject_ids'][ix],d['times'][ix],d['events'][ix],curve[:,:,1],reference[:,:,1],seed=seed)
    supplement=dict(status='completed_and_verified',original_labels_checked=True,raw_snapshot_files=len(raw_hashes),
        raw_snapshot_hashes=raw_hashes,xgb_shared_matrix_replay_errors=tree_errors,references=references,
        comparisons_to_original_v6=comparisons,predictive_mixture_nll=predictive_nll,source_sha256=sha256(Path(__file__)),
        limits='Original V6 differs in recipe and frozen training. Same XGB information, but NN normalization/clipping differ; not pure parameter-count comparison.')
    supplement['best_public_by_validation']=min(r['models'],key=lambda arm:np.mean([
        r['models'][arm][str(seed)]['validation_ibs'] for seed in (42,7)]))
    supplement['default_changed']=False
    supplement['text_pca_explained_variance_fraction']=float(prep['text_pca'].explained_variance_ratio_.sum())
    write_json(root/'final_summary.json',supplement)
    render(root,r,references,d)
    document(root,r,supplement)
    # 验证完的数值不改；仅将本轮新报告状态发布为完成，记录发布前后哈希。
    before=sha256(root/'report.json')
    r['status']='completed_and_independently_verified'
    write_json(root/'report.json',r)
    verification['verified_report_sha256_before_publication']=before
    verification['report_sha256']=sha256(root/'report.json')
    verification['publication_change']='report status only; no metrics/checkpoints/predictions changed'
    write_json(root/'independent_verification.json',verification)
    print(json.dumps(dict(status=supplement['status'],trainings=20,epochs_each=100,
        raw_files_unchanged=len(raw_hashes),tree_replay_errors=tree_errors)),flush=True)


def render(root,r,refs,d):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['original_v6','xgb']+list(r['models']);means={}
    for name in names:
        source=[refs[str(s)][name]['test']['readmission'] for s in (42,7)] if name in refs['42'] else [r['models'][name][str(s)]['folds']['test']['readmission'] for s in (42,7)]
        means[name]=[np.mean([m['points']['90']['auc'] for m in source]),np.mean([m['integrated_brier_7_365'] for m in source])]
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for i,name in enumerate(names):
        for j in (0,1):axes[j].scatter(means[name][j],i,s=70)
    for ax in axes:ax.set_yticks(range(len(names)),names);ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();axes[0].set_xlabel('AUC90 (two-seed mean; higher better)');axes[1].set_xlabel('IBS7-365 (two-seed mean; lower better)')
    fig.suptitle('Strict AECOPD; historical test; original V6 recipe differs')
    fig.tight_layout();fig.savefig(root/'performance_overview.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));ix=np.flatnonzero(d['patient_splits']=='test')
    for name in names:
        measurements=[]
        for seed in (42,7):
            if name=='original_v6':file=ORIGINAL/f'seed{seed}'/'control/test_predictions.npz'
            elif name=='xgb':file=AUDITED/'simple_controls'/f'xgb{seed}_test_predictions.npz'
            else:file=root/name/f"lr{r['learning_rates'][name]}"/f'seed{seed}'/'test_predictions.npz'
            with np.load(file) as z:curve=z['curve']
            m=metrics(d['times'][ix],d['events'][ix],curve[:,:,1]);measurements.append(m)
        axes[0].plot(np.arange(7,366),np.nanmean([m['auc_curve'][7:] for m in measurements],axis=0),label=name)
        axes[1].plot(np.arange(7,366),np.mean([m['brier_curve'][7:] for m in measurements],axis=0),label=name)
    axes[0].set_ylabel('AUC(t)');axes[1].set_ylabel('Brier(t)')
    for ax in axes:ax.set_xlabel('Days after discharge');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Complete continuous-time risk evaluation; two-seed mean; reused test')
    fig.tight_layout();fig.savefig(root/'time_dependent_metrics.png',dpi=170);plt.close(fig)


def document(root,r,s):
    def points(m):return [m['points'][str(day)]['auc'] for day in (30,90,180,365)]+[m['points']['90']['brier'],m['integrated_brier_7_365']]
    lines=['# 官方表格深度学习模块：完整受控对照结果','',
        '本轮全部20次训练均完成100epoch；10个按验证选择的模型、20折新进程重放通过。没有覆盖原V6或改变队列/严格结局/文本向量。',
        '', '[固定方案](V6_PUBLIC_TABULAR_PROTOCOL_2026-10-12.md) · [结果目录](../'+str(root).replace('\\','/')+'/README.md)',
        '', '## 输入及实际适配','',
        '19,401出院、5,351患者；训练13,592、验证2,949、历史测试2,860。共同信息原矩阵1390维；训练常量734列排除后，273连续+383二元列。检验/药物细节与有效性标记保留。四字段BERT章节均值及训练折16维PCA与XGBoost相同，不是全文原token训练，也不是重新编码。',
        '', '所有网络端到端训练，不冻结H34；每成员使用相同双原因患者相关Weibull联合似然。TabM16成员平均概率，不平均参数；其最终分布是Weibull混合，比单成员更灵活，不能将增益全归因于骨干/数值嵌入。report.json的NLL为成员平均训练损失，最终混合分布NLL单列于final_summary.json。模型/学习率选择只用验证IBS7–365；90天仅是报告点，每天累计/当日/条件当日概率均保存。',
        '', '## 历史测试结果（两种子平均，不是两种子集成）','',
        '|模型|AUC30|AUC90|AUC180|AUC365|Brier90↓|IBS7–365↓|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['original_v6','xgb']+list(r['models']):
        source=[s['references'][str(seed)][name]['test']['readmission'] for seed in (42,7)] if name in ('original_v6','xgb') else [r['models'][name][str(seed)]['folds']['test']['readmission'] for seed in (42,7)]
        values=np.mean([points(m) for m in source],axis=0)
        lines.append('|'+name+'|'+'|'.join(f'{v:.6f}' for v in values)+'|')
    lines+=['', '## 本轮结论','',
        '验证选择的公开候选为TabM+PLE。它相对同信息XGBoost的AUC90与IBS点估计在两个种子均更好，但患者配对95%区间均跨0，因此只能称小幅、待确认的增益，不能称显著突破或等效证明。',
        '', '原V6 AUC90约0.7992，TabM约0.7938；TabM的Brier90/IBS均值略好，但排序没有超过原V6，且原V6配方/历史冻结不同。当前不替换原V6默认；普通MLP、两种ResNet、FT均未胜出。',
        '', '多数单模型最佳点在2–19epoch；完成100epoch后验证反而恶化。该实证不支持“再多训练或只加深网络就会改善”的解释，但有限LR/容量预算也不能证明所有网络设置的性能上限。下一步不再无差别增加模块；TabM仅保留为有证据的候选。']
    lines+=['','## 实际训练、选择与资源','',
        '|候选|共同LR|最佳epoch（42/7）|参数量|四次训练累计分钟|最大显存GiB|验证IBS均值|',
        '|---|---:|---|---:|---:|---:|---:|']
    for arm in r['models']:
        m=[r['models'][arm][str(seed)] for seed in (42,7)]
        allruns=[value for group in r['fits'][arm].values() for value in group.values()]
        lines.append(f"|{arm}|{r['learning_rates'][arm]}|{m[0]['best_epoch']}/{m[1]['best_epoch']}|{m[0]['parameters']:,}|{sum(v['elapsed_seconds'] for v in allruns)/60:.2f}|{max(v['peak_gpu_gib'] for v in allruns):.3f}|{np.mean([v['validation_ibs'] for v in m]):.6f}|")
    lines+=['','## 与同信息XGBoost的配对差值','',
        '差值=候选−XGBoost。IBS负值更好，AUC正值更好。每种子患者聚类bootstrap200；区间条件于已经训练的模型，不计训练不确定性，也没有多重比较校正。',
        '', '|候选|种子|ΔAUC90（95%区间）|ΔIBS（95%区间）|','|---|---:|---|---|']
    for arm in r['models']:
        for seed in (42,7):
            v=r['comparisons_to_same_input_xgb'][arm][str(seed)]['test']
            lines.append(f"|{arm}|{seed}|{v['auc90_difference']:+.5f} [{v['auc90_95ci'][0]:+.5f}, {v['auc90_95ci'][1]:+.5f}]|{v['integrated_brier_difference']:+.6f} [{v['integrated_brier_95ci'][0]:+.6f}, {v['integrated_brier_95ci'][1]:+.6f}]|")
    lines+=['','## 图表','',
        '![整体性能](../'+str(root).replace('\\','/')+'/performance_overview.png)','',
        '![患者配对](../'+str(root).replace('\\','/')+'/paired_xgb.png)','',
        '![训练过程](../'+str(root).replace('\\','/')+'/training.png)','',
        '![整段时间评价](../'+str(root).replace('\\','/')+'/time_dependent_metrics.png)','',
        '## 证据限制与使用决定','',
        f"五个公开候选中，验证IBS均值最好的为 **{s['best_public_by_validation']}**。这是验证集选择，不是按测试AUC挑选；旧默认仍不覆盖。四字段16维PCA保留的训练方差比例约为{s['text_pca_explained_variance_fraction']*100:.1f}%，但方差保留不等于临床语义/预后信息保留；缺失状态也可能主导方差。本轮不能断言公开骨干已经用尽原始文本信息。",
        '',
        '本轮完成的是有限预算方法适配，不能证明所有深度学习模块的性能上限。只测试2个共同LR与1个固定容量；没有针对每个算法大规模调参。新网络与XGBoost信息来源相同，但网络数值归一化/截断不同。原V6的输入配方及历史冻结不同，作为强参照而不是纯骨干消融。',
        '', '完整100epoch不等于应该使用第100epoch：如果验证在早期最好、随后训练误差下降但验证变差，就是泛化恶化；扩大网络或延长训练不能自动解决。检验/药物的医学有效性不等于在“历史+诊断+去向”条件下对本库严格再入院标签一定有增量。',
        '', '按既定边界不自动替换默认模型、不删医学变量。日概率仍只对应MIMIC内观察到的再入院/死亡，不能视作院外完整真实风险。历史测试已被反复查看，所有优势须外部/新留出验证；不能以某一天AUC最高直接宣布突破。',
        '', f"新进程10模型20折重放、概率守恒及患者划分检查通过；原始快照{s['raw_snapshot_files']}文件哈希未变；共同XGBoost矩阵重放最大误差{max(s['xgb_shared_matrix_replay_errors'].values()):.3g}。",
        '', '## 方法的可核验来源','',
        '- [TabM：官方ICLR2025代码](https://github.com/yandex-research/tabm)：共享权重的高效集成。本项目适配为联合竞争风险，而非其原分类/回归benchmark。',
        '- [MLP/ResNet/FT-Transformer：官方NeurIPS2021代码](https://github.com/yandex-research/rtdl-revisiting-models)：使用真实公开实现，FT读CLS。',
        '- [数值嵌入：官方NeurIPS2022代码](https://github.com/yandex-research/rtdl-num-embeddings)：分段线性与周期嵌入，训练折拟合分位点；不代表可自动补齐缺失临床事实。',
        '', '复现命令与固定包见training/runtime/README.md；完整数字、全部参数候选及每日预测以本轮report.json为准。']
    if r['contract'].get('phase')=='complete_official_readout':
        lines[3:3]=['', '## 读出层补正与真实执行次数','',
            '首次d_out=None同时移除了ResNet/FT的末端归一化+ReLU，发现后保留为探索记录，不以其负结果评价完整官方模型。正式结果来自official_readout子目录：ResNet恢复BatchNorm+ReLU，FT恢复LayerNorm+ReLU，再接共同生存线性层。受影响12配置重新100epoch，MLP/TabM8配置核验后原样复用；整个过程实际32次训练，不冒充首次一次完成。',
            '', '[补正原因、范围及合同](V6_PUBLIC_TABULAR_READOUT_CORRECTION_2026-10-12.md)。测试已被反复使用；补正源于官方代码核查，不是按测试成绩选择架构。']
    Path('docs/V6_PUBLIC_TABULAR_RESULTS_2026-10-12.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':finish()
