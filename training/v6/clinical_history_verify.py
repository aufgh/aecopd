"""第六批独立重建教师缓存、历史索引和全部调参预测；不读取确认/测试。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.regularization_audit import checked,REFERENCE
from training.v6.clinical_history_experiment import ROOT,SEEDS,layout,cache_data,score
from training.v6.clinical_history_model import ARMS,HistoryReader
from training.v6.clinical_history_audit import ROOT as INDEX_ROOT
from training.v6.warm_residual import BASE_SEEDS
from training.v6.module_surgery_experiment import summary,save
from training.v6.task_optimization_audit import check_curve,check_saved
from training.v6.train import write_json


def load(seed,arm):
    path=ROOT/f'seed{seed}'/arm
    info=json.loads((path/'training_result.json').read_text())
    rows=json.loads((path/'history.json').read_text())
    state=torch.load(path/'best.pt',map_location='cuda',weights_only=True)
    if (info['seed']!=seed or info['arm']!=arm or info['completed_epochs']!=100 or
            [x['epoch'] for x in rows]!=list(range(101)) or sha256(path/'best.pt')!=info['checkpoint_sha256']):
        raise ValueError('history training/checkpoint differs')
    best=min(rows,key=lambda x:x['tuning_ibs'])
    if state['arm']!=arm or state['epoch']!=best['epoch'] or info['best_epoch']!=best['epoch'] or state['tuning_ibs']!=best['tuning_ibs']:
        raise ValueError('history checkpoint selection differs')
    model=HistoryReader(arm).cuda();model.load_state_dict(state['state_dict'])
    return model,info,best


def verify():
    if (ROOT/'.running.lock').exists():raise RuntimeError('clinical history training is running')
    d,c,h,ext,_,bins,tune_positions,_,search,_=checked()
    contract=json.loads((ROOT/'contract.json').read_text())
    for file,digest in contract['source_sha256'].items():
        if sha256(Path(file))!=digest:raise ValueError('history source changed: '+file)
    if contract['seeds']!=list(SEEDS) or contract['arms']!=list(ARMS):raise ValueError('history experiment specification differs')
    if contract['history_index_sha256']!=sha256(INDEX_ROOT/'past_clinical_indices.npz'):
        raise ValueError('historical index changed')
    if contract['search_contract_sha256']!=sha256(Path('training_runs/v6_regularization_search_2026-10-12/contract.json')):
        raise ValueError('history upstream contract changed')
    train_positions=np.flatnonzero(d.arrays['patient_splits']=='train');positions=np.r_[train_positions,tune_positions]
    history=layout(d,positions);tune=torch.arange(len(train_positions),len(positions),device='cuda')
    selected=json.loads((ROOT/'selected_before_confirmation.json').read_text())
    results={};errors={}
    for seed in SEEDS:
        if contract['teachers'][str(seed)]!=sha256(REFERENCE/f'seed{BASE_SEEDS[seed]}/control/best.pt'):
            raise ValueError('history teacher changed')
        if contract['frozen_base_sources'][str(seed)]!=sha256(Path(search['input_config']['parent'])/f'seed{BASE_SEEDS[seed]}/survival_base/best.pt'):
            raise ValueError('history frozen H34 changed')
        path=ROOT/f'teacher_seed{BASE_SEEDS[seed]}_train_tune_cache.npz'
        if not path.exists() or sha256(path)!=selected['cache_sha256'][str(seed)]:
            raise ValueError('history teacher cache changed')
        # 新进程重新执行完整教师编码，逐张量核对磁盘缓存；非仅文件hash。
        data,_=cache_data(seed,d,c,h,ext,bins,search,positions,train_positions,tune_positions)
        results[str(seed)]={}
        for name in ('original_teacher',*ARMS):
            if name=='original_teacher':m=HistoryReader('current_only').cuda();info=best=None
            else:m,info,best=load(seed,name)
            detail,a,k,curve=score(m,data,history,tune);check_curve(curve)
            if best is not None:
                if any(detail[key]!=best['tuning_'+key] for key in detail):raise ValueError('history training metric replay differs')
            else:
                if detail!=selected['references'][str(seed)]['tuning']:raise ValueError('teacher reference replay differs')
            # 所有无过去住院者必须保留原预测，不靠空历史产生新输出。
            empty=~history['present'][tune].any(1)
            np.testing.assert_array_equal(a[empty.cpu().numpy()],data['amplitude'][tune][empty].cpu().numpy())
            np.testing.assert_array_equal(k[empty.cpu().numpy()],data['shape'][tune][empty].cpu().numpy())
            target=ROOT/f'seed{seed}'/name/'tuning_predictions.npz';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():error=check_saved(target,d,tune_positions,a,k,curve)
            else:save(target,d,tune_positions,a,k,curve);error=None
            errors[f'{seed}/{name}']=error
            results[str(seed)][name]=dict(metrics=summary(curve,d,tune_positions,detail['nll']),training=info)
            del m
        del data
    scores={name:float(np.mean([results[str(s)][name]['metrics']['readmission']['integrated_brier_7_365'] for s in SEEDS]))
        for name in ('original_teacher',*ARMS)}
    if scores!=selected['tuning_scores'] or min(scores,key=scores.get)!=selected['selected']:
        raise ValueError('history selection changed')
    write_json(ROOT/'tuning_report.json',dict(models=results,tuning_scores=scores,selected=selected['selected'],
        confirmation_not_evaluated=True,test_not_evaluated=True,goal_achieved=False))
    replayed=all(x is not None for x in errors.values())
    proof=dict(status='history_tuning_independently_verified' if replayed else 'history_predictions_saved_pending_second_process',
        models=len(errors),errors=errors,maximum_error=max(errors.values()) if replayed else None,
        verifier_sha256=sha256(Path(__file__)),report_sha256=sha256(ROOT/'tuning_report.json'),
        selection_sha256=sha256(ROOT/'selected_before_confirmation.json'),teacher_cache_recomputed=True,
        empty_history_predictions_preserved=True,test_not_evaluated=True,goal_achieved=False)
    write_json(ROOT/'tuning_verification.json',proof)
    if replayed:
        status='no_historical_content_candidate' if selected['selected'] in ('original_teacher','current_only','times_only') else 'verified_history_candidate_pending_lock'
        write_json(ROOT/'state.json',dict(status=status,selected=selected['selected'],goal_achieved=False))
    print(json.dumps(dict(status=proof['status'],models=len(errors),maximum_error=proof['maximum_error'])),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();verify()
