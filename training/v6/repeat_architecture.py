"""用另一种子确认关键对照；旧H检查点固定，B及临床Δ重新训练。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from patient_data.export_mimiciv import sha256
from training.v6 import architecture_experiment as experiment


def run(seed, pilot, baseline_output, explore_output):
    if seed==42: raise ValueError('confirmation seed must differ')
    pilot_report=json.loads((pilot/'report.json').read_text())
    if pilot_report['status']!='completed_explore_exploration_single_seed': raise ValueError('pilot not complete')
    # 仅用验证选优；成分/比值、原B/候选/融合、pool参照为预先列出的核心比较。
    selected=pilot_report['validation_selected']
    names={'pool128','availability','components','nlr_sii','no_text','screen_only','bert_screen',selected}
    original=experiment.configurations
    filtered={name:value for name,value in original('explore').items() if name in names}
    common=dict(reference=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'),
                compact_dir=Path('patient_data/processed/v6_compact_multisource_2026-10-09'),
                combinations=Path('patient_data/processed/v6_cbc_combinations_2026-10-10'),
                candidate=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10'),
                epochs=60,batch_size=512,bootstrap=200,seed=seed,verify=False,baseline_run=baseline_output)
    experiment.run(SimpleNamespace(**common,phase='baseline',output=baseline_output))
    experiment.configurations=lambda phase: filtered if phase=='explore' else original(phase)
    try: experiment.run(SimpleNamespace(**common,phase='explore',output=explore_output))
    finally: experiment.configurations=original
    for folder in (baseline_output,explore_output):
        report=json.loads((folder/'report.json').read_text())
        report['contract']['code_sha256'][str(Path(__file__))]=sha256(Path(__file__))
        report['contract']['confirmation_selection']='Core arms prespecified; extra architecture from seed42 validation only'
        report['contract']['confirmation_pilot_report_sha256']=sha256(pilot/'report.json')
        report['contract']['limits'].append('H is the fixed seed42 historical reference; B and delta retrained under confirmation seed, not full H retraining variance')
        experiment.write_json(folder/'report.json',report)
        experiment.write_json(folder/'contract.json',report['contract'])
        experiment.verify(SimpleNamespace(output=folder))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--pilot',type=Path,default=Path('training_runs/v6_architecture_screen_seed42_2026-10-10'))
    p.add_argument('--baseline-output',type=Path,default=Path('training_runs/v6_expanded_baseline_seed7_2026-10-10'))
    p.add_argument('--explore-output',type=Path,default=Path('training_runs/v6_architecture_screen_seed7_2026-10-10'))
    a=p.parse_args(); run(a.seed,a.pilot,a.baseline_output,a.explore_output)
