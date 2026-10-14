"""只依据验证IBS选择新研究参照，不修改旧默认、旧检查点或测试指标。"""
import argparse
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def choose(root):
    r=json.loads((root/'report.json').read_text());proof=json.loads((root/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('neural proof stale')
    seeds=r['contract']['seeds'];names=list(r['models'][str(seeds[0])])
    scores={n:sum(r['models'][str(s)][n]['validation_ibs'] for s in seeds)/len(seeds) for n in names}
    best=min(scores,key=scores.get)
    selected=dict(status='validation_selected_research_reference',arm=best,selection='minimum mean validation IBS7-365 over predefined two seeds',
        validation_scores=scores,checkpoints={str(s):dict(path=str(root/f'seed{s}'/best/'best.pt'),
            sha256=r['models'][str(s)][best]['checkpoint_sha256'],epoch=r['models'][str(s)][best]['best_epoch']) for s in seeds},
        report_sha256=sha256(root/'report.json'),daily_curve_interface='0..365: S,Freadmission,Fdeath plus daily mass',
        limits=['Research reference only; previous defaults and assets unchanged.','No selection using test scores; observed test gains remain exploratory.'])
    write_json(root/'research_reference.json',selected);print(json.dumps(dict(selected=best,validation_ibs=scores[best])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'));choose(p.parse_args().root)
