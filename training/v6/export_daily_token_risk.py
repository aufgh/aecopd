"""按住院记录输出已核验的逐日概率，不误称每天独立二分类或新病人部署。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256


def export(root,seed,arm,fold,record,output):
    proof=json.loads((root/'independent_verification.json').read_text())
    if proof['report_sha256']!=sha256(root/'report.json'):raise ValueError('report not independently verified')
    with np.load(root/f'seed{seed}'/arm/(fold+'_predictions.npz')) as z:
        if not 0<=record<len(z['curve']):raise ValueError('invalid record position')
        curve=z['curve'][record];daily=z['daily_event_probability'][record];conditional=z['conditional_daily_event_probability'][record]
        valid=z['conditional_daily_valid'][record] if 'conditional_daily_valid' in z.files else curve[:-1,0]>1e-12
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(('day','event_free_probability','readmission_cumulative_probability','death_cumulative_probability',
            'readmission_daily_probability','death_daily_probability','readmission_daily_given_event_free','death_daily_given_event_free','conditional_valid'))
        for day in range(366):
            if day==0:w.writerow((0,*curve[0],0,0,'','',False))
            else:w.writerow((day,*curve[day],*daily[day-1],*(conditional[day-1] if valid[day-1] else ('','')),bool(valid[day-1])))
    print(json.dumps(dict(status='exported_verified_daily_curve',days=366,output=str(output),seed=seed,arm=arm,
        scope='existing cohort saved prediction; not private-data deployment or prospectively validated risk')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('training_runs/v6_coarse_fine_token_2026-10-12_r2'))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--arm',default='interaction_fine');p.add_argument('--fold',default='test')
    p.add_argument('--record',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    export(a.root,a.seed,a.arm,a.fold,a.record,a.output)
