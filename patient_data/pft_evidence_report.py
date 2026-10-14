"""源重建合格后的肺功能证据聚合，不读取结局或改变候选规则。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from patient_data.pft_evidence import ROOT, repeat_stats
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def readout():
    for tag in ('a','b'):
        p=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(ROOT/'audit.json') or p['records_sha256']!=sha256(ROOT/'records.jsonl.gz'):
            raise ValueError('PFT evidence is not independently source-verified')
    with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    eligible=[r for r in records if any(b.get('scalar_binding_eligible',False) for b in r['bindings'])]
    flags=Counter(); types=Counter(); annotation_units=0; ambiguous=Counter()
    supported={'volume_l','volume_ml','percent_predicted','ratio_percent'}
    for r in records:
        annotation_units+=any(len(b['quantities'])>1 for b in r['bindings'])
    for r in eligible:
        flags.update({k:int(v) for k,v in r['flags'].items()})
        for b in r['bindings']:
            if not b.get('scalar_binding_eligible',False):continue
            for q in b['quantities']:
                # 显式体积后的未知括号百分比只保留审计，不纳入明确标量计数。
                target=types if q['kind'] in supported else ambiguous
                target[b['metric']+'/'+q['kind']]+=1
    return dict(eligible_units=len(eligible),eligible_time_role_flags=dict(flags),
                explicit_quantity_counts=dict(types),retained_unknown_secondary_counts=dict(ambiguous),
                annotated_numeric_units=annotation_units,
                eligible_exact_quote_repeat=repeat_stats(eligible,'quote_sha256'),
                audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz'),
                report_code_sha256=sha256(Path(__file__)),new_model_fits=0,new_vectors=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);args=p.parse_args()
    result=readout();path=ROOT/'binding_readout.json'
    if args.mode=='build':
        if path.exists():raise FileExistsError('binding readout already exists')
        write_json(path,result)
    else:
        if json.loads(path.read_text())!=result:raise ValueError('binding aggregate replay differs')
        write_json(ROOT/'binding_readout_verification.json',dict(maximum_error=0,report_sha256=sha256(path)))
    print(json.dumps(result))


if __name__=='__main__':main()
