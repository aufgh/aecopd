"""已核验肺功能记录→六概念训练集合，保留全部证据，不平均/选最新。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from patient_data.pft_evidence import ROOT as EVIDENCE
from patient_data.pft_table_binding import ROOT as TABLE
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_fact_ledger_2026-10-13_r2')
PROTOCOL=Path('docs/V6_PFT_LEDGER_PROTOCOL_2026-10-13.md')
CONCEPTS=('fev1_volume_l','fvc_volume_l','fev1_percent_predicted','fvc_percent_predicted','fev1_fvc_ratio','dlco_percent_predicted')


def canonical(metric,kind,value):
    if kind in ('volume_l','volume_ml') and metric in ('FEV1','FVC'):
        return metric.lower()+'_volume_l',value/(1000 if kind=='volume_ml' else 1),'L'
    if kind=='percent_predicted' and metric in ('FEV1','FVC','DLCO'):
        return metric.lower()+'_percent_predicted',value,'%predicted'
    if kind=='ratio_percent' and metric=='FEV1/FVC':
        return 'fev1_fvc_ratio',value/100,'ratio'
    return None


def add_fact(bag,hadm,subject,metric,kind,value,reference):
    item=canonical(metric,kind,value)
    if item is None:return False
    concept,converted,unit=item;key=(hadm,concept,converted)
    if key not in bag:
        bag[key]=dict(hadm_id=hadm,subject_id=subject,concept=concept,value=converted,unit=unit,
                      measurement_time='unknown',current_discharge_measurement='unknown',evidence=[])
    if bag[key]['subject_id']!=subject:raise ValueError('patient membership differs')
    if reference not in bag[key]['evidence']:bag[key]['evidence'].append(reference)
    return True


def checked_parent(directory):
    audit=json.loads((directory/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((directory/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(directory/'audit.json') or p['records_sha256']!=sha256(directory/'records.jsonl.gz'):
            raise ValueError('parent proof differs')
    for name,digest in audit['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('protected parent source differs: '+name)
    with gzip.open(directory/'records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    return audit,records


def derive():
    bag={};counts=Counter();sources={}
    for parent in (EVIDENCE,TABLE):
        audit,records=checked_parent(parent);sources.update(audit['source_sha256'])
        for r in records:
            if parent==EVIDENCE:
                use=[(b['metric'],q) for b in r['bindings'] if b.get('scalar_binding_eligible',False) for q in b['quantities']]
                for metric,q in use:
                    reference=dict(parent='lexical',note_id=r['note_id'],original_sha256=r['original_sha256'],
                        quote_start=r['start_char'],quote_end=r['end_char'],quote_sha256=r['quote_sha256'],
                        value_start=r['start_char']+q['value_start'],value_end=r['start_char']+q['value_end'],
                        evidence_start=r['start_char']+q['evidence_start'],evidence_end=r['start_char']+q['evidence_end'],
                        evidence_sha256=q['evidence_sha256'],flags=r['flags'])
                    counts['source_quantities_seen']+=1
                    if add_fact(bag,r['hadm_id'],r['subject_id'],metric,q['kind'],q['value'],reference):counts['accepted_source_quantities']+=1
                    else:counts['unsupported_quantity_preserved_only_in_parent']+=1
            else:
                for q in r['quantities']:
                    counts['source_quantities_seen']+=1
                    if q['role']=='predicted':
                        counts['predicted_absolute_quantity_not_actual']+=1;continue
                    reference=dict(parent='table',note_id=r['note_id'],original_sha256=r['original_sha256'],
                        row_start=r['row_start'],row_end=r['row_end'],row_sha256=r['row_sha256'],
                        header_start=r['header_start'],header_end=r['header_end'],header_sha256=r['header_sha256'],
                        value_start=q['start_char'],value_end=q['end_char'],flags=r['flags'])
                    if add_fact(bag,r['hadm_id'],r['subject_id'],r['metric'],q['kind'],q['value'],reference):counts['accepted_source_quantities']+=1
                    else:counts['unsupported_quantity_preserved_only_in_parent']+=1
        for name in ('audit.json','records.jsonl.gz','verification_a.json','verification_b.json'):
            sources[str(parent/name)]=sha256(parent/name)
    facts=[bag[key] for key in sorted(bag)]
    for r in facts:r['evidence'].sort(key=lambda e:(e['note_id'],e['value_start'],e['parent']))
    per_concept=Counter(r['concept'] for r in facts)
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_fact_ledger.py')):sources[str(p)]=sha256(p)
    report=dict(status='training_background_PFT_set_not_model_or_gold',concepts=list(CONCEPTS),
        unique_members=len(facts),admissions=len({r['hadm_id'] for r in facts}),patients=len({r['subject_id'] for r in facts}),
        members_per_concept=dict(per_concept),assembly_counts=dict(counts),
        members_with_multiple_evidence=sum(len(r['evidence'])>1 for r in facts),source_sha256=sources,
        outcomes_used=False,high_frequency_used=False,new_model_fits=0,new_vectors=0,normalization_fitted=False,
        limits=['train only; not full cohort model input','all measurement times unknown',
                'no cross-date mean/latest/recovery','evidence multiplicity not count of tests',
                'canonical unit conversion not medical measurement validation'])
    return report,facts


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('ledger exists; inspect')
        ROOT.mkdir();write_json(ROOT/'manifest.json',report)
        with gzip.open(ROOT/'facts.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_parent_record_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text())
        with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if saved!=report or prior!=records:raise ValueError('independent fact ledger replay differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,members=len(records),
                   manifest_sha256=sha256(ROOT/'manifest.json'),facts_sha256=sha256(ROOT/'facts.jsonl.gz'),parent_records_rebuilt=True))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_parent_record_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':main()
