"""同章节明确表头三列PFT绑定，只做训练原源审计。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.pft_evidence import ROOT as EVIDENCE
from patient_data.pft_inventory import sha256_text
from patient_data.text_time_coverage import SNAP, SCREEN
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_table_2026-10-13')
PROTOCOL=Path('docs/V6_PFT_TABLE_PROTOCOL_2026-10-13.md')
HEADER=re.compile(r'%\s*(?:predicted|pred)\b|(?:actual|observed|measured|predicted|pred)\b',re.I)
NUM=r'(?:\d+(?:\.\d+)?|\.\d+)'
ROW=re.compile(r'^\s*(FEV\s*1\s*/\s*FVC|FEV\s*1|FVC|DLCO|TLC|RV)(?:\s*\((L|mL)\))?\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s*%?\s*$',re.I)


def header_roles(line):
    matches=list(HEADER.finditer(line))
    if re.sub(r'\s','',HEADER.sub('',line)) or len(matches)!=3:return None
    roles=tuple('percent_predicted' if m.group().startswith('%') else
                'predicted' if m.group().lower() in ('pred','predicted') else 'actual' for m in matches)
    return roles if set(roles)=={'actual','predicted','percent_predicted'} else None


def row_values(line):
    match=ROW.fullmatch(line)
    if match is None:return None
    metric=re.sub(r'\s+','',match.group(1)).upper()
    if match.group(2) and metric not in ('FEV1','FVC'):return None
    return dict(metric=metric,unit_literal=match.group(2),trailing_percent=line.rstrip().endswith('%'),cells=[dict(value=float(match.group(i)),
                start=match.start(i),end=match.end(i)) for i in (3,4,5)])


def find_header(lines,index):
    """不能越过叙述/另一表/不完整行，把任意前表头当当前证据。"""
    for j in range(index-1,max(-1,index-13),-1):
        if not lines[j].strip():continue
        roles=header_roles(lines[j])
        if roles:return j,roles
        if row_values(lines[j]) is None:return None
    return None


def line_ranges(text,start,end):
    result=[];at=start
    for line in text[start:end].splitlines(keepends=True):
        result.append((at,at+len(line),line.rstrip('\r\n')));at+=len(line)
    if at!=end:raise ValueError('section line coverage differs')
    return result


def derive():
    audit=json.loads((EVIDENCE/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((EVIDENCE/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(EVIDENCE/'audit.json') or p['records_sha256']!=sha256(EVIDENCE/'records.jsonl.gz'):
            raise ValueError('PFT evidence proof differs')
    for p,h in audit['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected source changed: '+p)
    with gzip.open(EVIDENCE/'records.jsonl.gz','rt',encoding='utf-8') as f:members=[json.loads(s) for s in f]
    need={r['note_id'] for r in members}
    raw={r['note_id']:r['text'] for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in need}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:docs={r['note_id']:r for r in map(json.loads,f) if r['note_id'] in need}
    records=[];seen=set();counts=Counter();metrics=Counter();roles_count=Counter();status=Counter();new_admissions=set();all_admissions=set();patients=set()
    prior_explicit={r['hadm_id'] for r in members if any(b.get('scalar_binding_eligible',False) for b in r['bindings'])}
    for r in members:
        text=raw[r['note_id']]
        if sha256_text(text)!=r['original_sha256'] or sha256_text(text[r['start_char']:r['end_char']])!=r['quote_sha256']:
            raise ValueError('original table member differs')
        section=next(s for s in docs[r['note_id']]['section_spans'] if s['start_char']<=r['start_char']<s['end_char'])
        ranges=line_ranges(text,section['start_char'],section['end_char']);lines=[p[2] for p in ranges]
        for b in r['bindings']:
            at=r['start_char']+b['metric_start'];index=next(i for i,(a,z,_) in enumerate(ranges) if a<=at<z)
            parsed=row_values(lines[index])
            if not parsed or parsed['metric']!=b['metric']:
                counts['not_complete_supported_three_column_row']+=1;continue
            found=find_header(lines,index)
            if found is None:
                counts['no_safe_contiguous_explicit_header']+=1;continue
            hi,roles=found;start,end,line=ranges[index];hs,he,header=ranges[hi]
            if parsed['trailing_percent'] and roles[-1]!='percent_predicted':
                counts['percent_marker_column_conflict']+=1;continue
            key=(r['note_id'],start)
            if key in seen:continue
            seen.add(key);qs=[]
            for role,cell in zip(roles,parsed['cells'],strict=True):
                a=start+cell['start'];z=start+cell['end']
                if float(text[a:z])!=cell['value']:raise ValueError('table numeric span differs')
                kind=('ratio_percent_predicted' if parsed['metric']=='FEV1/FVC' else 'percent_predicted') if role=='percent_predicted' else (
                    'volume_'+parsed['unit_literal'].lower() if parsed['unit_literal'] and parsed['metric'] in ('FEV1','FVC') else 'unit_unknown')
                qs.append(dict(role=role,value=cell['value'],kind=kind,start_char=a,end_char=z,
                               unit_literal='%' if role=='percent_predicted' else parsed['unit_literal']))
                roles_count[role+'/'+kind]+=1
            records.append(dict(note_id=r['note_id'],hadm_id=r['hadm_id'],subject_id=r['subject_id'],section=r['section'],
                header_start=hs,header_end=hs+len(header),header_sha256=sha256_text(header),
                row_start=start,row_end=start+len(line),row_sha256=sha256_text(line),original_sha256=r['original_sha256'],
                column_roles=list(roles),metric=parsed['metric'],quantities=qs,measurement_time='unknown',
                previous_binding_status=b['status'],flags=r['flags']))
            metrics[parsed['metric']]+=1;status[b['status']]+=1;all_admissions.add(r['hadm_id']);patients.add(r['subject_id'])
            if r['hadm_id'] not in prior_explicit:new_admissions.add(r['hadm_id'])
    sources={**audit['source_sha256']}
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_table_binding.py'),EVIDENCE/'audit.json',EVIDENCE/'records.jsonl.gz'):
        sources[str(p)]=sha256(p)
    report=dict(status='training_explicit_table_binding_not_clinical_gold',eligible_table_rows=len(records),
                table_admissions=len(all_admissions),table_patients=len(patients),additional_admissions_with_table_evidence=len(new_admissions),
                row_metrics=dict(metrics),column_quantity_roles=dict(roles_count),previous_binding_status=dict(status),rejection_counts=dict(counts),
                source_sha256=sources,new_model_fits=0,new_vectors=0,high_frequency_used=False,outcomes_used=False,
                limits=['unknown absolute units stay unknown','column words not measurement quality gold',
                        'all measurement dates unknown','ratio percent-predicted not actual ratio','no numeric size column inference'])
    return report,records


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('table audit exists; inspect')
        ROOT.mkdir();write_json(ROOT/'audit.json',report)
        with gzip.open(ROOT/'records.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_two_source_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if saved!=report or prior!=records:raise ValueError('independent PFT table binding differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,rows=len(records),
                   audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_source_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':main()
