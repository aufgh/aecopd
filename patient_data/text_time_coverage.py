"""训练折原句→原B/既有候选覆盖；不训练、不按结局筛选、不输出原文到日志。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.sentence_rules import COMPILED,WARNING,FUTURE_OUTCOME
from training.v6.encode_screen_candidates import FUTURE
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_text_time_coverage_2026-10-13')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
ASSETS=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/v5_section_aligned')
B=ASSETS/'field_text_baseline_2026_09_27'
SCREEN=Path('patient_data/processed/v6_rule_sentence_screen_2026-10-10_r4')
ENCODED=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10')
PROTOCOL=Path('docs/V6_TEXT_TIME_COVERAGE_AUDIT_PROTOCOL_2026-10-13.md')
ROLES={'hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues','followup_instructions'}
DISCHARGE=re.compile(r'\b(?:(?:at|on|upon|by|before|prior to|time of)\s+(?:the\s+)?discharge|discharged|discharging)\b',re.I)
CHANGE=re.compile(r'\b(?:improv\w*|wean\w*|resolv\w*|persist\w*|still|no longer|(?:back|return\w*)\s+to\s+(?:her|his|their|the)?\s*baseline)\b',re.I)
PLAN=re.compile(r'\b(?:will|plan\w*|recommend\w*|should|advis\w*|continue|to complete|follow.?up)\b',re.I)
HISTORY=re.compile(r'\b(?:history of|baseline|prior|previous\w*|at home)\b',re.I)
POST_DISCHARGE=re.compile(r'\b(?:after\s+(?:the\s+)?discharge|post[- ]discharge)\b',re.I)


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def time_flags(text,role):
    respiratory=bool(COMPILED['respiratory_status'].search(text) or COMPILED['oxygen_ventilation'].search(text))
    flags=dict(respiratory=respiratory,approved_role=role in ROLES,explicit_discharge=bool(DISCHARGE.search(text)),
        change_or_residual=bool(CHANGE.search(text)),plan=bool(PLAN.search(text)),historical=bool(HISTORY.search(text)),
        warning=bool(WARNING.search(text)),future_outcome=bool(FUTURE_OUTCOME.search(text)),post_discharge_words=bool(POST_DISCHARGE.search(text)))
    flags['priority_candidate']=respiratory and flags['approved_role'] and (flags['explicit_discharge'] or flags['change_or_residual']) and not flags['warning'] and not flags['future_outcome'] and not flags['post_discharge_words']
    return flags


def span(original,start,end,quote):
    if not 0<=start<end<=len(original) or original[start:end]!=quote:raise ValueError('original quote offset differs')


def coverage(start,end,intervals):
    if not 0<=start<end:raise ValueError('nonempty forward span required')
    pieces=sorted((max(start,a),min(end,b)) for a,b in intervals if a<end and b>start)
    total=0;right=start
    for a,b in pieces:
        if not 0<=a<b:raise ValueError('invalid intersecting interval')
        total+=max(0,b-max(a,right));right=max(right,b)
    return total/(end-start),any(a<=start and end<=b for a,b in intervals)


def checked_sources():
    sm=json.loads((SNAP/'metadata/manifest.json').read_text());state=json.loads((STATE/'manifest.json').read_text())
    bm=json.loads((B/'summary.json').read_text());bs=json.loads((B/'bert/state.json').read_text())
    qa=json.loads((ASSETS/'dataset_v4/qa_validation.json').read_text());screen=json.loads((SCREEN/'manifest.json').read_text())
    audit=json.loads((SCREEN/'source_audit.json').read_text());em=json.loads((ENCODED/'manifest.json').read_text())
    proof=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    checks={SNAP/'raw/discharge.csv.gz':sm['sources']['raw/discharge']['sha256'],
        SNAP/'cohort/admissions.csv.gz':sm['sources']['cohort/admissions']['sha256'],STATE/'clinical_state.npz':state['output_sha256']['clinical_state.npz'],
        ASSETS/'dataset_v4/note_sections.csv.gz':qa['output_sha256']['note_sections.csv.gz'],B/'shared_text.csv.gz':bm['manifest_sha256']}
    checks.update({SCREEN/k:v for k,v in screen['output_sha256'].items()});checks.update({ENCODED/k:v for k,v in em['output_sha256'].items()})
    if sm['status']!='validated' or bs['status']!='complete' or bs['manifest_sha256']!=bm['manifest_sha256']:raise ValueError('original source/encoding incomplete')
    if (audit['status']!='all_original_quotes_contexts_and_section_coverage_checked' or audit['manifest_sha256']!=sha256(SCREEN/'manifest.json') or
        em['source_manifest_sha256']!=audit['manifest_sha256'] or proof['manifest_sha256']!=sha256(ENCODED/'manifest.json')):raise ValueError('candidate source proof differs')
    if sha256(ASSETS/'dataset_v4/note_sections.csv.gz')!=bm['cohort_source_sections_sha256']:raise ValueError('B section lineage changed')
    for p,h in checks.items():
        if sha256(p)!=h:raise ValueError('source changed: '+str(p))
    # 额外保留原完整V6依赖的已记录来源指纹，无训练或结局分析。
    for parent in ('v6_module_surgery_2026-10-12_r3','v6_regularization_search_2026-10-12'):
        m=json.loads((Path('training_runs')/parent/'contract.json').read_text())
        for p,h in m['source_sha256'].items():
            if sha256(Path(p))!=h:raise ValueError('protected old source changed: '+p)
        if sha256(B/'bert/bioclinicalbert.float16.npy')!=m['data_provenance']['text']['vectors_sha256']:raise ValueError('original V6 B vectors changed')
    sources={str(p):h for p,h in checks.items()}
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_text_time_coverage.py'),B/'bert/state.json',B/'bert/bioclinicalbert.float16.npy',
              SCREEN/'manifest.json',SCREEN/'source_audit.json',ENCODED/'manifest.json',ENCODED/'independent_encoding_verification.json'):
        sources[str(p)]=sha256(p)
    return sources,bs


def derive():
    sources,bs=checked_sources()
    with np.load(STATE/'clinical_state.npz') as z:
        hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    training={str(h):str(s) for h,s,f in zip(hadms,people,splits) if f=='train'}
    if len(training)!=13592:raise ValueError('training split differs')
    raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['hadm_id'] in training}
    raw_hashes={note:digest(r['text']) for note,r in raw.items()}
    cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz') if r['hadm_id'] in training}
    sections={}
    for r in rows(ASSETS/'dataset_v4/note_sections.csv.gz'):
        if r['hadm_id'] not in training:continue
        original=raw[r['note_id']];a,b=int(r['start_char']),int(r['end_char'])
        if not 0<=a<=b<=len(original['text']) or original['text'][a:b]!=r['text']:raise ValueError('original section offset differs')
        if r['source_sha256']!=raw_hashes[r['note_id']]:raise ValueError('B original document changed')
        sections[r['section_id']]=r
    intervals=defaultdict(list);bchunks={};bcounts=Counter();bnotes=set();max_tokens=0
    for j,r in enumerate(rows(B/'shared_text.csv.gz')):
        if int(r['embedding_row'])!=j:raise ValueError('B rows reordered')
        if r['hadm_id'] not in training:continue
        if r['subject_id']!=training[r['hadm_id']]:raise ValueError('B subject/fold mismatch')
        sec=sections[r['section_id']];note=sec['note_id'];a,b=int(r['source_start_char']),int(r['source_end_char'])
        original=raw[note]['text'];quote=original[a:b]
        span(original,a,b,quote)
        if digest(quote)!=r['source_chunk_sha256'] or r['chunk_text'].partition('\n')[2]!=quote or digest(r['chunk_text'])!=r['text_sha256']:
            raise ValueError('B prefix/source content mismatch')
        if r['source_section_sha256']!=raw_hashes[note]:raise ValueError('B document fingerprint differs')
        intervals[note].append((a,b,j));bchunks[j]=r;bnotes.add(note);bcounts[r['field']]+=1;max_tokens=max(max_tokens,int(r['bert_token_count']))
    # 候选编码索引source_units属于整个组，不把它冒充每个块的内容。
    candidate_units=set();encoded_notes=set();encoded_groups={};encoded_train_chunks=0
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['patient_split']!='train':continue
            if r['hadm_id'] not in training or r['subject_id']!=training[r['hadm_id']]:raise ValueError('candidate encoding fold differs')
            encoded_train_chunks+=1;encoded_notes.add(r['note_id']);key=(r['note_id'],r['field'])
            encoded_groups.setdefault(key,[]).append((r['token_start'],r['token_end']))
            candidate_units.update((r['note_id'],int(a),int(b)) for a,b in r['source_units'])
    for ranges in encoded_groups.values():
        right=0
        for a,b in sorted(ranges):
            if a!=right or not a<b:raise ValueError('candidate token chunks gap/overlap')
            right=b
    counters=Counter();roles=defaultdict(Counter);targets=[];banned=set();time_patterns=Counter();note_set=set()
    print(json.dumps(dict(stage='original_and_B_checked',training_B_chunks=len(bchunks),training_B_notes=len(bnotes))),flush=True)
    with gzip.open(SCREEN/'all_units.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['kind']!='discharge' or r['patient_split']!='train':continue
            if r['hadm_id'] not in training or r['subject_id']!=training[r['hadm_id']]:raise ValueError('candidate fold/subject mismatch')
            original=raw[r['note_id']]
            if r['original_sha256']!=raw_hashes[r['note_id']]:raise ValueError('candidate original document changed')
            a,b=int(r['start_char']),int(r['end_char']);span(original['text'],a,b,r['quote'])
            counters['all_training_discharge_units']+=1;note_set.add(r['note_id'])
            if r['reason']=='observed_future_outcome' or FUTURE.search(r['quote']):banned.add((r['note_id'],r['section_ordinal']))
            flags=time_flags(r['quote'],r['section'])
            if not (flags['approved_role'] and flags['respiratory']):continue
            counters['respiratory_units_in_candidate_roles']+=1
            for k,v in flags.items():
                if v:counters['lexical:'+k]+=1
            roles[r['section']]['respiratory_units']+=1
            if not flags['priority_candidate']:continue
            span_pairs=[(x,y) for x,y,_ in intervals[r['note_id']]];fraction,single=coverage(a,b,span_pairs)
            containing=[j for x,y,j in intervals[r['note_id']] if x<=a and b<=y]
            hit=(r['note_id'],a,b) in candidate_units
            pattern='+'.join(k for k in ('explicit_discharge','change_or_residual','plan','historical') if flags[k]);time_patterns[pattern]+=1
            record={k:r[k] for k in ('note_id','hadm_id','subject_id','section','section_ordinal','start_char','end_char','unit_key','reason')}
            record.update(original_sha256=r['original_sha256'],quote_sha256=digest(r['quote']),flags=flags,selected=r['selected'],
                B_union_character_coverage=fraction,B_single_chunk_full=single,B_containing_rows=containing,
                B_same_document_present=r['note_id'] in bnotes,encoded_candidate_source_unit=hit,
                quote_chars=len(r['quote']),min_containing_B_chars=min((len(bchunks[j]['chunk_text'].partition('\n')[2]) for j in containing),default=None),
                source_stored_after_discharge=datetime.fromisoformat(original['storetime'])>datetime.fromisoformat(cohort[r['hadm_id']]['dischtime']))
            targets.append(record)
    # 完整章节隔离在全文扫描完成后确定，不能先看到安全句就当章节安全。
    for r in targets:
        r['candidate_section_quarantined']=(r['note_id'],r['section_ordinal']) in banned
        if r['encoded_candidate_source_unit'] and r['candidate_section_quarantined']:raise ValueError('quarantined target enters encoded source')
        c=roles[r['section']];c['priority_candidates']+=1
        for name,value in [('B_single_chunk_full',r['B_single_chunk_full']),('B_union_full',r['B_union_character_coverage']==1),
            ('B_no_character_coverage',r['B_union_character_coverage']==0),('B_other_document_or_absent',not r['B_same_document_present']),
            ('candidate_selected',r['selected']),('encoded_candidate_source_unit',r['encoded_candidate_source_unit']),
            ('candidate_section_quarantined',r['candidate_section_quarantined']),('source_stored_after_discharge',r['source_stored_after_discharge'])]:
            if value:counters[name]+=1;c[name]+=1
    fractions=[r['quote_chars']/r['min_containing_B_chars'] for r in targets if r['min_containing_B_chars']]
    result=dict(status='training_only_original_time_coverage_audit_not_semantic_gold',source_sha256=sources,training_admissions=13592,
        training_summary_documents=len(note_set),training_B_chunks=len(bchunks),training_B_notes=len(bnotes),B_chunks_by_field=dict(bcounts),
        B_training_max_recorded_token_count=max_tokens,B_encoding_state=bs,training_encoded_candidate_chunks=encoded_train_chunks,
        counters=dict(counters),by_role={k:dict(v) for k,v in roles.items()},lexical_time_pattern_counts=dict(time_patterns),
        priority_candidate_units=len(targets),priority_candidate_admissions=len({r['hadm_id'] for r in targets}),
        priority_candidate_chars_as_fraction_of_containing_B_chunk_quantiles=np.quantile(fractions,[0,.1,.5,.9,1]).tolist() if fractions else None,
        unencoded_priority_candidate_reasons=dict(Counter(r['reason'] for r in targets if not r['encoded_candidate_source_unit'])),
        limits=['No outcomes used or new fitting','Lexical candidate not verified discharge fact or semantic precision/recall',
            'already encoded does not prove learned semantic importance','small content fraction does not prove pooling is performance root cause',
            'candidate source_units identifies whole group; no individual block containment claim','FINAL retrospective including later storage not real-time availability'])
    return result,targets,bchunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',default='a',choices=('a','b'));a=p.parse_args()
    report,records,_=derive()
    if a.mode=='build':
        if ROOT.exists():raise FileExistsError('audit directory exists; inspect')
        ROOT.mkdir(parents=True)
        with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','xt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_json(ROOT/'audit.json',report)
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        if saved!=report:raise ValueError('source reconstruction/report differs')
        with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','rt',encoding='utf-8') as f:
            if list(map(json.loads,f))!=records:raise ValueError('source audit index differs')
        write_json(ROOT/f'verification_{a.tag}.json',dict(status='independent_source_rebuild_verified',report_sha256=sha256(ROOT/'audit.json'),
            index_sha256=sha256(ROOT/'priority_candidate_index.jsonl.gz'),maximum_count_or_index_error=0,training_only=True,no_model_fit=True))
    print(json.dumps({k:report[k] for k in ('status','counters','by_role','priority_candidate_units','priority_candidate_admissions',
        'priority_candidate_chars_as_fraction_of_containing_B_chunk_quantiles','unencoded_priority_candidate_reasons')}),flush=True)


if __name__=='__main__':main()
