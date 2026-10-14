"""已保存出院药物章来源库存；词面候选不是处方事实，不读结局。"""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v5.clinical_concepts import medication_classes

ROOT=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_AUDIT_PROTOCOL_2026-10-14.md')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
TASKS=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated')
SCREEN=Path('patient_data/processed/v6_rule_sentence_screen_2026-10-10_r4')
ENCODED=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10')
ALIGNED=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11')
COMPACT=Path('patient_data/processed/v6_compact_multisource_2026-10-09')
CLASSES=('inhaled_steroid','laba','lama')
FORM=re.compile(r'\b(?:inhal\w*|puffs?|nebul\w*|neb|diskus|ellipta|respimat|HFA|aerosol|MDI|DPI|respules?)\b',re.I)
ORAL=re.compile(r'\b(?:PO|by mouth|orally)\b',re.I)
FLAGS={
    'stop_hold_words':r'\b(?:stop\w*|discontinu\w*|hold|held|do not take)\b',
    'conditional_words':r'\b(?:if|unless|as needed|PRN)\b',
    'history_words':r'\b(?:previous\w*|prior|formerly|used to)\b',
    'dose_string_present':r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|puffs?|mL)\b',
    'frequency_string_present':r'\b(?:daily|BID|TID|QID|q\s*\d+\s*h|twice|once|three times|four times)\b',
}


def lexical(quote):
    # 此处name是保存的原句，只能称词面候选；不输出prescribed/stopped事实值。
    route=('PO',) if ORAL.search(quote) and not FORM.search(quote) else ()
    groups=medication_classes(quote,product_descriptions=(quote,),routes=route,
                              mapping_version='clinical_reviewed_2026_09')
    return dict(classes=sorted(groups&set(CLASSES)),inhaled_form_word=bool(FORM.search(quote)),
                explicit_oral_word=bool(ORAL.search(quote)),
                flags=[k for k,p in FLAGS.items() if re.search(p,quote,re.I)],clinically_verified=False)


def span(text,a,b,quote):
    if not 0<=a<=b<=len(text) or text[a:b]!=quote:raise ValueError('invalid saved source offset/quote')


def check_maps(source,task):
    spans=[]
    for m in task['source_maps']:
        if m['role']!='discharge_medications':continue
        a,b=m['original_start'],m['original_end'];ia,ib=m['input_start'],m['input_end']
        if not 0<=ia<=ib<=len(task['text']):raise ValueError('task input bounds invalid')
        span(source['text'],a,b,task['text'][ia:ib])
        if not any(s['start_char']<=a<=b<=s['end_char'] for s in source['sections'] if s['role']=='discharge_medications'):
            raise ValueError('map not within medication section')
        spans.append((a,b))
    return spans


def covered(section,ranges):
    a,b=section['start_char'],section['end_char'];rr=sorted((x,y) for x,y in ranges if a<=x<=y<=b)
    right=a
    for x,y in rr:
        if x!=right or not x<y:raise ValueError('medication map gap/overlap')
        right=y
    if right!=b:raise ValueError('medication section not fully task-covered')


def load_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def derive():
    started=time.perf_counter();sources={}
    def protect(p,expected=None):
        h=sha256(p)
        if expected is not None and h!=expected:raise ValueError('source hash changed: '+str(p))
        sources[str(p)]=h;return h
    manifests={}
    for directory in (STATE,TASKS,SCREEN,ENCODED,ALIGNED,COMPACT):
        protect(directory/'manifest.json');manifests[str(directory)]=json.loads((directory/'manifest.json').read_text())
    st,tm,rm,em,am,cm=(manifests[str(p)] for p in (STATE,TASKS,SCREEN,ENCODED,ALIGNED,COMPACT))
    if tm['parent_state_manifest_sha256']!=sources[str(STATE/'manifest.json')] or cm['parent_state_manifest_sha256']!=sources[str(STATE/'manifest.json')]:
        raise ValueError('parent member source differs')
    if tm['status']!='llm_tasks_source_checked' or rm['status']!='completed_rule_candidates_not_semantic_gold' or em['status']!='candidate_bert_complete':
        raise ValueError('source unfinished')
    for directory,names,manifest in (
        (STATE,('clinical_state.npz',),st),(TASKS,('sources.jsonl','tasks.jsonl','members.npz'),tm),
        (SCREEN,('selected_sentences.jsonl.gz',),rm),(ENCODED,('chunks.jsonl.gz',),em),
        (ALIGNED,('alignment.npz',),am),(COMPACT,('compact_context.npz',),cm)):
        for name in names:protect(directory/name,manifest['output_sha256'][name])
    protect(ENCODED/'independent_encoding_verification.json')
    ep=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    if ep['manifest_sha256']!=sources[str(ENCODED/'manifest.json')]:raise ValueError('old encoding proof differs')
    state=load_npz(STATE/'clinical_state.npz',('hadm_ids','subject_ids','patient_splits'))
    ti=np.flatnonzero(state['patient_splits']=='train');n=len(ti)
    if len(state['hadm_ids'])!=19401 or n!=13592:raise ValueError('cohort drift')
    people={str(h):str(s) for h,s in zip(state['hadm_ids'][ti],state['subject_ids'][ti])};lookup={str(h):i for i,h in enumerate(state['hadm_ids'][ti])}
    for directory,filename in ((TASKS,'members.npz'),(COMPACT,'compact_context.npz'),(ALIGNED,'alignment.npz')):
        a=load_npz(directory/filename,tuple(state))
        for k,v in state.items():np.testing.assert_array_equal(a[k],v)
    compact=load_npz(COMPACT/'compact_context.npz',('med_features','med_group_names'))
    med_indices=[list(compact['med_group_names']).index(x) for x in ('inhaled_corticosteroid','long_beta_agonist','long_muscarinic_antagonist')]
    old=(compact['med_features'][ti][:,med_indices,:2]>0).any(2)
    summaries={};sections={}
    with (TASKS/'sources.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['channel']!='discharge' or r['hadm_id'] not in people:continue
            if r['subject_id']!=people[r['hadm_id']] or r['note_id'] in summaries:raise ValueError('source subject/duplicate differs')
            if hashlib.sha256(r['text'].encode()).hexdigest()!=r['original_sha256']:raise ValueError('saved note SHA differs')
            ss=[s for s in r['sections'] if s['role']=='discharge_medications']
            for s in ss:span(r['text'],s['start_char'],s['end_char'],s['text'])
            summaries[r['note_id']]=r;sections[r['note_id']]=ss
    taskranges=defaultdict(list);tasks_count=0
    with (TASKS/'tasks.jsonl').open(encoding='utf-8') as f:
        for task in map(json.loads,f):
            if task['route']!='discharge_all' or task['note_id'] not in summaries:continue
            source=summaries[task['note_id']]
            for k in ('hadm_id','subject_id','original_sha256'):
                if task[k]!=source[k]:raise ValueError('task source identity differs')
            ranges=check_maps(source,task);taskranges[task['note_id']].extend(ranges);tasks_count+=bool(ranges)
    for note,ss in sections.items():
        for s in ss:covered(s,taskranges[note])
    encoded_units=set()
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['patient_split']!='train':continue
            if r['hadm_id'] not in people or r['subject_id']!=people[r['hadm_id']]:raise ValueError('encoded membership differs')
            encoded_units.update((r['note_id'],int(a),int(b)) for a,b in r['source_units'])
    evidence=[];counts=Counter();named=np.zeros((n,3),bool);with_form=np.zeros_like(named);available=np.zeros(n,bool)
    for note,ss in sections.items():
        if ss:available[lookup[summaries[note]['hadm_id']]]=True
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['kind']!='discharge' or r['patient_split']!='train' or r['section']!='discharge_medications':continue
            if r['hadm_id'] not in people or r['subject_id']!=people[r['hadm_id']] or r['note_id'] not in summaries:raise ValueError('sentence member differs')
            source=summaries[r['note_id']];a,b=r['start_char'],r['end_char'];span(source['text'],a,b,r['quote'])
            if r['original_sha256']!=source['original_sha256'] or not any(s['start_char']<=a<=b<=s['end_char'] for s in sections[r['note_id']]):
                raise ValueError('sentence section/identity differs')
            counts['saved_selected_medication_units']+=1;lex=lexical(r['quote'])
            if not lex['classes']:continue
            i=lookup[r['hadm_id']];encoded=(r['note_id'],a,b) in encoded_units
            counts['maintenance_mapping_candidate_units']+=1;counts['candidate_units_in_old_encoding_groups']+=encoded
            counts['candidate_units_with_inhaled_form_words']+=lex['inhaled_form_word']
            counts.update(lex['flags'])
            for group in lex['classes']:
                j=CLASSES.index(group);named[i,j]=True;with_form[i,j]|=lex['inhaled_form_word']
            evidence.append(dict(**{k:r[k] for k in ('hadm_id','subject_id','note_id','unit_key','start_char','end_char','quote','original_sha256')},
                lexical=lex,in_old_encoding_group=encoded))
    aligned=load_npz(ALIGNED/'alignment.npz',('refs',));med_role=am['roles'].index('discharge_medications')
    apositions={int(p) for p,f,_ in aligned['refs'] if int(f)==med_role and state['patient_splits'][int(p)]=='train'}
    prior={}
    for folder in ('v6_bonsai_discharge_pilot_2026-10-10_r3','v6_short_topics_results_96_2026-10-10_r2'):
        p=Path('patient_data/processed')/folder;protect(p/'status.json');protect(p/'contract.json')
        status=json.loads((p/'status.json').read_text());contract=json.loads((p/'contract.json').read_text())
        prior[folder]=dict(status=status['status'],total=status['total'],successful=status['successful'],limit=contract['limit'])
    pilot=Path('patient_data/processed/v6_short_topic_audit_96_2026-10-10_r2/audit.json');protect(pilot)
    audit=json.loads(pilot.read_text());prior['96_note_postcheck']=dict(notes=audit['notes'],tasks=audit['tasks'],
        accepted_after_postcheck=audit['accepted_after_postcheck'],maintenance_field_counts={k:audit['by_field'].get(k,0) for k in ('inhaled_ics','inhaled_laba','inhaled_lama')})
    cxr=Path('training_runs/v6_cxr_trajectory_seed42_2026-10-10_r2');protect(cxr/'config.json');protect(cxr/'contract.json')
    cfg=json.loads((cxr/'config.json').read_text());cc=json.loads((cxr/'contract.json').read_text())
    if cfg['facts_dir'] is not None or cc['available_channels']!=['cxr']:raise ValueError('prior full facts changed; inspect before repeating')
    byclass={group:dict(candidate_admissions=int(named[:,j].sum()),candidate_patients=len(np.unique(state['subject_ids'][ti][named[:,j]])),
        with_inhaled_form_word_admissions=int(with_form[:,j].sum()),
        old_compact_class_zero_despite_candidate_admissions=int((named[:,j]&~old[:,j]).sum())) for j,group in enumerate(CLASSES)}
    # 只有候选库存，绝不把这些数组当已验证的处方标签。
    arrays=dict(hadm_ids=state['hadm_ids'][ti],subject_ids=state['subject_ids'][ti],section_available=available,
                mapping_mentions=named,inhaled_form_word_mentions=with_form,old_compact_presence=old)
    reference_contract=Path('training_runs/v6_core_lab_cox_2026-10-14/contract.json');protect(reference_contract)
    reference=json.loads(reference_contract.read_text())
    teacher=Path('training_runs/v6_module_surgery_2026-10-12_r3/seed42/control/best.pt')
    protect(teacher,reference['parent_teacher_sha256'])
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_discharge_regimen_audit.py'),
        Path('training/v5/clinical_concepts.py'),Path('training/v6/clinical_facts_context.py'),
        Path('patient_data/clinical_facts/catalog.py'),Path('patient_data/clinical_facts/features.py')):protect(p)
    report=dict(status='train_only_saved_discharge_medication_source_inventory_not_fact_interface',
        source_sha256=sources,training_admissions=n,training_patients=len(np.unique(state['subject_ids'][ti])),
        saved_training_summary_notes=len(summaries),medication_section_notes=sum(bool(s) for s in sections.values()),
        medication_section_admissions=int(available.sum()),medication_section_patients=len(np.unique(state['subject_ids'][ti][available])),
        medication_sections=sum(len(s) for s in sections.values()),medication_tasks_with_maps=tasks_count,
        aligned_A_training_medication_admissions=len(apositions),counts=dict(counts),by_class=byclass,
        candidate_any_maintenance_admissions=int(named.any(1).sum()),
        old_fact_channel_experiment=dict(facts_dir=cfg['facts_dir'],available_channels=cc['available_channels']),
        inherited_frozen_reference=dict(teacher_sha256=reference['parent_teacher_sha256'],
            base_sha256=reference['parent_base_sha256'],data_provenance=reference['data_provenance'],
            label_array_sha256={k:reference['array_sha256'][k] for k in ('times','events')}),
        existing_pilots=prior,new_fits=0,new_vectors=0,new_llm_calls=0,no_outcomes_read=True,
        raw_discharge_read=False,source_maps_fully_covered=True,actual_seconds=time.perf_counter()-started,
        limits=['drug mapping words/form words are not confirmed prescribed/stopped facts or clinical gold',
            'old compact zero includes unobserved source; discordance is not proof of missing administration',
            'source_units identifies encoding-group membership, not per-chunk token inclusion or BERT semantics',
            'FINAL is retrospective; no real-time availability established',
            'B has no independent medication role but course may share information'])
    return report,arrays,evidence


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat')
    ROOT.mkdir();r,a,e=derive();np.savez_compressed(ROOT/'audit_arrays.npz',**a)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
        for row in e:f.write(json.dumps(row)+'\n')
    r['output_sha256']={k:sha256(ROOT/k) for k in ('audit_arrays.npz','local_evidence.jsonl.gz')}
    write(ROOT/'audit.json',r);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','output_sha256')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('audit artifact changed')
    if old!=new:raise ValueError('source/statistics differ')
    with np.load(ROOT/'audit_arrays.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        saved=list(map(json.loads,f))
    if saved!=e:raise ValueError('evidence differs')
    write(ROOT/'verification.json',dict(status='one_independent_saved_source_map_inventory_rebuild',maximum_error=0.,
        audit_sha256=sha256(ROOT/'audit.json'),actual_seconds=seconds,new_fits=0,new_vectors=0,raw_discharge_read=False))
    write(ROOT/'state.json',dict(status='source_inventory_independently_verified',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
