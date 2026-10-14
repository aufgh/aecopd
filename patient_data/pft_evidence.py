"""仅训练文本肺功能原源绑定和跨住院重复审计，不提供临床gold。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.pft_inventory import ROOT as INVENTORY, NAME, HISTORY, PLAN, PRE, POST, sha256_text
from patient_data.text_time_coverage import SCREEN, SNAP, checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v6.train import write_json

ROOT = Path('patient_data/processed/v6_pft_evidence_2026-10-13')
PROTOCOL = Path('docs/V6_PFT_EVIDENCE_PROTOCOL_2026-10-13.md')
NUM = r'(?:\d+(?:\.\d+)?|\.\d+)'
FIRST = re.compile(r'^\s*(?:(?:was|is|of)\b\s*|[:=]\s*)?([<>]=?\s*)?('+NUM+r')(?![\d.])', re.I)
UNIT = re.compile(r'^\s*(%|ml\b|l\b|lit(?:er|re)s?\b)', re.I)
PRED = r'(?:of\s+)?(?:predicted|pred\.?)(?!\w)'
AFTER_PRED = re.compile(r'^\s*'+PRED, re.I)
ANNOTATION = re.compile(r'^\s*\(\s*('+NUM+r')\s*%\s*('+PRED+r')?\s*\)', re.I)
DATE = re.compile(r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b')
YEAR = re.compile(r'\b(?:19|20|21)\d{2}\b')
REDACTED = re.compile(r'_{3,}|\[\*\*.*?\*\*\]')


def bindings(text):
    """边界/明确词面量纲绑定；不推断生理值、列顺序或检测时间。"""
    result = []; names = list(NAME.finditer(text))
    for i, name in enumerate(names):
        end = names[i+1].start() if i+1 < len(names) else len(text)
        tail = text[name.end():end]; first = FIRST.match(tail)
        metric = re.sub(r'\s+', '', name.group()).upper()
        row = dict(metric=metric, metric_start=name.start(), segment_end=end,
                   status='no_adjacent_numeric', quantities=[])
        if first is None:
            result.append(row); continue
        a = name.end()+first.start(2); b = name.end()+first.end(2); after = text[b:end]
        qualifier = (first.group(1) or '').strip(); unit = UNIT.match(after)
        range_match = re.match(r'^\s*[-–]\s*'+NUM, after)
        date_number = bool(DATE.match(text[a:end])) or bool(YEAR.fullmatch(first.group(2)))
        multi = bool(re.match(r'^\s+'+NUM, after))
        reason = ('date_like_number' if date_number else 'range_ambiguous' if range_match else
                  'bound_not_scalar' if qualifier else 'unlabeled_numeric_columns' if multi else None)
        literal = unit.group(1) if unit else None; kind = 'unit_unknown'
        unit_end = b+unit.end() if unit else b
        predicted = bool(AFTER_PRED.match(text[unit_end:end]))
        if literal == '%':
            kind = ('ratio_percent_predicted' if predicted else 'ratio_percent') if metric == 'FEV1/FVC' else (
                    'percent_predicted' if predicted else 'percent_unspecified')
        elif literal:
            kind = 'volume_ml' if literal.lower() == 'ml' else 'volume_l'
            if metric not in ('FEV1', 'FVC'):
                reason = reason or 'volume_unit_metric_mismatch'
        item = dict(value=float(first.group(2)), kind=kind, unit_literal=literal, qualifier=qualifier,
                    value_start=a, value_end=b, evidence_start=name.start(), evidence_end=unit_end,
                    evidence_sha256=sha256_text(text[name.start():unit_end]))
        row['quantities'].append(item)
        if not reason and literal and literal != '%':
            annotation = ANNOTATION.match(text[unit_end:end])
            if annotation:
                pa = unit_end+annotation.start(1); pb = unit_end+annotation.end(1)
                row['quantities'].append(dict(value=float(annotation.group(1)),
                    kind='percent_predicted' if annotation.group(2) else 'percent_unspecified', unit_literal='%',
                    qualifier='', value_start=pa, value_end=pb, evidence_start=unit_end,
                    evidence_end=unit_end+annotation.end(), evidence_sha256=sha256_text(text[unit_end:unit_end+annotation.end()])))
        ready = reason is None and kind in ('volume_l', 'volume_ml', 'percent_predicted', 'ratio_percent')
        row.update(status=reason or ('explicit_quantity_lexical' if ready else 'quantity_semantics_unknown'),
                   scalar_binding_eligible=ready)
        result.append(row)
    return result


def repeat_stats(records, key):
    groups = defaultdict(set)
    for row in records:
        groups[(row['subject_id'], row[key])].add(row['hadm_id'])
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    return dict(cross_admission_groups=len(repeated), patients=len({k[0] for k in repeated}),
                admissions=len(set().union(*repeated.values())) if repeated else 0)


def derive():
    original_report = json.loads((INVENTORY/'inventory.json').read_text())
    for tag in ('a','b'):
        proof = json.loads((INVENTORY/f'verification_{tag}.json').read_text())
        if proof['maximum_error'] or proof['inventory_sha256'] != sha256(INVENTORY/'inventory.json') or proof['records_sha256'] != sha256(INVENTORY/'records.jsonl.gz'):
            raise ValueError('original PFT inventory proof differs')
    for p, digest in original_report['source_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('original inventory source changed: '+p)
    sources, _ = checked_sources(); groups, quarantine, _ = selected_groups(SCREEN)
    units = [u for k, g in groups.items() if k[2] == 'train' for u in g if 'pulmonary_function' in u['topics']]
    needed = {u['note_id'] for u in units}
    raw = {r['note_id']: r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:
        docs = {r['note_id']: r for r in map(json.loads,f) if r['note_id'] in needed}
    status = Counter(); quantity = Counter(); flags_count = Counter(); all_records = []
    eligible_hadms = set(); eligible_patients = set(); metadata = Counter()
    for u in sorted(units,key=lambda r:(r['note_id'],r['start_char'])):
        original = raw[u['note_id']]; text = original['text']; quote = u['quote']
        if text[u['start_char']:u['end_char']] != quote or sha256_text(text) != u['original_sha256']:
            raise ValueError('PFT original source differs')
        sec = docs[u['note_id']]['section_spans'][u['section_ordinal']]
        if not 0 <= sec['start_char'] <= u['start_char'] < u['end_char'] <= sec['end_char'] <= len(text):
            raise ValueError('PFT section membership differs')
        facts = bindings(quote)
        if not facts:
            metadata['topic_without_target_metric_name'] += 1; continue
        for fact in facts:
            status[fact['status']] += 1
            for q in fact['quantities']:
                if float(quote[q['value_start']:q['value_end']]) != q['value']:
                    raise ValueError('numeric value source span differs')
                if sha256_text(quote[q['evidence_start']:q['evidence_end']]) != q['evidence_sha256']:
                    raise ValueError('quantity evidence source span differs')
                quantity[(fact['metric']+'/'+q['kind'])] += 1
            if fact.get('scalar_binding_eligible',False):
                eligible_hadms.add(u['hadm_id']); eligible_patients.add(u['subject_id'])
        section = text[sec['start_char']:sec['end_char']]
        flags = dict(historical_section=u['section']=='past_medical_history', history_words=bool(HISTORY.search(quote)),
                     plan_words=bool(PLAN.search(quote)), pre_bronchodilator_words=bool(PRE.search(quote)),
                     post_bronchodilator_words=bool(POST.search(quote)), same_unit_date_hint=bool(DATE.search(quote)),
                     same_unit_year_hint=bool(YEAR.search(quote)), same_unit_redaction=bool(REDACTED.search(quote)),
                     same_section_date_hint=bool(DATE.search(section)), same_section_redaction=bool(REDACTED.search(section)))
        flags_count.update({k:int(v) for k,v in flags.items()})
        signature = [(f['metric'],f['status'],[(q['value'],q['kind'],q['qualifier']) for q in f['quantities']]) for f in facts if f['quantities']]
        row = dict(note_id=u['note_id'],hadm_id=u['hadm_id'],subject_id=u['subject_id'],section=u['section'],
                   start_char=u['start_char'],end_char=u['end_char'],original_sha256=u['original_sha256'],
                   quote_sha256=sha256_text(quote),quantities_signature=sha256_text(json.dumps(signature,sort_keys=True)),
                   section_sha256=sha256_text(section),flags=flags,bindings=facts,measurement_time='unknown',
                   charttime=original['charttime'],storetime=original['storetime'])
        all_records.append(row)
    numeric = [r for r in all_records if any(f['quantities'] for f in r['bindings'])]
    sources.update({str(p):sha256(p) for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_evidence.py'),
                    INVENTORY/'inventory.json',INVENTORY/'records.jsonl.gz',INVENTORY/'verification_a.json',INVENTORY/'verification_b.json')})
    report = dict(status='training_evidence_binding_audit_not_clinical_gold',units=len(units),
                  units_with_target_metric=len(all_records),units_with_numeric_candidate=len(numeric),
                  binding_status=dict(status),metric_quantity_counts=dict(quantity),time_role_flags=dict(flags_count),
                  extra_counts=dict(metadata),explicit_scalar_binding_admissions=len(eligible_hadms),
                  explicit_scalar_binding_patients=len(eligible_patients),
                  same_patient_exact_quote_repeat=repeat_stats(numeric,'quote_sha256'),
                  same_patient_numeric_signature_repeat=repeat_stats(numeric,'quantities_signature'),
                  source_sha256=sources,quarantine=quarantine,new_model_fits=0,new_vectors=0,
                  high_frequency_used=False,outcomes_used=False,nontraining_facts_built=False,
                  limits=['explicit quantity binding not valid measurement gold', 'all PFT measurement times unknown',
                          'same section date hints are not assigned to measurements', 'repeats not proof of copy or new measurement',
                          'r4 topic coverage not exhaustive original text', 'FINAL retrospective availability unchanged'])
    return report,all_records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('audit exists; inspect rather than overwrite')
        ROOT.mkdir();write_json(ROOT/'audit.json',report)
        with gzip.open(ROOT/'records.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_two_independent_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(line) for line in f]
        if saved!=report or prior!=records:raise ValueError('independent PFT evidence replay differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,records=len(records),
                   audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_source_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','quarantine')}))


if __name__=='__main__':main()
