"""训练折肺功能数值词面库存，不是事实gold或可直接入网特征。"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.text_time_coverage import SCREEN, SNAP, checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v6.train import write_json

ROOT = Path('patient_data/processed/v6_pft_inventory_2026-10-13')
PROTOCOL = Path('docs/V6_PFT_INVENTORY_PROTOCOL_2026-10-13.md')
NAME = re.compile(r'\b(?:FEV\s*1\s*/\s*FVC|FEV\s*1|FVC|DLCO)\b', re.I)
VALUE = re.compile(r'^\s*(?:[:=]|(?:was|is|of)\b)?\s*([<>]=?\s*)?(\d+(?:\.\d+)?|\.\d+)(?![\d.])', re.I)
UNIT = re.compile(r'^\s*(%|m[lL]\b|[lL]\b|lit(?:er|re)s?\b)', re.I)
PREDICTED = re.compile(r'\b(?:predicted|pred\.?\b)', re.I)
DATE = re.compile(r'\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2})\b')
HISTORY = re.compile(r'\b(?:history|prior|previous\w*|baseline|last PFT|old)\b', re.I)
PLAN = re.compile(r'\b(?:plan\w*|recommend\w*|will|should|schedule\w*)\b', re.I)
POST = re.compile(r'\bpost[- ]?(?:bronchodilator|BD)\b', re.I)
PRE = re.compile(r'\bpre[- ]?(?:bronchodilator|BD)\b', re.I)


def candidates(text):
    """只取指标后首数值；保留未知单位，禁止跨下个指标借限定。"""
    names = list(NAME.finditer(text))
    result = []
    for i, match in enumerate(names):
        end = names[i + 1].start() if i + 1 < len(names) else len(text)
        tail = text[match.end():end]
        value = VALUE.match(tail)
        if not value:
            continue
        after = tail[value.end():]
        unit = UNIT.match(after)
        literal = unit.group(1) if unit else None
        metric = re.sub(r'\s+', '', match.group()).upper()
        ranged = bool(re.match(r'^\s*[-–]\s*\d', after))
        # 同指标50字符内线索，只标记，不证明预计值或测量有效性。
        predicted = bool(PREDICTED.search(after[:50]))
        role = ('ratio_percent' if metric == 'FEV1/FVC' else
                'percent_predicted_lexical' if predicted else 'percent_unspecified') if literal == '%' else (
                'volume_ml' if literal and literal.lower() == 'ml' else
                'volume_l' if literal else 'unit_unknown')
        result.append(dict(metric=metric, value=float(value.group(2)),
                           qualifier=(value.group(1) or '').strip(), unit_literal=literal,
                           quantity_role=role, range_ambiguous=ranged,
                           metric_start=match.start(), value_start=match.end()+value.start(2),
                           value_end=match.end()+value.end(2)))
    return result


def derive():
    sources, _ = checked_sources()
    groups, quarantine, manifest_hash = selected_groups(SCREEN)
    units = [u for key, group in groups.items() if key[2] == 'train'
             for u in group if 'pulmonary_function' in u['topics']]
    needed = {u['note_id'] for u in units}
    original = {r['note_id']: r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    stats = Counter(units=len(units)); section_counts = Counter(); metric_counts = Counter(); roles = Counter()
    mentions = set(); numeric_admissions = set(); numeric_patients = set(); all_notes = set(); records = []
    for u in sorted(units, key=lambda u: (u['note_id'], u['start_char'])):
        raw = original[u['note_id']]
        if raw['text'][u['start_char']:u['end_char']] != u['quote']:
            raise ValueError('original quote differs')
        if sha256_text(raw['text']) != u['original_sha256']:
            raise ValueError('original document hash differs')
        mentions.add(u['hadm_id']); all_notes.add(u['note_id']); section_counts[u['section']] += 1
        values = candidates(u['quote'])
        stats['units_with_numeric_candidate'] += bool(values)
        stats['units_without_adjacent_numeric_candidate'] += not bool(values)
        for v in values:
            metric_counts[v['metric']] += 1; roles[v['quantity_role']] += 1
            stats['range_ambiguous'] += v['range_ambiguous']
        if not values:
            continue
        numeric_admissions.add(u['hadm_id']); numeric_patients.add(u['subject_id'])
        flags = dict(historical_words=bool(HISTORY.search(u['quote'])), plan_words=bool(PLAN.search(u['quote'])),
                     pre_bronchodilator_words=bool(PRE.search(u['quote'])), post_bronchodilator_words=bool(POST.search(u['quote'])))
        stats.update({k: int(v) for k, v in flags.items()})
        dates = DATE.findall(u['quote']); stats['numeric_units_with_date_words'] += bool(dates)
        records.append(dict(note_id=u['note_id'], hadm_id=u['hadm_id'], subject_id=u['subject_id'],
                            section=u['section'], start_char=u['start_char'], end_char=u['end_char'],
                            quote_sha256=sha256_text(u['quote']), original_sha256=u['original_sha256'],
                            flags=flags, date_words=dates, candidates=values))
    sources[str(PROTOCOL)] = sha256(PROTOCOL); sources[str(Path(__file__))] = sha256(Path(__file__))
    sources['training/tests/test_v6_pft_inventory.py'] = sha256(Path('training/tests/test_v6_pft_inventory.py'))
    report = dict(status='training_lexical_inventory_not_validated_measurements', stats=dict(stats),
                  sections=dict(section_counts), metrics=dict(metric_counts), quantity_roles=dict(roles),
                  mentioned_admissions=len(mentions), mentioned_notes=len(all_notes),
                  numeric_candidate_admissions=len(numeric_admissions), numeric_candidate_patients=len(numeric_patients),
                  selected_manifest_sha256=manifest_hash, source_sha256=sources, future_quarantine=quarantine,
                  outcome_labels_used=False, new_model_fits=0, new_vectors=0, high_frequency_used=False,
                  limitations=['selected r4 candidates not exhaustive original text', 'date and role words are not temporal gold',
                               'adjacent number is not verified measurement; missing units stay unknown',
                               'FINAL retrospective notes not guaranteed real-time discharge availability'])
    return report, records


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('build', 'verify'), required=True)
    parser.add_argument('--tag', choices=('a', 'b'), default='a'); args = parser.parse_args()
    report, records = derive()
    if args.mode == 'build':
        if ROOT.exists():
            raise FileExistsError('inventory exists; inspect rather than overwrite')
        ROOT.mkdir(); write_json(ROOT/'inventory.json', report)
        with gzip.open(ROOT/'records.jsonl.gz', 'wt', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True)+'\n')
        write_json(ROOT/'state.json', dict(status='inventory_pending_independent_verification', new_model_fits=0))
    else:
        saved = json.loads((ROOT/'inventory.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz', 'rt', encoding='utf-8') as f:
            prior = [json.loads(line) for line in f]
        if saved != report or prior != records:
            raise ValueError('independent PFT inventory differs')
        write_json(ROOT/f'verification_{args.tag}.json', dict(maximum_error=0, records=len(records),
                   inventory_sha256=sha256(ROOT/'inventory.json'), records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a', 'b')):
            write_json(ROOT/'state.json', dict(status='inventory_two_independent_processes_verified', new_model_fits=0))
    # 只输出聚合，患者标识／日期原词／原文均留本地。
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'future_quarantine')}))


if __name__ == '__main__':
    main()
