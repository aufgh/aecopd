"""训练患者原文抽样：不读取结局，不截断保存的原文，不生成训练输入。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from training.v5.section_alignment_data.section_parser import parse_sections

ENRICHMENT = {
    'ventilation': r'\b(?:bipap|intubat|hypercarb|hypercapn|respiratory acidosis)',
    'access_adherence': r'\b(?:noncompli|nonadher|could not afford|ran out|inhaler technique)',
    'functional_nutrition': r'\b(?:cachec|malnutri|weight loss|decondition|aspiration)',
    'differential': r'\b(?:pulmonary embol|pneumothorax|bronchiect|heart failure)',
    'home_support': r'\b(?:lives alone|homeless|caregiver|home oxygen)',
}

def audit(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    with gzip.open(source/'cohort/patient_splits.csv.gz', 'rt', encoding='utf-8-sig', newline='') as f:
        training = {r['subject_id'] for r in csv.DictReader(f) if r['patient_split'] == 'train'}
    with gzip.open(source/'cohort/admissions.csv.gz', 'rt', encoding='utf-8-sig', newline='') as f:
        eligible = {r['hadm_id'] for r in csv.DictReader(f) if r['discharge_landmark_eligible'] == 't'}
    rows = []
    headings: Counter = Counter()
    total = 0
    with gzip.open(source/'raw/discharge.csv.gz', 'rt', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            total += 1
            headings.update(s.role for s in parse_sections(row['text']))
            if row['subject_id'] in training and row['hadm_id'] in eligible:
                rows.append(row)
    ordered = sorted(rows, key=lambda r: hashlib.sha256(('clinical-audit-2026-10-03:'+r['note_id']).encode()).hexdigest())
    chosen = []
    seen = set()
    def take(candidates, count, stratum):
        added = 0
        for r in candidates:
            if r['subject_id'] in seen:
                continue
            chosen.append((r, stratum))
            seen.add(r['subject_id'])
            added += 1
            if added == count:
                break
    take(ordered, 40, 'hash_random')
    for key, pattern in ENRICHMENT.items():
        take((r for r in ordered if re.search(pattern, r['text'], re.I)), 6, key)
    output.mkdir(parents=True)
    review = []
    with (output/'source_review.jsonl').open('w', encoding='utf-8') as f:
        for index, (row, stratum) in enumerate(chosen, 1):
            item = dict(row, review_id=f'R{index:03}', stratum=stratum,
                        source_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
            review.append(f"## {item['review_id']} / {stratum}\n")
            # 仅供定位阅读；完整原文在 JSONL。节选不是临床真值或模型输入。
            for section in parse_sections(row['text']):
                if section.role in {'history_of_present_illness', 'past_medical_history', 'hospital_course',
                                    'discharge_physical_exam', 'discharge_condition', 'discharge_instructions',
                                    'social_history'} and section.content_status == 'informative':
                    lines = section.text.splitlines()
                    informative = [line for line in lines if re.search(
                        r'(?i)dysp|breath|oxygen|bipap|copd|wheez|sputum|cough|inhal|steroid|predni|antibio|'
                        r'infect|pneum|heart fail|edema|ambulat|assist|adher|afford|smok|cachex|malnutri|'
                        r'hypercap|hypercar|hypox|acido|tolerat|baseline|improv|resolv|persist|home|renal|anemi', line)]
                    block = '\n'.join(informative)[:2100] if informative else section.text[:500]
                    review.append(f"### {section.role} [{section.start_char}:{section.end_char}]\n{block}\n")
    (output/'reading_excerpt.md').write_text('\n'.join(review), encoding='utf-8')
    wanted = {r['hadm_id'] for r, _ in chosen}
    radiology = []
    with gzip.open(source/'raw/radiology.csv.gz', 'rt', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['subject_id'] in training and row['hadm_id'] in wanted:
                radiology.append(row)
    selected_rad = radiology[:20]
    with (output/'radiology_review.jsonl').open('w', encoding='utf-8') as f:
        for index, row in enumerate(selected_rad, 1):
            f.write(json.dumps(dict(row, review_id=f'I{index:03}', source_sha256=hashlib.sha256(row['text'].encode()).hexdigest()))+'\n')
    report = {'source': str(source.resolve()), 'discharge_total': total, 'training_discharge_total':len(rows),
              'review_documents':len(chosen), 'review_patients':len(seen),
              'strata':dict(Counter(s for _, s in chosen)), 'radiology_review_documents':len(selected_rad),
              'heading_occurrences':dict(headings), 'uses_outcomes':False,
              'clinical_truth_validated':False, 'sampling_not_prevalence_estimate':True,
              'discharge_landmark_eligibility_checked':True}
    (output/'source_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.source,args.output),ensure_ascii=False))
