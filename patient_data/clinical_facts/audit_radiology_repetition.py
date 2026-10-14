"""多胸片复查审计；只输出匿名统计，不以报告次数推断病情恶化。"""
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import re
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.radiology_compare import records


def main():
    selected = Path('patient_data/processed/v6_imaging_all_2026-10-10_r2')
    snapshot = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
    data, members, manifest = records(selected)
    grouped = defaultdict(list)
    for r in data:
        if r['modality'] == 'cxr': grouped[r['hadm_id']].append(r)
    admissions = {r['hadm_id']: r for r in rows(snapshot / 'cohort/admissions.csv.gz')}
    top = sorted(grouped.values(), key=len, reverse=True)[:10]
    needed = {r['note_id'] for reports in top for r in reports}
    originals = {r['note_id']: r['text'] for r in rows(snapshot / 'raw/radiology.csv.gz') if r['note_id'] in needed}
    result = []
    for reports in top:
        dates = Counter(r['charttime'][:10] for r in reports)
        texts = [originals[r['note_id']] for r in reports]
        indications = []
        for text in texts:
            match = re.search(r'(?:INDICATION|REASON FOR (?:EXAM|EXAMINATION))\s*:\s*(.*?)(?=\n\s*(?:TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)\s*:|\Z)', text, re.I | re.S)
            indications.append(match.group(1) if match else '')
        admission = admissions[reports[0]['hadm_id']]
        los = (datetime.fromisoformat(admission['dischtime']) - datetime.fromisoformat(admission['admittime'])).total_seconds() / 86400
        result.append(dict(reports=len(reports), length_of_stay_days=round(los, 1),
                           distinct_charttimes=len({r['charttime'] for r in reports}), report_dates=len(dates), maximum_reports_per_day=max(dates.values()),
                           exact_full_text_duplicates=len(texts) - len(set(texts)),
                           indications_mentioning_device_check=sum(bool(re.search(r'line|tube|intubat|extubat|catheter', s, re.I)) for s in indications),
                           reports_mentioning_device_terms=sum(bool(re.search(r'endotracheal|chest tube|central venous|\bPICC\b|catheter', s, re.I)) for s in texts)))
    counts = Counter(len(v) for v in grouped.values())
    value = dict(status='source_checked_anonymous_report_statistics', report_counts=dict(sorted(counts.items())),
                 covered_admissions=len(grouped), multiple_report_admissions=sum(v for k, v in counts.items() if k > 1), top_ten=result,
                 selected_manifest_sha256=sha256(selected / 'manifest.json'),
                 limitations=['Device-term matches are mentions, not verified placement or causation', 'Distinct full texts can still contain redundant clinical content', 'Higher report frequency is not equivalent to clinical worsening'])
    output = Path('patient_data/processed/v6_radiology_repetition_audit_2026-10-10.json')
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
