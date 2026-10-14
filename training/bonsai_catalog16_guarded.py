"""首轮错误的局部工程回归与匹配2/4并发测速；不宣称独立泛化验证。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.bonsai_catalog16 import PATTERNS, make_payload, execute
from training.benchmark_bonsai_sections import section_sources
from training.bonsai_clinical_evidence_pilot import read_jsonl, local_api

ROOT = Path('patient_data/text_semantic_review_2026-10-03')
# 在首轮错误后固定的回归集，不能伪装成独立新验证。
CASES = (
 ('R014', 'discharge_residual_dyspnea'), ('R016', 'acute_respiratory_diagnosis'),
 ('R018', 'systemic_steroid_taper'), ('R018', 'respiratory_antibiotic_course'),
 ('R019', 'inhaler_technique_adherence'), ('R020', 'discharge_residual_dyspnea'),
 ('R020', 'airflow_wheezing_distress'), ('R020', 'oxygen_trajectory_discharge'),
 ('R014', 'respiratory_antibiotic_course'), ('R020', 'inhaled_medication_changes'),
)


def guarded_sources(text: str, key: str) -> tuple[list[dict], dict]:
    result, rejected, seen = [], [], set()
    for source in section_sources(text):
        hard = r'(?im)^[ \t]*(?:\d+[.)](?=\s)|(?:VITALS|GENERAL|GEN|HEENT|CARDIAC|LUNGS|PULMONARY|RESPIRATORY|NECK/BACK|NECK|BACK|ABDOMEN|ABD|EXTREMITIES|EXT|SKIN|NEUROLOGIC|NEURO):)'
        starts = sorted({0, len(source['text']), *(m.start() for m in re.finditer(hard, source['text']))})
        for left, right in zip(starts, starts[1:]):
            region = source['text'][left:right]
            for match in re.finditer(r'\S[\s\S]*?(?:[.!?](?=\s+(?:[A-Z#]|\d+[.)](?:\s|$)))|(?=\n[ \t\r]*\n)|$)', region):
                raw = match.group()
                quote = raw.strip()
                relevant = PATTERNS[key].search(quote)
                if key == 'discharge_residual_dyspnea':
                    relevant = relevant or re.search(r'catch(?:ing)? breath|breathless', quote, re.I)
                if not relevant:
                    continue
                begin = left+match.start()+len(raw)-len(raw.lstrip())
                end = begin+len(quote)
                before = source['text'][:begin]
                context = source['text'][max(0, begin-400):min(len(source['text']), end+400)]
                role = source['role']
                # 通用Physical Exam可含入院和出院子块，使用明确的局部标题而非猜测。
                markers = list(re.finditer(r'(?im)^[ \t]*(Admission|Discharge)(?:[ \t]+Physical[ \t]+Exam)?[ \t:]*\r?$', before))
                time_role = markers[-1].group(1).lower() if markers else None
                reason = None
                if key in {'discharge_residual_dyspnea', 'cough_sputum_burden', 'airflow_wheezing_distress'}:
                    explicit_late = bool(re.search(r'\b(?:at|on|prior to|by) discharge\b|discharged', quote, re.I))
                    if re.search(r'\bPRN\b|\bSig:|as needed|if you|call.*(?:doctor|911)', quote, re.I):
                        reason = 'PRN_or_hypothetical_not_observed_status'
                    elif role in {'history_of_present_illness', 'chief_complaint', 'admission_physical_exam', 'admission_medications', 'discharge_medications'} and not explicit_late:
                        reason = 'explicit_non_discharge_source'
                    elif time_role == 'admission' and not explicit_late:
                        reason = 'explicit_admission_subsection'
                    elif not (explicit_late or role in {'discharge_physical_exam', 'discharge_condition'} or time_role == 'discharge'
                              or re.search(r'improv|resolved|recover|wean|transitioned', quote, re.I)):
                        reason = 'ambiguous_event_time_requires_review'
                    if key == 'airflow_wheezing_distress' and re.search(r'clear to auscultation|\bCTAB\b|no wheezes|no w/r/r|no increased work', quote, re.I) and not re.search(r'improv|resolved', quote, re.I):
                        reason = reason or 'routine_normal_exam_not_recovery'
                if key == 'oxygen_trajectory_discharge' and not re.search(r'oxygen|\bO2\b|wean|breathing|without supplemental|home.*NC', quote, re.I):
                    reason = reason or 'isolated_RA_vitals_not_oxygen_trajectory'
                if key == 'inhaler_technique_adherence':
                    actual = re.search(r'incorrect|unable|difficulty|poor|non.?adher|non.?compl|did not|miss.*dose|refus|not.*prescribed|not.*tak|technique|demonstrat|teach.?back', quote, re.I)
                    target = re.search(r'inhal|spacer|\bMDI\b|\bDPI\b|albuterol|advair|azith|levoflox|respiratory.*(?:drug|med)|antibiotic', quote+' '+context, re.I)
                    if not actual or not target or re.search(r'please|remember to|it is important', quote, re.I):
                        reason = 'no_assessed_respiratory_adherence_or_technique'
                item = {'source_id': f'{key}:{len(result)+1}', 'role': role, 'start': source['start']+begin,
                        'end': source['start']+end, 'text': quote, 'context': context,
                        'problem_heading': source.get('problem_heading', ''), 'primary_section': True}
                if text[item['start']:item['end']] != quote:
                    raise ValueError('Candidate offset changed')
                if reason:
                    rejected.append({**item, 'reason': reason})
                elif quote not in seen:
                    result.append(item); seen.add(quote)
    return result, {'rejected_candidates': rejected, 'rejected_count': len(rejected),
                    'does_not_prove_absence': True, 'no_silent_long_unit_drop': True}


def run(output: Path, workers: int):
    if output.exists():
        raise FileExistsError(output)
    slots = local_api('http://127.0.0.1:8080', '/slots')
    if len(slots) != 4 or any(s['n_ctx'] != 8192 or s['is_processing'] for s in slots):
        raise ValueError('Need idle 4 x 8192 context slots')
    rows = {r['review_id']: r for r in read_jsonl(ROOT/'eligible_sources/source_review.jsonl')}
    jobs = []
    for rid, key in CASES:
        sources, audit = guarded_sources(rows[rid]['text'], key)
        payload = make_payload((key,), {key: sources})
        count = len(local_api('http://127.0.0.1:8080', '/tokenize', {'content': '\n'.join(m['content'] for m in payload['messages'])})['tokens'])
        if count+3200+512 > 8192:
            raise ValueError('No silent context truncation')
        jobs.append({'review_id': rid, 'task': key, 'keys': (key,), 'sources': {key: sources},
                     'payload': payload, 'retrieval_audit': audit, 'input_tokens_estimate': count,
                     'source_sha256': hashlib.sha256(rows[rid]['text'].encode()).hexdigest()})
    output.mkdir(parents=True)
    protocol = {'workers': workers, 'cases': CASES, 'slots': slots, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'post-error development regression only, not new independent validation',
                'same_official_medium_sampling_and_1024_thinking_budget': True, 'no_encoding_training': True}
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    began, samples, results, stop = time.monotonic(), [], [], threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                samples.append({'elapsed': time.monotonic()-began, 'gpu': [int(x) for x in gpu.stdout.strip().split(',')],
                                'active_slots': sum(s['is_processing'] for s in local_api('http://127.0.0.1:8080', '/slots'))})
            except Exception:
                pass
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, job, rows[job['review_id']]['text']): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                result = future.result()
                # 被时间规则保留为待核验的内容不冒充真正未提及。
                key = result['keys'][0]
                if not result['sources'][key] and result['retrieval_audit']['rejected_count']:
                    result['field_status'][key] = 'FILTERED_CANDIDATES_NOT_PROVEN_ABSENT'
                results.append(result)
                (output/f'{futures[future]:02}_{result["review_id"]}_{key}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'done': len(results), 'total': len(jobs), 'review_id': result['review_id'], 'field': key, 'seconds': result['seconds'], 'error': result.get('error')}, ensure_ascii=False), flush=True)
    finally:
        stop.set(); thread.join(timeout=5)
    elapsed = time.monotonic()-began
    stats = {'seconds': round(elapsed, 3), 'workers': workers, 'jobs': len(jobs), 'api_calls': sum(not r.get('no_api_call') for r in results),
             'failed_jobs': sum('error' in r for r in results), 'validation_issues': sum(len(r.get('validation_issues', [])) for r in results),
             'max_active_slots': max(s['active_slots'] for s in samples), 'gpu_peak_mib': max(s['gpu'][1] for s in samples),
             'gpu_mean_util': sum(s['gpu'][0] for s in samples)/len(samples)}
    (output/'gpu_samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    (output/'speed.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(2, 4), required=True)
    args = parser.parse_args(); run(args.output, args.workers)
