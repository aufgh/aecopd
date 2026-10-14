"""合成易错病例：相同目录分别测 medium / none；不是临床准确率评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bonsai_clinical_evidence_pilot import extract_one


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    cases = json.loads(Path(__file__).with_name('clinical_evidence_regression_cases.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    summary = []
    for mode in ('medium', 'none'):
        for case in cases:
            print(f"START {case['review_id']} {mode}", flush=True)
            result = extract_one(case, 'http://127.0.0.1:8080', mode)
            checks = []
            evidence = result.get('evidence', {})
            for field in case['expected_empty']:
                checks.append({'field': field, 'rule': 'empty', 'pass': field in evidence and not evidence[field]})
            for field in case['expected_present']:
                checks.append({'field': field, 'rule': 'present', 'pass': bool(evidence.get(field))})
            for check in case['time_checks']:
                entries = [x for x in evidence.get(check['field'], []) if check['contains'] in x['quote']]
                checks.append({'field': check['field'], 'contains': check['contains'],
                               'rule': 'retain_original_timing_words', 'pass': bool(entries)})
            result['regression_checks'] = checks
            result['regression_pass'] = result['status'] == 'source_verified_only' and all(c['pass'] for c in checks)
            (output/f"{case['review_id']}_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            small = {k: result.get(k) for k in ('review_id', 'reasoning', 'status', 'seconds', 'reasoning_chars', 'regression_pass', 'regression_checks')}
            summary.append(small)
            print(f"END {case['review_id']} {mode} pass={result['regression_pass']} {result['seconds']}s", flush=True)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
