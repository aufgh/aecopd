"""r3审计适配器：不改r2文件，复用父引用校验，仅替换限定条目reader。"""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
from patient_data import discharge_regimen_entry_audit as audit
from patient_data.discharge_regimen_entries_checked import read_unit
from patient_data.export_mimiciv import sha256

R2=audit.ROOT
ROOT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r3')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_ENTRY_R3_PROTOCOL_2026-10-14.md')
original_derive=audit.derive


def derive():
    prior=json.loads((R2/'audit.json').read_text())
    proof=json.loads((R2/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(R2/'audit.json'):
        raise ValueError('r2 proof stale')
    for file,h in prior['output_sha256'].items():
        if sha256(R2/file)!=h:raise ValueError('r2 evidence changed')
    for file,h in prior['source_sha256'].items():
        if sha256(Path(file))!=h:raise ValueError('r2 captured source changed')
    report,entries,samples=original_derive()
    with gzip.open(R2/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as stream:
        old={r['entry_key']:r for r in map(json.loads,stream)}
    differences=Counter();actions=Counter()
    if set(old)!={r['entry_key'] for r in entries}:raise ValueError('candidate membership changed')
    for record in entries:
        previous=old[record['entry_key']]
        if any(record[k]!=previous[k] for k in ('quote','source_start','source_end','original_sha256')):
            raise ValueError('original quote/offset changed')
        reading=record['reading'];actions[str(reading['prefix_action'])]+=1
        for k in ('status','classes','canonical_identity'):
            differences[k]+=reading[k]!=previous['reading'][k]
    report.update(rule_version='r3_canonical_form_combo_identity_original_action',related_tests=21,
        differences_from_verified_r2=dict(differences),prefix_actions=dict(actions))
    for p in (Path(__file__),Path('patient_data/discharge_regimen_entries_checked.py'),
              Path('training/tests/test_discharge_regimen_entries_checked.py'),
              Path('patient_data/v6_expanded_medication_fix.py'),R2/'audit.json',R2/'verification.json',
              R2/'entry_evidence.jsonl.gz'):
        report['source_sha256'][str(p)]=sha256(p)
    return report,entries,samples


# 显式仅进程内适配；所有受保护r2源文件保持不变。
audit.ROOT=ROOT;audit.PROTOCOL=PROTOCOL;audit.read_unit=read_unit;audit.derive=derive


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'))
    mode=parser.parse_args().mode
    try:audit.run() if mode=='run' else audit.verify()
    except Exception as error:
        if ROOT.exists():audit.write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
