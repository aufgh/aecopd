"""核验固定批次原文出处和严格结局支持量；不训练、不改变成员。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(batch: Path, snapshot: Path, labels: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads((batch/'selection.json').read_text(encoding='utf-8'))
    if sha(batch/selection['members_file']) != selection['member_sha256']:
        raise ValueError('Frozen member manifest changed')
    members = {str(r['hadm_id']): r for r in selection['members']}
    if len(members) != selection['selected_notes']:
        raise ValueError('Duplicate member admission')
    raw_manifest = json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    if (raw_manifest['status'] != 'validated' or
            sha(snapshot/'raw/discharge.csv.gz') != raw_manifest['sources']['raw/discharge']['sha256']):
        raise ValueError('Raw discharge provenance changed')
    original = {r['hadm_id']: r for r in rows(snapshot/'raw/discharge.csv.gz') if r['hadm_id'] in members}
    if set(original) != set(members):
        raise ValueError('Member discharge source missing')
    encoded = rows(batch/'encoding/shared_text.csv.gz')
    summary = json.loads((batch/'encoding/manifest_summary.json').read_text(encoding='utf-8'))
    if sha(batch/'encoding/shared_text.csv.gz') != summary['manifest_sha256']:
        raise ValueError('Encoded manifest changed')
    grouped: dict[str, list[dict]] = defaultdict(list)
    for i, record in enumerate(encoded):
        source = original.get(record['hadm_id'])
        member = members.get(record['hadm_id'])
        if source is None or member is None:
            raise ValueError('Encoded admission outside fixed members')
        source_sha = hashlib.sha256(source['text'].encode()).hexdigest()
        if (source['subject_id'] != record['subject_id'] or source['note_id'] != record['note_id'] or
                record['patient_split'] != member['patient_split'] or source_sha != record['source_sha256'] or
                source_sha != member['source_sha256'] or int(record['embedding_row']) != i):
            raise ValueError('Encoded source identity/order mismatch')
        a, b = int(record['start_char']), int(record['end_char'])
        text = record['chunk_text']
        if not (0 <= a < b <= len(source['text'])) or not text.endswith(source['text'][a:b]):
            raise ValueError('Encoded source span mismatch')
        if hashlib.sha256(text.encode()).hexdigest() != record['text_sha256']:
            raise ValueError('Encoded text hash mismatch')
        grouped[record['hadm_id']].append(record)
    if len(encoded) != summary['rows']:
        raise ValueError('Encoded row count mismatch')
    subject_splits = {}
    for member in members.values():
        subject, split = str(member['subject_id']), member['patient_split']
        if subject in subject_splits and subject_splits[subject] != split:
            raise ValueError('Patient crosses splits')
        subject_splits[subject] = split
    qa = json.loads((labels.parent/'summary.json').read_text(encoding='utf-8'))
    if qa['status'] != 'strict_labels_reconstructed_and_checked' or sha(labels) != qa['label_sha256']:
        raise ValueError('Strict endpoint labels are not validated')
    counts, note_counts = defaultdict(Counter), defaultdict(Counter)
    for record in rows(labels):
        split = subject_splits.get(record['subject_id'])
        if split is None:
            continue
        counts[split][record['event_type']] += 1
        if record['hadm_id'] in members:
            note_counts[split][record['event_type']] += 1
    support = {}
    for split in ('train', 'validation', 'test'):
        support[split] = {'patients': sum(s == split for s in subject_splits.values()),
            'eligible_discharge_histories': sum(counts[split].values()),
            'strict_readmission_events': counts[split]['1'], 'competing_deaths': counts[split]['2'],
            'no_observed_event_in_contract': counts[split]['0'],
            'note_members': sum(note_counts[split].values()),
            'note_member_strict_events': note_counts[split]['1'],
            'enough_for_existing_auc_case_minimum_20': counts[split]['1'] >= 20}
    # 训练折四份固定哈希样本及四份最长文书，后者用于压力检查，不估计总体准确率。
    train = [h for h, r in members.items() if r['patient_split'] == 'train']
    ranked = sorted(train, key=lambda h: hashlib.sha256(('audit:'+h).encode()).hexdigest())[:4]
    longest = sorted(train, key=lambda h: (-len(original[h]['text']), h))[:4]
    selected = list(dict.fromkeys(ranked+longest))
    packet = []
    for hadm in selected:
        source = original[hadm]
        packet.append({'member': members[hadm], 'original': source['text'], 'encoding_rows': grouped[hadm]})
    supported = all(support[s]['strict_readmission_events'] >= 20 and
                    support[s]['eligible_discharge_histories']-support[s]['strict_readmission_events'] >= 20
                    for s in ('validation', 'test'))
    result = {'status': ('source_provenance_checked_evaluation_ready' if supported else
                         'source_provenance_checked_evaluation_blocked'),
        'source_note_count': len(original), 'encoding_rows_checked': len(encoded),
        'member_sha256': selection['member_sha256'], 'manifest_sha256': summary['manifest_sha256'],
        'label_sha256': qa['label_sha256'], 'primary_event': qa['primary_event'],
        'support': support, 'training_started': False,
        'evaluation_blocker': (None if supported else
                              'Insufficient validation/test events or controls for the existing minimum of 20; see support counts.'),
        'clinical_accuracy_validated': False, 'manual_review_training_notes': len(packet),
        'review_sampling': 'four fixed-hash training notes plus four longest; not an unbiased accuracy estimate',
        'member_selection_unchanged': True}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'training_review.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.batch, args.snapshot, args.labels, args.output), ensure_ascii=False))
