"""原文与严格结局审核的边界回归，不需要GPU。"""
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.audit_broad_fragment_batch import audit, sha


class BatchAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.batch, self.snapshot = self.root/'batch', self.root/'snapshot'
        self.labels = self.root/'labels/labels.csv.gz'
        text = 'Admission symptom.\nAt discharge needs oxygen.'
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        member = dict(subject_id=1, hadm_id=10, note_id='n', patient_split='train', source_sha256=source_hash)
        self.csv(self.batch/'members.csv', [member])
        self.json(self.batch/'selection.json', dict(members=[member], selected_notes=1,
                  members_file='members.csv', member_sha256=sha(self.batch/'members.csv')))
        self.csv(self.snapshot/'raw/discharge.csv.gz', [dict(subject_id='1', hadm_id='10', note_id='n', text=text)])
        self.json(self.snapshot/'metadata/manifest.json', dict(status='validated', sources={
                  'raw/discharge': {'sha256': sha(self.snapshot/'raw/discharge.csv.gz')}}))
        chunk = 'Clinical excerpt. Source section: hospital course. '+text
        self.record = dict(embedding_row=0, subject_id='1', hadm_id='10', note_id='n',
                           patient_split='train', source_sha256=source_hash, start_char=0,
                           end_char=len(text), chunk_text=chunk,
                           text_sha256=hashlib.sha256(chunk.encode()).hexdigest())
        self.manifest()
        self.csv(self.labels, [dict(subject_id='1', hadm_id='10', event_type='1')])
        self.json(self.labels.parent/'summary.json', dict(status='strict_labels_reconstructed_and_checked',
                  label_sha256=sha(self.labels), primary_event='strict test endpoint'))

    @staticmethod
    def json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def csv(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'wt', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(value[0]))
            writer.writeheader()
            writer.writerows(value)

    def manifest(self):
        path = self.batch/'encoding/shared_text.csv.gz'
        self.csv(path, [self.record])
        self.json(self.batch/'encoding/manifest_summary.json', dict(rows=1, manifest_sha256=sha(path)))

    def run_audit(self):
        return audit(self.batch, self.snapshot, self.labels, self.root/'audit')

    def test_multiline_source_and_event_support(self):
        result = self.run_audit()
        self.assertEqual(result['encoding_rows_checked'], 1)
        self.assertEqual(result['support']['train']['strict_readmission_events'], 1)
        self.assertFalse(result['support']['validation']['enough_for_existing_auc_case_minimum_20'])
        self.assertFalse(result['training_started'])

    def test_wrong_patient(self):
        self.record['subject_id'] = '2'
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.run_audit()

    def test_invalid_reverse_span(self):
        self.record['start_char'], self.record['end_char'] = 8, 3
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'span'):
            self.run_audit()

    def test_changed_quote(self):
        self.record['chunk_text'] = 'Invented text.'
        self.record['text_sha256'] = hashlib.sha256(b'Invented text.').hexdigest()
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'span'):
            self.run_audit()

    def test_member_manifest_change(self):
        (self.batch/'members.csv').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'member'):
            self.run_audit()


if __name__ == '__main__':
    unittest.main()
