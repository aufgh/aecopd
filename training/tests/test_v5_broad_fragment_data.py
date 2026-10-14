"""宽主题接入的来源、患者历史和共有初始化回归。"""
import csv
import gzip
import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from training.v5.broad_fragment_data import (
    attach_broad_text, checked_selection, common_roles, old_or_no_text, preserve_selected_histories, sha,
)
from training.v5.event_data import EventData
from training.v5.independent_data import IndependentData
from training.v5.run_broad_fragment_comparison import initial_shared_hash


class BroadFragmentDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        base = IndependentData(
            np.array([1, 1, 2, 2, 3, 3]), np.arange(10, 16),
            np.array(['train', 'train', 'validation', 'validation', 'test', 'test']),
            np.full(6, 365., np.float32), np.zeros(6, np.int64), np.zeros((6, 2), np.float32),
            (np.array([0, 1]), np.array([2, 3]), np.array([4, 5])),
            np.zeros((6, 5), np.float32), np.ones((6, 5), np.float32),
            np.ones(6, np.int64), np.ones(6, np.int64), tuple(np.array([i]) for i in range(6)),
            ('context:age',), ('context',), tuple(((0, i),) for i in range(6)),
            np.zeros((6, 768), np.float16), 768, 'old_manifest', 'labels', 'strict',
            field_roles=('history_of_present_illness', 'past_medical_history', 'hospital_course', 'discharge_condition'))
        self.data = EventData(base, np.empty((0, 4), np.float32), np.empty(0, np.int64),
            np.empty(0, bool), tuple(np.empty(0, np.int64) for _ in range(6)), {},
            recovery_by_admission=np.zeros((6, 6), np.float32),
            medication_by_admission=np.zeros((6, 48), np.float32))
        self.members = [dict(subject_id=1, hadm_id=10, note_id='n1', patient_split='train', source_sha256='s1'),
                        dict(subject_id=2, hadm_id=12, note_id='n2', patient_split='validation', source_sha256='s2')]
        self.selection = dict(members=self.members, selected_notes=2, members_file='members.csv')
        self.csv(self.root/'members.csv', self.members)
        self.selection['member_sha256'] = sha(self.root/'members.csv')
        self.json(self.root/'selection.json', self.selection)
        self.rows = []
        for i, member in enumerate(self.members):
            text = f'Original clinical sentence {i}.'
            self.rows.append(dict(member, embedding_row=i, start_char=0, end_char=len(text),
                             chunk_text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             field='respiratory_course|respiratory_support'))
        self.manifest()

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
        directory = self.root/'encoding'
        self.csv(directory/'shared_text.csv.gz', self.rows)
        digest = sha(directory/'shared_text.csv.gz')
        self.json(directory/'manifest_summary.json', dict(status='manifest_ready', rows=2,
                  provenance_validated=True, manifest_sha256=digest))
        self.json(directory/'vector_validation.json', dict(manifest_sha256=digest,
                  encoders={k: {'status': 'validated'} for k in ('bert', 'qwen')}))
        for name, width, filename in (('bert', 768, 'bioclinicalbert.float16.npy'),
                                      ('qwen', 2560, 'qwen3_embedding.float16.npy')):
            self.json(directory/name/'state.json', dict(status='complete', manifest_sha256=digest,
                completed_rows=2, model='Bio_ClinicalBERT', encoder='Qwen3-Embedding-4B'))
            vectors = np.zeros((2, width), np.float16)
            vectors[:, 0] = 1
            np.save(directory/name/filename, vectors)

    def test_frozen_membership_checked(self):
        result = checked_selection(self.data, self.root/'selection.json')
        self.assertEqual(result['selected_notes'], 2)
        (self.root/'members.csv').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'membership'):
            checked_selection(self.data, self.root/'selection.json')

    def test_wrong_patient_or_split_rejected(self):
        base = replace(self.data.base, subject_ids=np.array([99, 1, 2, 2, 3, 3]))
        with self.assertRaisesRegex(ValueError, 'patient/admission/split'):
            checked_selection(replace(self.data, base=base), self.root/'selection.json')

    def test_all_selected_histories_including_no_note_retained(self):
        data = preserve_selected_histories(self.data, self.selection)
        self.assertEqual([s.tolist() for s in data.base.sequences], [[0, 1], [2, 3]])
        self.assertEqual(len(data.base.hadm_ids), 6)  # 原始索引保持，不误重排其它张量。

    def test_truncated_source_history_rejected(self):
        base = replace(self.data.base, sequences=(np.array([0]), np.array([2, 3]), np.array([4, 5])))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            preserve_selected_histories(replace(self.data, base=base), self.selection)

    def test_old_and_no_text_use_same_members_and_roles(self):
        old = old_or_no_text(self.data, self.selection, enabled=True)
        none = old_or_no_text(self.data, self.selection, enabled=False)
        self.assertTrue(old.base.text_by_admission[0])
        self.assertEqual(old.base.text_by_admission[1], ())
        self.assertFalse(any(none.base.text_by_admission))
        self.assertEqual(old.base.field_roles, none.base.field_roles)

    def test_new_refs_replace_not_append_and_multitopic_not_duplicated(self):
        result = attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')
        self.assertEqual(result.base.text_by_admission[0], ((4, 0),))
        self.assertEqual(result.base.text_by_admission[1], ())
        self.assertEqual(sum(map(len, result.base.text_by_admission)), 2)
        self.assertEqual(result.base.field_roles, common_roles(self.data))
        self.assertIs(result.base.token_values, self.data.base.token_values)
        self.assertIs(result.recovery_by_admission, self.data.recovery_by_admission)

    def test_both_encoders_same_refs_and_hash(self):
        bert = attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')
        qwen = attach_broad_text(self.data, self.selection, self.root/'encoding', 'qwen')
        self.assertEqual(bert.base.text_by_admission, qwen.base.text_by_admission)
        self.assertEqual(bert.base.manifest_sha256, qwen.base.manifest_sha256)
        self.assertEqual((bert.base.text_width, qwen.base.text_width), (768, 2560))

    def test_incomplete_encoder_rejected(self):
        path = self.root/'encoding/bert/state.json'
        state = json.loads(path.read_text())
        state['status'] = 'running'
        self.json(path, state)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')

    def test_wrong_embedding_row_or_subject_rejected(self):
        self.rows[0]['embedding_row'] = 9
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'identity/order/content'):
            attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')

    def test_unnormalized_vectors_rejected(self):
        np.save(self.root/'encoding/bert/bioclinicalbert.float16.npy', np.zeros((2, 768), np.float16))
        with self.assertRaisesRegex(ValueError, 'unnormalized'):
            attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')

    def test_shared_initial_weights_do_not_depend_on_text_width(self):
        bert = attach_broad_text(self.data, self.selection, self.root/'encoding', 'bert')
        qwen = attach_broad_text(self.data, self.selection, self.root/'encoding', 'qwen')
        b, bp = initial_shared_hash(bert, 2026, torch.device('cpu'))
        q, qp = initial_shared_hash(qwen, 2026, torch.device('cpu'))
        self.assertEqual(b, q)
        self.assertGreater(qp, bp)


if __name__ == '__main__':
    unittest.main()
