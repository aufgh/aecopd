"""紧凑接口不改变旧输入；患者关联/哈希、缺失和二元语义严格检查。"""
from dataclasses import fields
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from patient_data.compact_multisource import MED_CHANNELS, MICRO_NAMES, SUPPORT_NAMES
from patient_data.export_mimiciv import sha256
from patient_data.v6_lab_state import AXES, MED_GROUPS
from training.v6.compact_context_data import attach, CompactContextBatch
from training.tests.test_v6_core import batch


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.original = batch()
        self.arrays = {'hadm_ids': np.arange(3), 'subject_ids': np.arange(3), 'patient_splits': np.array(['train', 'validation', 'test']),
                       'med_exposure': self.original.med_exposure.numpy()}
        self.base = SimpleNamespace(arrays=self.arrays, provenance={'state_manifest_sha256': 'parent'},
            collate=lambda positions, device: (torch.zeros(3, 6, device=device), self.original, torch.ones(3, device=device), torch.zeros(3, device=device)))
        med = np.zeros((3, 7, 4), np.float32); med[:, :, 0] = self.arrays['med_exposure']; med[:, :, 2] = 1
        self.compact = {k: v for k, v in self.arrays.items() if k != 'med_exposure'}
        self.compact.update(med_features=med, micro_values=np.zeros((3, 5), np.float32), micro_masks=np.zeros((3, 5), bool),
            support_values=np.zeros((3, 3), np.float32), support_masks=np.zeros((3, 3), bool),
            med_group_names=np.array(tuple(MED_GROUPS)), med_channel_names=np.array(MED_CHANNELS),
            micro_names=np.array(MICRO_NAMES), support_names=np.array(SUPPORT_NAMES), axis_names=np.array(AXES))

    def save(self, directory):
        np.savez(directory/'compact_context.npz', **self.compact)
        meta = {'status': 'compact_multisource_validated_not_trained', 'parent_state_manifest_sha256': 'parent',
                'output_sha256': {'compact_context.npz': sha256(directory/'compact_context.npz')}}
        (directory/'manifest.json').write_text(json.dumps(meta))

    def test_batch_keeps_labs_old_exposure_and_separate_order_channel(self):
        # 测试中挑出未有执行证据的类，医嘱不能偷偷加进旧med_exposure。
        self.compact['med_features'][0, 6] = [0, 1, 1, 1]
        self.arrays['med_exposure'][0, 6] = 0
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); self.save(p); data = attach(self.base, p)
            _, result, _, _ = data.collate(np.arange(3))
            self.assertEqual(result.med_features[0, 6, 1], 1)
            self.assertEqual(result.clinical.med_exposure[0, 6], 0)
            self.assertIs(result.clinical.lab_axis_inputs, self.original.lab_axis_inputs)
            self.assertFalse(result.micro_masks.any())
            self.assertFalse(result.support_masks.any())
            self.assertEqual([f.name for f in fields(result)], ['clinical', 'med_features', 'micro_values', 'micro_masks', 'support_values', 'support_masks'])
            self.assertIsNot(result.clinical, self.original)

    def test_member_order_checked(self):
        self.compact['hadm_ids'] = self.compact['hadm_ids'][::-1]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); self.save(p)
            with self.assertRaises(ValueError): attach(self.base, p)

    def test_invalid_positive_without_source(self):
        self.compact['micro_values'][0, 0] = 1
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); self.save(p)
            with self.assertRaises(ValueError): attach(self.base, p)

    def test_order_only_not_execution_and_not_both(self):
        self.compact['med_features'][0, 0] = [1, 1, 1, 1]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); self.save(p)
            with self.assertRaises(ValueError): attach(self.base, p)

    def test_no_new_time_or_counts_in_batch(self):
        names = [f.name for f in fields(CompactContextBatch)]
        self.assertFalse(any(word in name for name in names for word in ('time', 'count', 'duration', 'dose')))


if __name__ == '__main__': unittest.main()
