"""异常与恢复特征不能把缺测、同期重复或无复查误判为康复。"""

import unittest

import torch

from patient_data.lab_recovery_features import abnormal_severity, summarize_axis
from training.tests.test_v5_independent_fusion import _batch
from training.v5.event_data import EventBatch
from training.v5.event_set_model import EventSetFusionModel


class LabRecoveryFeatureTests(unittest.TestCase):
    def test_low_arterial_ph_then_normal_has_recovery(self):
        first = abnormal_severity("arterial_low_ph", 7.25, 7.35, 7.45)
        last = abnormal_severity("arterial_low_ph", 7.38, 7.35, 7.45)
        result = summarize_axis([(0.0, first), (1.0, last)])
        self.assertEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], result[2])
        self.assertEqual(result[3], 0.0)
        self.assertEqual(result[4], 1.0)

    def test_one_sample_or_same_time_cannot_claim_recovery(self):
        self.assertEqual(summarize_axis([(0.0, 2.0)])[4], 0.0)
        self.assertEqual(summarize_axis([(0.0, 2.0), (0.0, 0.0)])[4], 0.0)

    def test_reference_range_is_directional(self):
        self.assertAlmostEqual(abnormal_severity("creatinine_high", 1.8, .6, 1.2), 1.0)
        self.assertEqual(abnormal_severity("creatinine_high", .9, .6, 1.2), 0.0)
        self.assertEqual(abnormal_severity("albumin_low", 4.0, 3.5, 5.0), 0.0)
        self.assertEqual(abnormal_severity("creatinine_high", 12., .6, 1.2), 5.0)
        self.assertGreater(abnormal_severity("creatinine_high", 12., .6, 1.2,
                                             severity_cap=None), 5.0)

    def test_missing_is_not_normal(self):
        self.assertEqual(summarize_axis([]), (0., 0., 0., 0., 0., 0.))

    def test_recovery_branch_changes_only_observed_admission(self):
        torch.manual_seed(17)
        model = EventSetFusionModel(concept_count=3, type_count=2, field_count=2,
                                    text_width=8, width=32, event_width=16,
                                    event_concept_count=32, clinical_recovery_width=54).eval()
        events = torch.tensor([[1., .1, 0., 0.], [2., .2, 0., 0.]])
        common = (_batch(text=False), events, torch.tensor([1, 1]), torch.tensor([[1, 1]]))
        mask = torch.zeros((1, 2, 54))
        active = mask.clone()
        active[0, 1, 0] = 1.
        active[0, 1, 2] = 1.
        with torch.no_grad():
            sham, _ = model._events(EventBatch(*common, event_is_lab=torch.tensor([True, True]),
                                                 event_numeric_mask=torch.tensor([True, True]),
                                                 recovery_by_admission=mask))
            changed, _ = model._events(EventBatch(*common, event_is_lab=torch.tensor([True, True]),
                                                    event_numeric_mask=torch.tensor([True, True]),
                                                    recovery_by_admission=active))
        self.assertTrue(torch.equal(sham[0], changed[0]))
        self.assertFalse(torch.equal(sham[1], changed[1]))


if __name__ == "__main__":
    unittest.main()
