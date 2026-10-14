from dataclasses import replace
import unittest
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.model import HistoryBaseline
from training.v6.clinical_facts_context import FactsBatch
from training.v6.radiology_experiment import RadiologyAnchoredModel, WIDTHS
from training.v6.risk import competing_nll
from patient_data.clinical_facts.short_topic_audit import extra_guard


class RadiologyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def sample(self):
        values = tuple(torch.zeros(3, w) for w in WIDTHS)
        values[0][:, 0] = .3  # 冻结编码连续向量不是二元标签。
        return FactsBatch(compact(), values, torch.tensor([[True, False, False]] * 3))

    def test_zero_delta_and_frozen_history(self):
        m = RadiologyAnchoredModel(HistoryBaseline(), 8, width=16)
        b = self.sample(); h = torch.zeros(3, 6)
        torch.testing.assert_close(m(h, b).log_amplitude, m.baseline(h).log_amplitude, rtol=0, atol=0)
        torch.nn.init.normal_(m.delta_head[-1].weight, std=.1)
        competing_nll(m(h, b), torch.tensor([3., 50., 365.]), torch.tensor([1, 2, 0])).backward()
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))
        self.assertGreater(m.encoder.fact_branches[0][0].weight.grad.abs().sum(), 0)

    def test_empty_branch_no_content_effect(self):
        m = RadiologyAnchoredModel(HistoryBaseline(), 8, width=16).eval()
        b = self.sample(); a, branches = m.encoder(b)
        changed = replace(b, values=tuple(v.clone() for v in b.values)); changed.values[1].normal_()
        other, _ = m.encoder(changed)
        torch.testing.assert_close(a, other, rtol=0, atol=0)
        self.assertEqual(branches['ct'].abs().sum(), 0)

    def test_dyspnea_not_hypoxemia(self):
        f = dict(field='exertional_desaturation', value='present', quote='dyspnea on exertion', section_role='hospital_course')
        self.assertEqual(extra_guard(f, {}), 'dyspnea_or_exercise_limitation_not_desaturation')

    def test_future_course_not_completed(self):
        f = dict(field='steroid_discharge_course', value='completed', quote='Your last day of prednisone is tomorrow', section_role='discharge_instructions')
        self.assertEqual(extra_guard(f, {}), 'future_completion_not_already_completed')

    def test_night_hypoxia_not_exertional_test(self):
        f = dict(field='exertional_desaturation', value='absent', quote='hypoxic at night without CPAP', section_role='hospital_course')
        self.assertEqual(extra_guard(f, {}), 'oxygenation_without_exertional_context')

    def test_exertional_measured_normality_can_be_kept(self):
        f = dict(field='exertional_desaturation', value='absent', quote='satted at 95% on ambulation', section_role='hospital_course')
        self.assertIsNone(extra_guard(f, {}))


if __name__ == '__main__': unittest.main()
