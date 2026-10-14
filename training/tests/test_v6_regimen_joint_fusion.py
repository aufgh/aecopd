"""融合前显式上下文、共同起点和可训练白名单。"""
from dataclasses import replace
import unittest
import torch

from training.tests.test_v6_module_surgery import sample
from training.v6.model import HistoryBaseline
from training.v6.module_surgery import SurgeryModel
from training.v6.regimen_joint_fusion import JointFusion, RegimenBatch, ACTIVE_PREFIXES
from training.v6.risk import competing_nll


class JointFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(8)
        self.teacher = SurgeryModel(HistoryBaseline(), 42, 'control', [[-1., 1.]]*132, text_width=8).eval()
        torch.nn.init.normal_(self.teacher.delta_head[-1].weight, std=.03)
        self.h = torch.randn(3, 6)
        self.batch = RegimenBatch(sample(), torch.rand(3, 34), torch.tensor([True, False, True]))

    def model(self, arm='content'):
        return JointFusion(self.teacher, arm).eval()

    def test_exact_trained_teacher_initial_risk(self):
        expected = self.teacher(self.h, self.batch.original)
        for arm in ('availability', 'content'):
            m = self.model(arm); actual = m(self.h, self.batch)
            torch.testing.assert_close(actual.log_amplitude, expected.log_amplitude, rtol=0, atol=0)
            torch.testing.assert_close(actual.shape, expected.shape, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in m.regimen.parameters()), 4480)

    def test_explicit_batch_required(self):
        with self.assertRaises(TypeError):
            self.model()(self.h, self.batch.original)

    def test_trainable_white_list_and_h34_eval(self):
        m = self.model(); m.train()
        self.assertFalse(m.baseline.training)
        for n, p in m.teacher.named_parameters():
            self.assertEqual(p.requires_grad, n.startswith(ACTIVE_PREFIXES))
        self.assertFalse(any(p.requires_grad for p in m.teacher.encoder.extra_meds.parameters()))

    def test_missing_new_injection_zero_when_learned(self):
        m = self.model()
        with torch.no_grad():
            m.regimen.weight.fill_(.1)
        x = m.injection(self.batch.features, self.batch.available)
        self.assertEqual(torch.count_nonzero(x[1]), 0)
        self.assertGreater(float(x[0].detach().sum()), 0)

    def test_availability_content_hidden_after_learning(self):
        m = self.model('availability')
        with torch.no_grad():
            m.regimen.weight.fill_(.1)
        changed = replace(self.batch, features=self.batch.features+100)
        torch.testing.assert_close(m(self.h, self.batch).log_amplitude,
                                   m(self.h, changed).log_amplitude, rtol=0, atol=0)

    def test_content_sensitive_after_learning(self):
        m = self.model()
        with torch.no_grad():
            m.regimen.weight[0, 0] = .2
        changed = replace(self.batch, features=self.batch.features+1)
        self.assertFalse(torch.equal(m(self.h, self.batch).log_amplitude,
                                     m(self.h, changed).log_amplitude))

    def test_no_batch_context_residue(self):
        m = self.model()
        with torch.no_grad():
            m.regimen.weight.fill_(.1)
        a = m(self.h, self.batch).log_amplitude.clone()
        m(self.h, replace(self.batch, features=self.batch.features+100))
        torch.testing.assert_close(m(self.h, self.batch).log_amplitude, a, rtol=0, atol=0)

    def test_gradient_and_actual_update_baseline_unchanged(self):
        m = self.model(); before = {k: v.clone() for k, v in m.baseline.state_dict().items()}
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
        loss = competing_nll(m(self.h, self.batch), torch.tensor([30., 90., 365.]), torch.tensor([1, 2, 0]))
        loss.backward()
        self.assertGreater(float(m.regimen.weight.grad.abs().sum()), 0)
        self.assertGreater(float(m.teacher.encoder.base.text_projection[1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))
        opt.step()
        self.assertGreater(float(m.regimen.weight.detach().abs().sum()), 0)
        for k, v in m.baseline.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_invalid_interface(self):
        m = self.model()
        with self.assertRaises(ValueError):
            m.injection(torch.zeros(3, 35), self.batch.available)
        with self.assertRaises(ValueError):
            m.injection(self.batch.features, self.batch.available.float())

    def test_strict_best_and_patience_anchor_separate(self):
        from training.v6.early_stopping import EarlyStopping
        early = EarlyStopping(7, 10, 1e-5)
        stops = []
        for epoch in range(11):
            _, stop = early.observe(epoch, .1-epoch*1e-8)
            if stop:
                stops.append(epoch)
        self.assertEqual(early.best_epoch, 10)
        self.assertEqual(early.anchor, .1)
        self.assertEqual(stops, [10])

    def test_tie_is_not_nomination_and_material_gate(self):
        from training.v6.regimen_joint_experiment import lock
        def metric(ibs=.1, auc=.8, death=.06):
            return dict(readmission=dict(integrated_brier_7_365=ibs, points={'90': {'auc': auc}}),
                        death=dict(integrated_brier_7_365=death))
        r = dict(reference={'metrics': metric()}, fits={a: {'metrics': metric()} for a in ('availability', 'content')})
        self.assertEqual(lock(r)['nominated'], [])
        r['fits']['content']['metrics'] = metric(.098, .81, .065)
        self.assertEqual(lock(r)['nominated'], ['content'])
        self.assertFalse(lock(r)['material_signal'])


if __name__ == '__main__':
    unittest.main()
