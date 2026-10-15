"""强 V6 风险保持、Query 实际旁路、共有初值与风险梯度路径。"""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from training.tests.test_v7_model import ToyBaseline, toy_batch
from training.v7.strong_anchor_experiment import StrongV7, StrongBatch
from training.v7.strong_no_cross import StrongV7NoCross


class StrongNoCrossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.batch = StrongBatch(toy_batch(), torch.tensor([0, 1]))
        self.anchors = dict(anchor_a=np.array([[-2., -3.], [-1., -2.]], np.float32),
                            anchor_k=np.array([[.2, 2.], [1.3, .5]], np.float32),
                            valid=np.ones(2, bool))
        self.base = ToyBaseline()
        self.parents = torch.ones(38, 10)

    def model(self):
        return StrongV7NoCross(self.base, self.parents, self.anchors).eval()

    def test_identical_initial_parameters_anchor_and_optimizer(self):
        control = StrongV7(self.base, self.parents, self.anchors).eval()
        model = self.model()
        for name, value in control.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        out = model(self.batch)
        torch.testing.assert_close(out.log_amplitude, model.anchor_a, rtol=0, atol=0)
        torch.testing.assert_close(out.shape, model.anchor_k, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 9337424)
        groups, names = model.optimizer_groups()
        self.assertEqual([g['lr'] for g in groups], [1e-4, 3e-4])
        self.assertFalse(any(n.startswith(('drug_cross.', 'rad_cross.')) for group in names.values() for n in group))

    def test_cross_modules_are_never_called_and_cannot_change_risk(self):
        model = self.model()
        torch.nn.init.normal_(model.delta_head[-1].weight, std=.02)
        with patch.object(model.drug_cross, 'forward', side_effect=AssertionError('drug cross called')), \
             patch.object(model.rad_cross, 'forward', side_effect=AssertionError('rad cross called')):
            before = model(self.batch)
            with torch.no_grad():
                for module in (model.drug_cross, model.rad_cross):
                    for parameter in module.parameters():
                        parameter.fill_(1e5)
            after = model(self.batch)
        torch.testing.assert_close(before.log_amplitude, after.log_amplitude, rtol=0, atol=0)
        torch.testing.assert_close(before.shape, after.shape, rtol=0, atol=0)

    def test_risk_gradients_reach_encoders_but_not_frozen_modules(self):
        model = self.model()
        torch.nn.init.normal_(model.delta_head[-1].weight, std=.02)
        out = model(self.batch)
        (out.log_amplitude.square().mean() + out.shape.square().mean()).backward()
        for module in (model.baseline, model.drug_cross, model.rad_cross, model.contrast_projection):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        for module in (model.lab_projection, model.drug_numeric, model.text_projection,
                       model.temporal, model.history_pool, *(p.project for p in model.field_pool)):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))


if __name__ == '__main__':
    unittest.main()
