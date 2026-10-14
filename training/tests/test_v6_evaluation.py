"""病例/竞争对照、时间排序、患者配对区间与常数对照。"""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from training.v6.evaluation import metrics, paired_interval, calibration_bins
from training.v6.model import HistoryBaseline, ConstantCorrection
from training.v6.plot import roc
from training.v5.competing_concordance import cif_concordance
from training.v6.train import fit, predict


class EvaluationTests(unittest.TestCase):
    def test_cumulative_cases_and_competing_controls(self):
        times = np.r_[np.full(30, 10.), np.full(30, 5.), np.full(40, 365.)]
        events = np.r_[np.ones(30, int), np.full(30, 2), np.zeros(40, int)]
        risk = np.repeat(np.r_[np.full(30, .9), np.full(30, .1), np.full(40, .2)][:, None], 366, axis=1)
        m = metrics(times, events, risk, ordering=True)
        self.assertEqual(m['points']['30']['cases'], 30)
        self.assertEqual(m['points']['30']['controls_including_competing_events'], 70)
        self.assertEqual(m['points']['30']['auc'], 1.)
        y = (events == 1).astype(int)
        x, yy = roc(risk[:, 30], y)
        self.assertAlmostEqual(np.trapezoid(yy, x), roc_auc_score(y, risk[:, 30]))

    def test_early_censoring_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.array([10.]), np.array([0]), np.zeros((1, 366)))

    def test_ordering_competitor_early_target_later_and_tied_times(self):
        # 两个同日目标互不配对；竞争死亡虽较早仍作为目标的可比对象。
        r = cif_concordance(np.array([20., 20., 30., 5., 365.]), np.array([1,1,1,2,0]),
                            np.array([.9,.9,.8,.1,.1]))
        self.assertEqual(r['comparable_pairs'], 8)
        self.assertEqual(r['c_index'], 1.)

    def test_identical_models_have_zero_cluster_difference(self):
        n = 80
        times = np.r_[np.full(30, 30.), np.full(50, 365.)]
        events = np.r_[np.ones(30, int), np.zeros(50, int)]
        risk = np.repeat(np.linspace(.01,.8,n)[:,None], 366, axis=1)
        r = paired_interval(np.repeat(np.arange(40),2), times, events, risk, risk, replicates=20)
        self.assertEqual(r['integrated_brier_95ci'], [0., 0.])
        self.assertEqual(r['auc90_95ci'], [0., 0.])

    def test_calibration_does_not_split_identical_scores(self):
        bins = calibration_bins(np.full(40,365), np.zeros(40), np.full((40,366),.2))
        self.assertEqual(len(bins), 1)
        self.assertEqual(bins[0]['n'], 40)

    def test_constant_correction_frozen_and_initial_equivalence(self):
        base = HistoryBaseline()
        h = torch.randn(5,6)
        expected = base(h)
        model = ConstantCorrection(base)
        output = model(h)
        torch.testing.assert_close(output.log_amplitude, expected.log_amplitude)
        torch.testing.assert_close(output.shape, expected.shape)
        output.log_amplitude.sum().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(model.correction.grad.abs().sum().item(), 0)

    def test_fit_checkpoint_reload_and_frozen_baseline(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        n = 60
        h = torch.randn(n, 6)
        times = torch.cat((torch.full((20,), 30.), torch.full((20,), 20.), torch.full((20,), 365.)))
        events = torch.cat((torch.ones(20, dtype=torch.long), torch.full((20,), 2), torch.zeros(20, dtype=torch.long)))
        model = ConstantCorrection(HistoryBaseline())
        frozen = {k: v.clone() for k, v in model.baseline.state_dict().items()}
        batches = [(np.arange(n), h, None, times, events)]
        data = SimpleNamespace(times=times.numpy(), events=events.numpy())
        with tempfile.TemporaryDirectory() as directory:
            result = fit(model, data, batches, batches, np.arange(n), Path(directory) / 'constant',
                         epochs=2, lr=.001, seed=9, patience=10)
            self.assertTrue(result['checkpoint_reloaded'])
            self.assertEqual(result['epochs_trained'], 2)
            self.assertTrue(all(torch.equal(v, frozen[k]) for k, v in model.baseline.state_dict().items()))
            a, k, loss, _ = predict(model, batches)
            self.assertTrue(np.isfinite(a).all() and np.isfinite(k).all() and np.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
