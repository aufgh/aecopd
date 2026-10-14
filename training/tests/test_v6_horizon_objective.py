"""90天目标边界、同分指标、共同骨干、零残差与冻结梯度。"""
import copy
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.architecture_experiment import omit_current
from training.v6.exploration_model import ExpandedBaseline, expanded_inputs
from training.v6.compact_model import CompactAnchoredModel
from training.v6.model import HistoryBaseline
from training.v6.horizon_objective import (labels90, ranked_curves, binary_metrics, paired90,
    BinaryBaseline, BinaryConstant, BinaryResidual, loss_for, fit90, predict90)


class HorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_labels_boundaries_death_and_future_event(self):
        np.testing.assert_equal(labels90(np.array([0., 90., 90.01, 20., 365., 365.]),
                                          np.array([1, 1, 1, 2, 0, 2])), [1, 1, 0, 0, 0, 0])
        for t,e in (([89], [0]), ([float('nan')], [1]), ([-1], [1]), ([366], [1]), ([90], [3])):
            with self.assertRaises(ValueError): labels90(np.array(t), np.array(e))

    def test_metrics_ties_and_constant_predictor(self):
        y=np.array([1, 0, 1, 0]); p=np.ones(4)*.5
        m=binary_metrics(y,p)
        self.assertEqual(m['auc90'], .5); self.assertEqual(m['average_precision'], .5)
        self.assertEqual(m['brier90'], .25); self.assertEqual(len(m['calibration_bins']),1)
        self.assertAlmostEqual(m['logloss90'], np.log(2))
        m=binary_metrics(y,np.array([.9,.3,.8,.2]))
        self.assertEqual(m['auc90'],1.); self.assertEqual(m['average_precision'],1.)
        np.testing.assert_equal(ranked_curves(y,p)['fpr'], [0,1])
        with self.assertRaises(ValueError): binary_metrics(y,np.array([.5,.5,1.1,.5]))

    def test_pair_identical_predictions_zero_intervals(self):
        y=np.array([1,0,1,0]); p=np.array([.8,.4,.7,.3])
        m=paired90(np.array([10,10,20,20]),y,p,p,replicates=20)
        self.assertEqual(m['patients'],2)
        for d in m['metrics'].values():
            self.assertEqual(d['difference'],0); self.assertEqual(d['ci95'],[0,0])

    def test_encoder_initialization_equal_and_binary_zero_start(self):
        torch.manual_seed(42)
        s=CompactAnchoredModel(ExpandedBaseline(HistoryBaseline()),8,width=16)
        # 构造基础风险不属于临床骨干；正式make_model在其后重设seed。
        torch.manual_seed(42)
        s=CompactAnchoredModel(copy.deepcopy(s.baseline),8,width=16)
        base=BinaryBaseline(.1)
        torch.manual_seed(42)
        b=BinaryResidual(base,8,width=16)
        for key,value in s.encoder.state_dict().items():
            torch.testing.assert_close(value,b.encoder.state_dict()[key],rtol=0,atol=0)
        c=omit_current(compact()); h=expanded_inputs(torch.randn(3,6),compact())
        b.eval(); actual=b(h,c); expected=base(h)
        torch.testing.assert_close(actual.logits,expected.logits,rtol=0,atol=0)
        loss_for(actual,torch.tensor([90.,20.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in b.baseline.parameters()))
        self.assertGreater(b.delta_head[-1].weight.grad[0].abs().sum().item(),0)
        self.assertEqual(b.delta_head[-1].weight.grad[1:].abs().sum().item(),0)

    def test_binary_constant_has_no_individualized_delta(self):
        base=BinaryBaseline(.1); m=BinaryConstant(base); h=torch.randn(3,34)
        torch.testing.assert_close(m(h).logits,base(h).logits)
        with torch.no_grad(): m.correction.fill_(.3)
        torch.testing.assert_close(m(h).logits-base(h).logits,torch.ones(3)*.3)

    def test_clinical_input_omits_current_values_and_masks(self):
        c=compact(); omitted=omit_current(c)
        self.assertEqual(omitted.clinical.current_values.abs().sum(),0)
        self.assertFalse(omitted.clinical.current_masks.any())
        torch.testing.assert_close(omitted.med_features,c.med_features)
        torch.testing.assert_close(omitted.clinical.snippet_vectors,c.clinical.snippet_vectors)

    def test_binary_bce_extreme_logits_finite(self):
        b=BinaryBaseline(.1)
        with torch.no_grad(): b.logit.bias.fill_(100)
        loss=loss_for(b(torch.zeros(3,34)),torch.tensor([20.,90.,365.]),torch.tensor([2,1,0]))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(torch.isfinite(b.logit.bias.grad).all())

    def test_metrics_against_sklearn_and_existing_auc(self):
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
        from training.baselines.rank_metrics import auc_rank
        rng=np.random.default_rng(42)
        y=np.r_[np.ones(30,dtype=int),np.zeros(50,dtype=int)]
        for p in (rng.uniform(.01,.99,80),np.round(rng.uniform(0,1,80),1),np.ones(80)*.2):
            m=binary_metrics(y,p)
            self.assertAlmostEqual(m['auc90'],roc_auc_score(y,p))
            self.assertAlmostEqual(m['auc90'],auc_rank(p,y))
            self.assertAlmostEqual(m['average_precision'],average_precision_score(y,p))
            self.assertAlmostEqual(m['brier90'],brier_score_loss(y,p))
            # 日志损失端点截断明确为1e-8，和sklearn默认机器精度不同。
            self.assertAlmostEqual(m['logloss90'],log_loss(y,np.clip(p,1e-8,1-1e-8)))

    def test_fit_selects_brier90_and_preserves_frozen_checkpoint(self):
        base=BinaryBaseline(.1); model=BinaryResidual(base,8,width=16)
        h=torch.randn(3,34); c=omit_current(compact())
        t=torch.tensor([20.,40.,365.]); e=torch.tensor([1,2,0])
        prepared=[(np.arange(3),h,c,t,e)]
        before={k:v.clone() for k,v in base.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'fit'
            result=fit90(model,prepared,prepared,np.array([1,0,0]),directory,
                         epochs=2,lr=.001,seed=42,patience=2)
            state=torch.load(directory/'best.pt',weights_only=True)['state_dict']
            model.load_state_dict(state)
            actual=predict90(model,prepared)['risk90']
            self.assertAlmostEqual(result['best_validation_brier90'],float(np.mean((actual-[1,0,0])**2)))
            for k,v in before.items(): torch.testing.assert_close(v,model.baseline.state_dict()[k],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
