"""扩展冻结、组合公式、缺失/排列/数值编码以及全部探索配方的合成检查。"""
import unittest
from dataclasses import replace
import numpy as np
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.exploration_model import ExpandedBaseline, expanded_inputs, ExplorationBatch, ExplorationModel, Piecewise, field_states
from training.v6.model import HistoryBaseline
from training.v6.architecture_experiment import omit_current, configurations
from training.v6.risk import competing_nll
from patient_data.clinical_combinations import ratios, scale
from training.v6.encode_screen_candidates import route, FUTURE
from training.v6.nonlinear_baseline_probe import NonlinearExpandedBaseline


def payload():
    c=compact()
    return ExplorationBatch(omit_current(c),torch.cat((torch.ones(3,5,6),torch.ones(3,5,3)),2),
                            torch.randn(3,8),torch.tensor([[0,0],[1,3],[1,1]]))


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_expanded_baseline_starts_at_same_H(self):
        h=HistoryBaseline(); b=ExpandedBaseline(h); x=torch.randn(3,6); c=compact()
        expected=h(x); actual=b(expanded_inputs(x,c))
        torch.testing.assert_close(expected.log_amplitude,actual.log_amplitude)
        torch.testing.assert_close(expected.shape,actual.shape)
        self.assertEqual(sum(p.numel() for p in b.parameters()),72)

    def test_dp_omission_removes_masks_and_preserves_other_inputs(self):
        c=compact(); b=omit_current(c)
        self.assertEqual(b.clinical.current_values.abs().sum(),0)
        self.assertFalse(b.clinical.current_masks.any())
        torch.testing.assert_close(c.clinical.lab_axis_inputs,b.clinical.lab_axis_inputs)
        torch.testing.assert_close(c.med_features,b.med_features)

    def test_zero_start_and_frozen_gradient_all_modes(self):
        x=torch.randn(3,34); b=payload()
        for mode in ('pooled','retained','set','piecewise'):
            m=ExplorationModel(ExpandedBaseline(HistoryBaseline()),8,width=16,mode=mode,
                               knots=torch.linspace(-2,2,7).expand(22,6,7))
            actual,expected=m(x,b),m.baseline(x)
            torch.testing.assert_close(actual.log_amplitude,expected.log_amplitude)
            torch.testing.assert_close(actual.shape,expected.shape)
            competing_nll(actual,torch.tensor([30.,90.,365.]),torch.tensor([1,2,0])).backward()
            self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))

    def test_text_order_invariance_and_route_states(self):
        m=ExplorationModel(ExpandedBaseline(HistoryBaseline()),8,width=16,mode='retained').eval()
        b=payload(); c=b.compact.clinical
        states,present=field_states(m.encoder,c,3)
        order=torch.tensor([5,3,1,4,2,0])
        s,p=field_states(m.encoder,replace(c,snippet_vectors=c.snippet_vectors[order],snippet_index=c.snippet_index[order]),3)
        torch.testing.assert_close(states,s); torch.testing.assert_close(present,p)
        self.assertEqual(states.shape,(3,4,16)); self.assertFalse(present[2].any())

    def test_all_missing_set_no_nan(self):
        b=payload(); c=b.compact; old=c.clinical
        c=replace(c,clinical=replace(old,lab_axis_inputs=torch.zeros_like(old.lab_axis_inputs),
                    snippet_vectors=old.snippet_vectors[:0],snippet_index=old.snippet_index[:0]),med_features=torch.zeros_like(c.med_features),
                  micro_values=torch.zeros_like(c.micro_values),micro_masks=torch.zeros_like(c.micro_masks),
                  support_values=torch.zeros_like(c.support_values),support_masks=torch.zeros_like(c.support_masks))
        b=replace(b,compact=c,composite=torch.zeros_like(b.composite),candidate_vectors=b.candidate_vectors[:0],candidate_index=b.candidate_index[:0])
        for mode in ('pooled','retained','set','piecewise'):
            m=ExplorationModel(ExpandedBaseline(HistoryBaseline()),8,width=16,mode=mode,knots=torch.zeros(22,6,7)).eval()
            h,branches=m.encoder(b)
            self.assertTrue(torch.isfinite(h).all())
            for v in branches.values(): self.assertEqual(v.abs().sum(),0)

    def test_piecewise_zero_iqr_and_bounds(self):
        p=Piecewise(torch.zeros(22,6,7)); x=torch.ones(3,22,9)
        y=p(x); self.assertEqual(y.shape,(3,22,39)); self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(((y[:,:,:36]>=0)&(y[:,:,:36]<=1)).all())

    def test_ratios_and_zero_denominator(self):
        self.assertEqual(ratios(80,10,200),(8,1600))
        self.assertEqual(ratios(0,10,200),(0,0))
        for n,l in ((1,0),(-1,2),(float('inf'),2)): self.assertEqual(ratios(n,l),(None,None))
        self.assertEqual(ratios(1,2,None),(.5,None))

    def test_train_only_scale(self):
        raw=np.ones((3,5,9)); raw[:, :, :6]=np.array([1,2,1e5])[:,None,None]
        a,p=scale(raw,np.array(['train','train','test']))
        raw[2,:,:6]=1e20; b,q=scale(raw,np.array(['train','train','test']))
        self.assertEqual(p,q); np.testing.assert_equal(a[:2],b[:2])
        raw[:,:,6]=0; a,_=scale(raw,np.array(['train','train','test'])); self.assertEqual(abs(a[:,:,:6]).sum(),0)

    def test_routes_and_future_quarantine_guard(self):
        self.assertEqual(route('hospital_course'),2); self.assertEqual(route('discharge_medications'),3)
        self.assertEqual(route('past_medical_history'),1); self.assertEqual(route('history_of_present_illness'),0)
        self.assertIsNotNone(FUTURE.search('After discharge he was readmitted.'))
        self.assertIsNone(FUTURE.search('Please follow up after discharge.'))

    def test_candidate_configs_are_isolated(self):
        c=configurations('explore')
        self.assertEqual(c['nlr_sii'],{'combination':'ratios'})
        self.assertEqual(c['pool64'],{'width':64}); self.assertEqual(c['bert_screen'],{'text':'fused'})

    def test_nonlinear_baseline_seed_and_zero_start(self):
        h=HistoryBaseline(); x=torch.randn(3,34)
        NonlinearExpandedBaseline.initialization_seed=7
        torch.manual_seed(123); a=NonlinearExpandedBaseline(h)
        torch.manual_seed(456); b=NonlinearExpandedBaseline(h)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        torch.testing.assert_close(a(x).log_amplitude,h(x[:,:6]).log_amplitude)
        NonlinearExpandedBaseline.initialization_seed=42


if __name__=='__main__': unittest.main()
