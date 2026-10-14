import unittest
from dataclasses import replace
import numpy as np
import torch
from patient_data.medication_identity import identity, component_dose
from training.v6.token_model import (TokenBatch,TokenModel,TokenEncoder,fast_cif,daily_probabilities,masked_reconstruction_batch)
from training.v6.exploration_model import ExpandedBaseline
from training.v6.model import HistoryBaseline
from training.v6.risk import predict_cif,competing_nll


def sample():
    n=3; labs=torch.zeros(n,29,20); labs[:,:,6]=1; labs[:,:,0]=.3
    valid=torch.zeros_like(labs,dtype=torch.bool);valid[:,:,:3]=True
    return TokenBatch(labs,valid,torch.ones(n,29,dtype=torch.bool),torch.ones(n,7,4),
        torch.zeros(n,21,23),torch.zeros(n,21,dtype=torch.bool),torch.ones(n,8,2),
        torch.randn(n,4,768),torch.arange(4).expand(n,4),torch.ones(n,4,dtype=torch.bool),
        torch.zeros(n,29,dtype=torch.bool))


class TestTokens(unittest.TestCase):
    def test_combo_and_units(self):
        self.assertEqual(identity('Albuterol-Ipratropium'), 'albuterol_ipratropium')
        self.assertEqual(identity('Advair Diskus'), 'fluticasone_salmeterol')
        self.assertIsNone(identity('Sodium Chloride'))
        d=dict(parent_field_ordinal='1',product_description='Albuterol',dose_given='2',dose_given_unit='puffs')
        self.assertEqual(component_dose([d,d],'albuterol')[:2],(2.,'puff'))
        self.assertIsNone(component_dose([d,dict(d,dose_given='3')],'albuterol')[0])
        self.assertEqual(component_dose([dict(d,dose_given='100',dose_given_unit='mcg')],'albuterol')[:2],(.1,'mg'))

    def test_daily_cif_matches_and_mass(self):
        a=np.array([[-2.,-1.],[-.2,-3.]]);k=np.array([[.2,1.7],[2.,.15]])
        c=fast_cif(a,k);old=predict_cif(a,k,np.arange(366))
        np.testing.assert_allclose(c,old,atol=1e-12)
        daily,conditional=daily_probabilities(c)
        np.testing.assert_allclose(daily.sum(1),c[:,-1,1:],atol=1e-12)
        self.assertTrue((conditional.sum(2)<=1+1e-10).all())
        self.assertTrue((daily>=-1e-10).all())

    def test_zero_start_and_frozen_gradients(self):
        b=sample();base=ExpandedBaseline(HistoryBaseline());m=TokenModel(base,interaction=True).eval();h=torch.randn(3,34)
        out=m(h,b);expected=base(h)
        torch.testing.assert_close(out.log_amplitude,expected.log_amplitude)
        torch.testing.assert_close(out.shape,expected.shape)
        competing_nll(out,torch.tensor([10.,200.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))

    def test_text_order_and_empty(self):
        b=sample();enc=TokenEncoder(interaction=True).eval()
        x=enc(b)[0];perm=torch.tensor([2,0,3,1])
        shuffled=replace(b,text=b.text[:,perm],text_fields=b.text_fields[:,perm],text_present=b.text_present[:,perm])
        torch.testing.assert_close(x,enc(shuffled)[0],atol=2e-6,rtol=2e-6)
        empty=replace(b,lab_present=b.lab_present&False,coarse=b.coarse*0,named_present=b.named_present&False,
                      other=b.other*0,text_present=b.text_present&False)
        torch.testing.assert_close(enc(empty)[0],torch.zeros(3,128))

    def test_mask_only_observed(self):
        b=sample();b=replace(b,lab_present=b.lab_present & False,numeric_valid=b.numeric_valid & False)
        masked,target=masked_reconstruction_batch(b,42,rate=1.)
        self.assertFalse(target.any());self.assertFalse(masked.masked_labs.any())
        self.assertFalse(hasattr(b,'events'));self.assertFalse(hasattr(b,'times'))

    def test_same_analyte_masked_together(self):
        from patient_data.v6_lab_state import AXES
        names=list(AXES);b=sample()
        for seed in range(10):
            masked,_=masked_reconstruction_batch(b,seed)
            for left,right in (('glucose_low','glucose_high'),('wbc_low','wbc_high'),
                               ('sodium_low','sodium_high'),('potassium_low','potassium_high'),
                               ('arterial_low_ph','arterial_high_pco2'),('bicarbonate_low','bicarbonate_high')):
                torch.testing.assert_close(masked.masked_labs[:,names.index(left)],masked.masked_labs[:,names.index(right)])
            torch.testing.assert_close(masked.masked_labs[:,27],masked.masked_labs[:,28])


if __name__=='__main__':unittest.main()
