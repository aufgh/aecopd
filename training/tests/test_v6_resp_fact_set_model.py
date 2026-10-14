import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from training.v6.resp_fact_set_model import FactCorrection,FactBatch

def batch():
    torch.manual_seed(10)
    frozen=SimpleNamespace(hidden=torch.randn(3,32),raw=torch.randn(3,4),prior=torch.randn(3,2),shape_raw=torch.randn(2))
    return FactBatch(frozen,torch.tensor([0.,1.,2.]),torch.tensor([0,7,7]),torch.tensor([0,0,2]),
        torch.tensor([0,0,0]),torch.tensor([0,1,1]),torch.tensor([0,2,3]),torch.tensor([False,True,True]),
        torch.tensor([False,True,True]),torch.tensor([0,0,1]),torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,0.,0.]]),torch.tensor([True,True,False]))

def active(arm):
    m=FactCorrection(arm)
    with torch.no_grad():m.head[-1].weight.fill_(.03);m.head[-1].bias.fill_(.1)
    return m

class Tests(unittest.TestCase):
    def test_capacity_and_shared_initialization(self):
        a,b=FactCorrection('availability'),FactCorrection('content')
        self.assertEqual(sum(p.numel() for p in a.parameters()),998)
        for key,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[key],rtol=0,atol=0)

    def test_initial_original_risk(self):
        b=batch();f=b.frozen;o=FactCorrection('content')(b)
        torch.testing.assert_close(o.log_amplitude,(f.prior+f.raw.reshape(3,2,2)[:,:,0]).clamp(-10,6),atol=0,rtol=0)
        torch.testing.assert_close(o.shape,F.softplus(f.shape_raw+f.raw.reshape(3,2,2)[:,:,1])+.1,atol=0,rtol=0)

    def test_availability_all_content_hidden(self):
        b=batch();m=active('availability');a=m(b)
        changed=replace(b,values=b.values+40,concepts=(b.concepts+1)%8,roles=(b.roles+1)%3,
            polarities=1-b.polarities,units=(b.units+1)%3,conditions=(b.conditions+1)%4,
            numeric_present=torch.zeros(3,dtype=torch.bool),numeric_valid=torch.zeros(3,dtype=torch.bool))
        c=m(changed)
        torch.testing.assert_close(a.log_amplitude,c.log_amplitude,atol=0,rtol=0)
        torch.testing.assert_close(a.shape,c.shape,atol=0,rtol=0)

    def test_availability_count_not_visible(self):
        b=batch();m=active('availability');a=m(b)
        c=replace(b,**{k:getattr(b,k)[:0] for k in ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid','owners')})
        torch.testing.assert_close(a.log_amplitude,m(c).log_amplitude,atol=0,rtol=0)

    def test_missing_source_original_after_update(self):
        b=batch();o=active('content')(b);f=b.frozen
        torch.testing.assert_close(o.delta[2],torch.zeros(2,2),atol=0,rtol=0)
        torch.testing.assert_close(o.log_amplitude[2],(f.prior+f.raw.reshape(3,2,2)[:,:,0])[2].clamp(-10,6),atol=0,rtol=0)

    def test_permutation(self):
        b=batch();m=active('content');ix=torch.tensor([2,0,1])
        c=replace(b,**{k:getattr(b,k)[ix] for k in ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid','owners')})
        torch.testing.assert_close(m(b).log_amplitude,m(c).log_amplitude,atol=1e-6,rtol=1e-6)

    def test_invalid_scale_values_masked(self):
        b=replace(batch(),numeric_valid=torch.tensor([False,False,True]));m=active('content');a=m(b)
        changed=replace(b,values=torch.tensor([99.,55.,2.]));torch.testing.assert_close(a.log_amplitude,m(changed).log_amplitude,atol=0,rtol=0)

    def test_content_reads_typed_field(self):
        b=batch();m=active('content');changed=replace(b,concepts=torch.tensor([1,7,7]))
        self.assertFalse(torch.equal(m(b).log_amplitude,m(changed).log_amplitude))

    def test_empty_source_finite(self):
        b=batch();b=replace(b,**{k:getattr(b,k)[:0] for k in ('values','concepts','roles','polarities','units','conditions','numeric_present','numeric_valid','owners')},
            aggregate=torch.zeros(3,3),observed=torch.zeros(3,dtype=torch.bool))
        o=active('content')(b);self.assertTrue(torch.isfinite(o.log_amplitude).all());self.assertTrue(torch.isfinite(o.shape).all())

    def test_bad_mask_rejected(self):
        with self.assertRaises(ValueError):FactCorrection('content')(replace(batch(),numeric_present=torch.zeros(3,dtype=torch.bool)))

if __name__=='__main__':unittest.main()
