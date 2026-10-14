import unittest
from types import SimpleNamespace
import numpy as np
import torch
from training.v6.fast_multiview import Reader,Correction,Batch,vicreg,eligible_split,digest
from training.v6.curve_adaptation import FrozenBatch

class Tests(unittest.TestCase):
    def batch(self):
        f=FrozenBatch(torch.zeros(4,32),torch.zeros(4,4),torch.zeros(4,2),torch.zeros(2),torch.ones(4),torch.zeros(4,dtype=torch.long))
        return Batch(f,torch.randn(4,32),torch.tensor([[1.,1.],[0.,1.],[1.,0.],[0.,0.]]))
    def test_zero_risk(self):
        b=self.batch();out=Correction()(b)
        torch.testing.assert_close(out.log_amplitude,torch.zeros(4,2),atol=0,rtol=0)
        torch.testing.assert_close(out.shape,torch.full((4,2),float(torch.nn.functional.softplus(torch.tensor(0.))+.1)),atol=0,rtol=0)
    def test_missing_gate_after_update(self):
        b=self.batch();m=Correction()
        with torch.no_grad():m.head[-1].weight.fill_(.1);m.head[-1].bias.fill_(.1)
        out=m(b);self.assertTrue(out.delta[0].ne(0).any());self.assertTrue(out.delta[1:].eq(0).all())
    def test_reader_shared_init(self):
        a,b=Reader(),Reader()
        self.assertTrue(all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items()))
    def test_risk_capacity(self):self.assertEqual(sum(p.numel() for p in Correction().parameters()),1140)
    def test_formula_against_manual(self):
        torch.manual_seed(42);x=torch.randn(11,16);y=torch.randn(11,16)
        expected=25*(x-y).square().mean()
        expected+=25*(torch.relu(1-torch.sqrt(x.var(0)+1e-4)).mean()+torch.relu(1-torch.sqrt(y.var(0)+1e-4)).mean())/2
        for a in (x,y):
            c=(a-a.mean(0)).T@(a-a.mean(0))/10
            expected+=(c.square().sum()-c.diag().square().sum())/16
        torch.testing.assert_close(vicreg(x,y),expected)
    def test_symmetry_and_permutation(self):
        x,y=torch.randn(20,16),torch.randn(20,16);order=torch.randperm(20)
        torch.testing.assert_close(vicreg(x,y),vicreg(y,x));torch.testing.assert_close(vicreg(x,y),vicreg(x[order],y[order]))
    def test_variance_collapse_penalty_and_gradient(self):
        x=torch.zeros(32,16,requires_grad=True);y=torch.randn(32,16,requires_grad=True)
        loss=vicreg(x,y);self.assertGreater(float(vicreg(x,x)),20.);loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.abs().sum()),0.)
    def test_small_batch_reject(self):
        with self.assertRaises(ValueError):vicreg(torch.zeros(1,16),torch.zeros(1,16))
    def test_patient_split(self):
        subjects=np.repeat(np.arange(1000),2);paired=np.ones(2000,bool);train=np.arange(2000)
        a,b=eligible_split(subjects,train,paired);self.assertFalse(set(subjects[a])&set(subjects[b]));self.assertEqual(len(a)+len(b),2000)
    def test_digest(self):
        self.assertEqual(digest({'x':np.zeros((2,2))}),digest({'x':np.zeros((2,2))}))
        self.assertNotEqual(digest({'x':np.zeros((2,2))}),digest({'x':np.zeros(4)}))

if __name__=='__main__':unittest.main()
