"""确保去掉原NLL后只由Focal/对比/共同正则更新，不改变旧训练函数。"""
import unittest
import torch
from torch.nn import functional as F
from training.v6.risk import RiskOutput
from training.v7.losses import objective
from training.v7.loss_comparison import loss_terms


def inputs():
    a=torch.tensor([[-1.,-2.],[-.5,-1.]],dtype=torch.float64,requires_grad=True)
    k=torch.tensor([[.8,1.2],[1.1,.9]],dtype=torch.float64,requires_grad=True)
    delta=torch.zeros(2,2,2,dtype=torch.float64,requires_grad=True)
    out=RiskOutput(a,k,delta)
    reps={n:F.normalize(torch.randn(2,256,dtype=torch.float64),dim=1).requires_grad_() for n in ('text','lab','drug','rad')}
    masks={n:torch.ones(2,dtype=torch.bool) for n in reps}
    return out,reps,masks,torch.tensor([1,2]),torch.tensor([1,2])


class LossComparisonTests(unittest.TestCase):
    def test_previous_formula_equal(self):
        out,reps,masks,subjects,events=inputs();times=torch.tensor([10.,100.],dtype=torch.float64)
        old,parts=objective(out,times,events,reps,masks,subjects,recommended=True)
        new,detail=loss_terms(out,times,events,reps,masks,subjects,nll_weight=1.,contrast_weight=.1)
        torch.testing.assert_close(old,new,rtol=1e-15,atol=1e-15)
        for key in ('nll','focal','contrast','delta_penalty'):torch.testing.assert_close(parts[key],detail[key],rtol=0,atol=0)

    def test_exact_time_within_same_horizon_does_not_enter_no_nll(self):
        out,reps,masks,subjects,events=inputs()
        first,p1=loss_terms(out,torch.tensor([8.,100.]),events,reps,masks,subjects,nll_weight=0.,contrast_weight=.1)
        second,p2=loss_terms(out,torch.tensor([13.,110.]),events,reps,masks,subjects,nll_weight=0.,contrast_weight=.1)
        self.assertNotEqual(float(p1['nll'].detach()),float(p2['nll'].detach()))
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        g1=torch.autograd.grad(first,(out.log_amplitude,out.shape),retain_graph=True)
        g2=torch.autograd.grad(second,(out.log_amplitude,out.shape),retain_graph=True)
        for a,b in zip(g1,g2,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_no_contrast_source_heads_receive_no_gradient(self):
        out,reps,masks,subjects,events=inputs()
        total,parts=loss_terms(out,torch.tensor([8.,100.]),events,reps,masks,subjects,nll_weight=0.,contrast_weight=0.)
        total.backward()
        self.assertTrue(all(v.grad is None for v in reps.values()))
        self.assertTrue(torch.isfinite(out.log_amplitude.grad).all())
        self.assertTrue(torch.isfinite(out.shape.grad).all())
        self.assertGreater(float(out.log_amplitude.grad.abs().sum()),0)
        self.assertEqual(parts['positive_pairs'],0)

    def test_nll_addition_gradient_is_exact(self):
        out,reps,masks,subjects,events=inputs();times=torch.tensor([8.,100.])
        without,_=loss_terms(out,times,events,reps,masks,subjects,nll_weight=0.,contrast_weight=0.)
        with_nll,parts=loss_terms(out,times,events,reps,masks,subjects,nll_weight=1.,contrast_weight=0.)
        ga=torch.autograd.grad(with_nll-without,out.log_amplitude,retain_graph=True)[0]
        gb=torch.autograd.grad(parts['nll'],out.log_amplitude)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-14,atol=1e-14)


if __name__=='__main__':unittest.main()
