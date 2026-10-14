import unittest
import numpy as np
import torch
from training.v6.curve_score import daily_cif,curve_scores
from training.v6.token_model import fast_cif
from training.v6.risk import RiskOutput
from training.v6.evaluation import metrics


class Tests(unittest.TestCase):
    def parameters(self):
        return (torch.tensor([[-3.,-2.],[1.,-1.],[-10.,6.]],dtype=torch.float64),
                torch.tensor([[.15,.8],[1.,2.],[3.,.1]],dtype=torch.float64))

    def test_official_forward_and_probability_conservation(self):
        a,k=self.parameters();curve=daily_cif(RiskOutput(a,k)).numpy()
        np.testing.assert_allclose(curve,fast_cif(a.numpy(),k.numpy()),atol=1e-12,rtol=0)
        np.testing.assert_allclose(curve.sum(2),1.,atol=1e-12,rtol=0)
        self.assertGreaterEqual(np.diff(curve[:,:,1:],axis=1).min(),-1e-12)
        np.testing.assert_array_equal(curve[:,0],np.tile([1.,0.,0.],(3,1)))

    def test_evaluation_ibs_identity_and_competing_labels(self):
        a,k=self.parameters();curve=daily_cif(RiskOutput(a,k))
        t=torch.tensor([0.,30.,365.]);e=torch.tensor([1,2,0])
        scores=curve_scores(curve,t,e)
        for name,cause in [('readmission',1),('death',2)]:
            reference=metrics(t.numpy(),e.numpy(),curve[:,:,cause].numpy(),cause=cause)
            self.assertAlmostEqual(float(scores[name].mean()),reference['integrated_brier_7_365'],places=13)
        self.assertTrue(torch.all(scores['joint']>=scores['readmission']+scores['death']))

    def test_gradcheck_with_subunit_and_superunit_shapes(self):
        a=torch.tensor([[-2.,-1.]],dtype=torch.float64,requires_grad=True)
        k=torch.tensor([[.3,1.7]],dtype=torch.float64,requires_grad=True)
        def f(a,k):
            return curve_scores(daily_cif(RiskOutput(a,k)),torch.tensor([22.]),torch.tensor([1]))['joint'].mean()
        self.assertTrue(torch.autograd.gradcheck(f,(a,k),eps=1e-6,atol=1e-6,rtol=1e-4))

    def test_invalid_early_censoring_rejected(self):
        a,k=self.parameters();curve=daily_cif(RiskOutput(a,k))
        with self.assertRaises(ValueError):curve_scores(curve,torch.tensor([10.,365.,365.]),torch.tensor([0,0,0]))

    @unittest.skipUnless(torch.cuda.is_available(),'real CUDA backward-precision regression')
    def test_fp32_backward_matches_double_linear_reference(self):
        torch.manual_seed(17);layer=torch.nn.Linear(32,4).cuda();hidden=torch.randn(256,32,device='cuda')
        original=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            raw=layer(hidden);out=RiskOutput(raw[:,[0,2]],torch.nn.functional.softplus(raw[:,[1,3]])+.1)
            loss=curve_scores(daily_cif(out),torch.full((256,),30.,device='cuda'),
                              torch.ones(256,dtype=torch.long,device='cuda'))['joint'].sum()/13592
            g=torch.autograd.grad(loss,raw,retain_graph=True)[0]
            reference=g.double().T@hidden.double()
            torch.backends.cuda.matmul.allow_tf32=False
            actual=torch.autograd.grad(loss,layer.weight)[0].double()
            self.assertLess(float((actual-reference).norm()/reference.norm()),1e-6)
        finally:torch.backends.cuda.matmul.allow_tf32=original


if __name__=='__main__':unittest.main()
