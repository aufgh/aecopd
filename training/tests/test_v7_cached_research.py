"""零修正精确还原、合法shape和同容量软路由。"""
import unittest
import torch
from training.v7.cached_research import Reader


class CachedResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_initial_anchor_exact_for_all_readers(self):
        x=torch.randn(3,512);a=torch.randn(3,2);k=torch.tensor([[.2,.5],[1.,2.],[.1,4.]])
        for view in ('global','ple','v7_frozen','routing_uniform','routing_prior'):
            m=Reader(view).eval();out=m(x,a,k)
            torch.testing.assert_close(out.log_amplitude,a,rtol=0,atol=0)
            torch.testing.assert_close(out.shape,k,rtol=0,atol=0)

    def test_route_only_initial_bias_differs(self):
        left,right=Reader('routing_uniform'),Reader('routing_prior')
        self.assertEqual(sum(p.numel() for p in left.parameters()),sum(p.numel() for p in right.parameters()))
        for name,p in left.named_parameters():
            if name!='route':torch.testing.assert_close(p,dict(right.named_parameters())[name],rtol=0,atol=0)
        self.assertFalse(torch.equal(left.route,right.route))

    def test_nonzero_update_bounded_positive_shape(self):
        m=Reader('global');x=torch.zeros(2,512);a=torch.zeros(2,2);k=torch.ones(2,2)
        with torch.no_grad():m.global_raw.copy_(torch.tensor([20.,-100.,-20.,100.]))
        out=m(x,a,k);self.assertTrue((out.shape>0).all());self.assertTrue(torch.isfinite(out.shape).all())
        self.assertLessEqual(float(out.log_amplitude.max().detach()),6.)
        self.assertGreaterEqual(float(out.log_amplitude.min().detach()),-10.)


if __name__=='__main__':unittest.main()
