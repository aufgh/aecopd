from types import SimpleNamespace
import unittest
import numpy as np
import torch
from training.v6.discharge_regimen_screen import Correction,feature_view,lock
from training.v6.fast_multiview import Batch


class RegimenScreenTests(unittest.TestCase):
    def batch(self,present=True):
        f=SimpleNamespace(hidden=torch.randn(3,32),raw=torch.randn(3,4),prior=torch.randn(3,2),shape_raw=torch.zeros(2))
        return Batch(f,torch.randn(3,34),torch.tensor([[float(present),1.]]*3))

    def test_capacity_and_zero_start(self):
        m=Correction();self.assertEqual(sum(p.numel() for p in m.parameters()),1172)
        b=self.batch();out=m(b);old=b.frozen.raw.reshape(-1,2,2)
        torch.testing.assert_close(out.log_amplitude,(b.frozen.prior+old[:,:,0]).clamp(-10,6),rtol=0,atol=0)
        torch.testing.assert_close(out.shape,torch.nn.functional.softplus(b.frozen.shape_raw+old[:,:,1])+.1,rtol=0,atol=0)

    def test_missing_gate_nonzero_head(self):
        m=Correction();b=self.batch(False)
        with torch.no_grad():m.head[-1].weight.fill_(1.);m.head[-1].bias.fill_(1.)
        self.assertEqual(m(b).delta.abs().sum(),0)

    def test_availability_does_not_see_content(self):
        x=np.random.default_rng(2).normal(size=(3,34)).astype(np.float32)
        np.testing.assert_array_equal(feature_view(x,'availability'),feature_view(1-x,'availability'))
        np.testing.assert_array_equal(feature_view(x,'content'),x)

    def test_views_do_not_mutate_source(self):
        x=np.ones((3,34),np.float32);y=feature_view(x,'content');y[:]=0;self.assertEqual(x.sum(),102)

    def test_not_nominate_tie(self):
        metric=lambda v:dict(metrics={'readmission':{'integrated_brier_7_365':v}})
        r=dict(reference=metric(1)['metrics'],fits={a:metric(1) for a in ('availability','content')})
        self.assertEqual(lock(r)['nominated'],[])

    def test_content_must_win_both(self):
        metric=lambda v:dict(metrics={'readmission':{'integrated_brier_7_365':v}})
        r=dict(reference=metric(1)['metrics'],fits={'availability':metric(.9),'content':metric(.8)})
        self.assertEqual(lock(r)['nominated'],['content'])


if __name__=='__main__':unittest.main()
