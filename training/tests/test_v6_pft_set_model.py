import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from training.v6.pft_set_model import PFTBatch,PFTCorrection,ARMS


class Tests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(5);n=3
        f=SimpleNamespace(hidden=torch.randn(n,32),raw=torch.randn(n,4)*.1,prior=torch.randn(n,2)-2,shape_raw=torch.zeros(2))
        return PFTBatch(f,torch.randn(4),torch.tensor([0,0,2,5]),torch.randn(4,3),torch.tensor([0,0,0,1]),torch.randn(n,9),torch.tensor([1,1,0],dtype=torch.bool))

    def test_common_weights_and_initial_risk(self):
        b=self.batch();a,c=[PFTCorrection(arm) for arm in ARMS]
        self.assertEqual(sum(p.numel() for p in a.parameters()),998)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,c.state_dict()[k]))
        torch.testing.assert_close(a(b).log_amplitude,c(b).log_amplitude,atol=0,rtol=0)

    def test_availability_ignores_actual_values(self):
        b=self.batch();a=PFTCorrection('availability')
        with torch.no_grad():a.head[-1].weight.fill_(.1)
        torch.testing.assert_close(a(b).log_amplitude,a(replace(b,values=b.values*100)).log_amplitude,atol=0,rtol=0)

    def test_source_missing_stays_original(self):
        b=self.batch();a=PFTCorrection('content');before=a(b)
        with torch.no_grad():a.head[-1].weight.fill_(1);a.head[-1].bias.fill_(1)
        self.assertTrue(torch.equal(before.log_amplitude[-1],a(b).log_amplitude[-1]));self.assertTrue(torch.equal(before.shape[-1],a(b).shape[-1]))

    def test_set_permutation_invariant(self):
        b=self.batch();a=PFTCorrection('content')
        with torch.no_grad():a.head[-1].weight.fill_(.1)
        order=torch.tensor([3,2,0,1]);other=replace(b,values=b.values[order],concepts=b.concepts[order],metadata=b.metadata[order],owners=b.owners[order])
        torch.testing.assert_close(a(b).log_amplitude,a(other).log_amplitude,atol=1e-6,rtol=0)

    def test_no_tokens_finite(self):
        b=self.batch();b=replace(b,values=b.values[:0],concepts=b.concepts[:0],metadata=b.metadata[:0],owners=b.owners[:0],observed=torch.zeros(3,dtype=torch.bool))
        self.assertTrue(torch.isfinite(PFTCorrection('content')(b).shape).all())

    def test_invalid_concepts_rejected(self):
        b=self.batch()
        with self.assertRaises(ValueError):PFTCorrection('content')(replace(b,concepts=b.concepts+6))


if __name__=='__main__':unittest.main()
