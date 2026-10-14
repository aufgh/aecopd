import unittest
import torch
from torch import nn
from training.v6.detail_extension_experiment import DetailBatch,DetailEncoder


class Stub(nn.Module):
    def forward(self,batch,*,mask_pmh=False): return batch,{}


class ExtensionTests(unittest.TestCase):
    def test_empty_branch_cannot_create_observation(self):
        m=DetailEncoder(Stub())
        h=torch.randn(2,128)
        b=DetailBatch(h,torch.zeros(2,90),torch.zeros(2,192))
        out,_=m(b)
        self.assertTrue(torch.equal(out,h))

    def test_inactive_patient_not_changed_by_other_patient(self):
        m=DetailEncoder(Stub())
        h=torch.randn(2,128); labs=torch.zeros(2,90); labs[1,6]=1
        out,_=m(DetailBatch(h,labs,torch.zeros(2,192)))
        self.assertTrue(torch.equal(out[0],h[0])); self.assertFalse(torch.equal(out[1],h[1]))

    def test_nonfinite_and_wrong_shape_rejected(self):
        m=DetailEncoder(Stub()); h=torch.zeros(2,128)
        for x in (torch.zeros(2,89),torch.full((2,90),float('nan'))):
            with self.assertRaises(ValueError): m(DetailBatch(h,x,torch.zeros(2,192)))


if __name__=='__main__': unittest.main()
