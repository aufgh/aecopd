import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch
from training.v6.local_text_model import LocalTextBatch,LocalTextCorrection,ARMS


class Tests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(8);n=4
        f=SimpleNamespace(hidden=torch.randn(n,32),raw=torch.randn(n,4)*.1,prior=torch.randn(n,2)-2,shape_raw=torch.zeros(2))
        vectors=torch.randn(5,768)
        return LocalTextBatch(f,vectors,torch.randn(5,10),torch.tensor([0,0,1,2,2]),torch.randn(n,11),torch.tensor([1,1,1,0],dtype=torch.bool),vectors.clone())

    def test_common_initial_risk_and_weights(self):
        b=self.batch();models=[LocalTextCorrection(a) for a in ARMS];reference=models[0](b)
        for m in models[1:]:
            for k,v in models[0].state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
            torch.testing.assert_close(m(b).shape,reference.shape,atol=0,rtol=0)
            torch.testing.assert_close(m(b).log_amplitude,reference.log_amplitude,atol=0,rtol=0)

    def test_availability_never_sees_vector_content(self):
        b=self.batch();m=LocalTextCorrection('availability')
        with torch.no_grad():m.head[-1].weight.fill_(.1)
        torch.testing.assert_close(m(b).log_amplitude,m(replace(b,vectors=b.vectors*100)).log_amplitude,atol=0,rtol=0)

    def test_empty_source_exactly_original_after_training(self):
        b=self.batch();m=LocalTextCorrection('local');before=m(b)
        with torch.no_grad():m.head[-1].weight.fill_(1);m.head[-1].bias.fill_(1)
        out=m(b);self.assertTrue(torch.equal(before.shape[-1],out.shape[-1]));self.assertTrue(torch.equal(before.log_amplitude[-1],out.log_amplitude[-1]))

    def test_same_time_tokens_order_insensitive(self):
        b=self.batch();m=LocalTextCorrection('local')
        with torch.no_grad():m.head[-1].weight.fill_(.1)
        order=torch.tensor([4,1,3,0,2]);other=replace(b,vectors=b.vectors[order],local_vectors=b.local_vectors[order],metadata=b.metadata[order],owners=b.owners[order])
        torch.testing.assert_close(m(b).log_amplitude,m(other).log_amplitude,atol=1e-6,rtol=0)

    def test_no_sentence_batch_finite(self):
        b=self.batch();empty=replace(b,vectors=b.vectors[:0],metadata=b.metadata[:0],owners=b.owners[:0],observed=torch.zeros(4,dtype=torch.bool))
        self.assertTrue(torch.isfinite(LocalTextCorrection('local')(empty).shape).all())


if __name__=='__main__':unittest.main()
