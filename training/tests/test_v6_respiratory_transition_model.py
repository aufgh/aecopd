import unittest
import torch
from training.v6.curve_adaptation import FrozenBatch
from training.v6.respiratory_transition_model import RespiratoryCorrection,RespBatch,view,AVAIL,TIMING


class Tests(unittest.TestCase):
    def batch(self,observed=True):
        torch.manual_seed(9);n=8
        f=FrozenBatch(torch.randn(n,32),torch.randn(n,4)*.1,torch.randn(n,2)-2,torch.tensor([.3,.5]),
            torch.arange(1,n+1,dtype=torch.float32),torch.tensor([0,1,2,0,1,2,0,1]))
        return RespBatch(f,torch.randn(n,96),torch.full((n,),observed))

    def test_zero_initial_risks_are_identical(self):
        b=self.batch();outputs=[RespiratoryCorrection(a)(b) for a in ('global','availability','timing','content')]
        for out in outputs[1:]:
            torch.testing.assert_close(out.log_amplitude,outputs[0].log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,outputs[0].shape,atol=0,rtol=0)

    def test_three_content_models_have_equal_weights_and_capacity(self):
        models=[RespiratoryCorrection(a) for a in ('availability','timing','content')]
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),2132)
        for m in models[1:]:
            for k,v in models[0].state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]))

    def test_disallowed_values_are_really_hidden(self):
        b=self.batch();x=b.features.clone();x[:,0]+=100;x[:,2]+=100;x[:,20]+=100
        self.assertTrue(torch.equal(view(x,'availability'),view(b.features,'availability')))
        self.assertTrue(torch.equal(view(x,'timing'),view(b.features,'timing')))
        x[:,6]+=100
        self.assertTrue(torch.equal(view(x,'availability'),view(b.features,'availability')))
        self.assertFalse(torch.equal(view(x,'timing'),view(b.features,'timing')))

    def test_absent_source_stays_original_after_weights_change(self):
        b=self.batch(False);m=RespiratoryCorrection('content');original=m(b)
        with torch.no_grad():m.head[-1].weight.fill_(.5);m.head[-1].bias.fill_(1.)
        changed=m(b)
        torch.testing.assert_close(original.log_amplitude,changed.log_amplitude,atol=0,rtol=0)
        torch.testing.assert_close(original.shape,changed.shape,atol=0,rtol=0)
        self.assertTrue(torch.equal(changed.delta,torch.zeros_like(changed.delta)))


if __name__=='__main__':unittest.main()
