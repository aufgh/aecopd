import unittest
import torch
from training.v6.task_gradient_diagnostic import cause_losses,gradient_summary
from training.v6.risk import RiskOutput,competing_nll


class Tests(unittest.TestCase):
    def test_cause_terms_equal_joint_loss_and_gradient(self):
        a=torch.randn(8,2,requires_grad=True);k=torch.rand(8,2,requires_grad=True)+.4
        out=RiskOutput(a,k);t=torch.tensor([0.,1.,5.,30.,90.,180.,300.,365.]);e=torch.tensor([1,2,0,1,2,1,2,0])
        parts=cause_losses(out,t,e);joint=competing_nll(out,t,e)
        torch.testing.assert_close(parts.sum(),joint,atol=1e-6,rtol=1e-6)
        left=torch.autograd.grad(parts.sum(),a,retain_graph=True)[0]
        right=torch.autograd.grad(joint,a)[0]
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)

    def test_missing_and_opposed_gradients(self):
        a=[torch.tensor([1.,0.]),None];b=[torch.tensor([-1.,0.]),torch.tensor([0.])]
        self.assertEqual(gradient_summary(a,b)['cosine'],-1.)
        self.assertIsNone(gradient_summary([None],[None])['cosine'])


if __name__=='__main__':unittest.main()
