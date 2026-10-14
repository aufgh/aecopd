import unittest
import torch
from training.v6.clinical_history_model import HistoryReader,ARMS


class Tests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(3)
        return (torch.randn(3,128),torch.randn(3,4,128),torch.tensor([[3.,8.,8.,0.],[0.,0.,0.,0.],[2.,9.,1.,0.]]),
            torch.tensor([[1,1,1,0],[0,0,0,0],[1,1,1,0]],dtype=torch.bool),torch.randn(3,2),torch.rand(3,2)+.5)

    def test_initial_identity_shared_parameters_and_empty_history(self):
        x=self.inputs();states=[]
        for arm in ARMS:
            torch.manual_seed(42);m=HistoryReader(arm).eval();states.append(m.state_dict())
            out=m(*x)
            torch.testing.assert_close(out.log_amplitude,x[-2],atol=0,rtol=0)
            torch.testing.assert_close(out.shape,x[-1],atol=0,rtol=0)
            with torch.no_grad():m.head[-1].weight.fill_(.1)
            out=m(*x)
            torch.testing.assert_close(out.log_amplitude[1],x[-2][1],atol=0,rtol=0)
        for state in states[1:]:
            for key,value in states[0].items():torch.testing.assert_close(value,state[key],atol=0,rtol=0)

    def test_joint_permutation_and_padding_invariance(self):
        x=self.inputs();perm=torch.tensor([2,0,3,1])
        for arm in ARMS:
            m=HistoryReader(arm).eval()
            with torch.no_grad():m.head[-1].weight.normal_(0,.1)
            out=m(*x)
            shuffled=m(x[0],x[1][:,perm],x[2][:,perm],x[3][:,perm],x[4],x[5])
            torch.testing.assert_close(out.log_amplitude,shuffled.log_amplitude,atol=1e-6,rtol=1e-6)
            poisoned=x[1].clone();poisoned[~x[3]]=999.
            other=m(x[0],poisoned,x[2],x[3],x[4],x[5])
            torch.testing.assert_close(out.log_amplitude,other.log_amplitude,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
