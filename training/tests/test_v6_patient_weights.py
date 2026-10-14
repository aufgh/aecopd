import unittest
import numpy as np
import torch
from training.v6.patient_weight_loss import training_weights,per_record_nll,weighted_nll
from training.v6.risk import RiskOutput,competing_nll


class Tests(unittest.TestCase):
    def test_global_normalization_and_equal_patient_mass(self):
        people=np.array([1,1,1,2,3,3])
        np.testing.assert_array_equal(training_weights(people,0.),np.ones(6,np.float32))
        w=training_weights(people,1.)
        self.assertAlmostEqual(float(w.mean()),1.)
        np.testing.assert_allclose([w[people==p].sum() for p in (1,2,3)],2.,atol=1e-6)
        self.assertNotAlmostEqual(float(w[:3].mean()),1.)

    def test_identity_loss_and_parameter_gradients(self):
        torch.manual_seed(9)
        a=torch.randn(5,2,requires_grad=True);k=(torch.rand(5,2)+.3).requires_grad_()
        out=RiskOutput(a,k);t=torch.tensor([0.,15.,99.,365.,60.]);e=torch.tensor([1,2,0,0,1])
        original=competing_nll(out,t,e)
        derived=weighted_nll(out,t,e,torch.ones(5))
        torch.testing.assert_close(original,derived,atol=0,rtol=0)
        first=torch.autograd.grad(original,(a,k),retain_graph=True)
        second=torch.autograd.grad(derived,(a,k))
        for x,y in zip(first,second):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_vector_gradients_equal_sum_of_single_records(self):
        a=torch.tensor([[.1,-.5],[-1.,.2]],requires_grad=True);k=torch.ones(2,2,requires_grad=True)
        t=torch.tensor([10.,365.]);e=torch.tensor([1,0]);out=RiskOutput(a,k)
        gradients=torch.autograd.grad(per_record_nll(out,t,e).sum(),(a,k),retain_graph=True)
        for i in range(2):
            reference=competing_nll(RiskOutput(a[i:i+1],k[i:i+1]),t[i:i+1],e[i:i+1])
            single=torch.autograd.grad(reference,(a,k),retain_graph=True)
            for all_rows,one in zip(gradients,single):torch.testing.assert_close(all_rows[i],one[i],atol=0,rtol=0)

    def test_outer_product_recovers_actual_linear_gradient(self):
        torch.manual_seed(3);layer=torch.nn.Linear(32,4);hidden=torch.randn(8,32)
        raw=layer(hidden);raw.retain_grad();loss=raw.square().sum();loss.backward()
        augmented=torch.cat((hidden,torch.ones(8,1)),1)
        vectors=raw.grad[:,:,None]*augmented[:,None,:]
        torch.testing.assert_close(vectors.sum(0)[:,:32],layer.weight.grad,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(vectors.sum(0)[:,32],layer.bias.grad,atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
