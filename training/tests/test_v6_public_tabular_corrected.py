"""核对完整官方归一化/激活/线性读出，不被零初始化头掩盖。"""
import copy
import unittest
import torch
from torch import nn
import numpy as np
from training.v6.public_tabular_corrected import CorrectedModel
from training.v6.public_tabular_model import PublicModel


class CorrectedReadoutTest(unittest.TestCase):
    def test_unaffected_models_identical_state_and_predictions(self):
        for arm in ('mlp','tabm_ple'):
            args=(arm,3,2,[torch.tensor([-2.,0.,2.])]*3,np.zeros(4,np.float32))
            torch.manual_seed(3);old=PublicModel(*args).eval()
            torch.manual_seed(3);new=CorrectedModel(*args).eval()
            self.assertEqual(list(old.state_dict()),list(new.state_dict()))
            for key,value in old.state_dict().items():self.assertTrue(torch.equal(value,new.state_dict()[key]))
            x=torch.randn(7,3);b=torch.randint(0,2,(7,2)).float()
            a,k=old(x,b);aa,kk=new(x,b);torch.testing.assert_close(a,aa,rtol=0,atol=0);torch.testing.assert_close(k,kk,rtol=0,atol=0)

    def test_readout_matches_full_official_output(self):
        torch.set_num_threads(2)
        for arm in ('resnet_ple','resnet_periodic','ft_transformer'):
            m=CorrectedModel(arm,3,2,[torch.tensor([-2.,0.,2.])]*3,np.zeros(4,np.float32)).eval()
            with torch.no_grad():m.head.weight.normal_(0,.01)
            ref=copy.deepcopy(m.backbone)
            output=nn.Sequential(copy.deepcopy(m.readout[0]),nn.ReLU(),copy.deepcopy(m.head))
            if arm.startswith('resnet'):ref.output=output
            else:ref.backbone.output=output
            x=torch.randn(7,3);b=torch.randint(0,2,(7,2)).float()
            expected=(ref(torch.cat((m.embedding(x).flatten(1),b),1)) if arm.startswith('resnet') else ref(x,b.long()))
            a,k=m(x,b)
            torch.testing.assert_close(a[:,0],expected[:,:2].clamp(-10,6))
            torch.testing.assert_close(k[:,0],torch.nn.functional.softplus(expected[:,2:])+.1)
            loss=(a.square().mean()+k.square().mean());loss.backward()
            self.assertTrue(torch.isfinite(m.readout[0].weight.grad).all())


if __name__=='__main__':unittest.main()
