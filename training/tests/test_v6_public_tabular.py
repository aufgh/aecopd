"""合成数据测试，不接触患者数据；使用隔离官方依赖运行。"""
import unittest
import copy
import numpy as np
import torch
from training.v6.public_tabular_data import fit_preprocessing,transform
from training.v6.public_tabular_model import ARMS,PublicModel,initial_bias,ensemble_nll,ensemble_curve
from training.v6.token_model import daily_probabilities


class PublicTabularTest(unittest.TestCase):
    def test_train_only_preprocessing_and_missing(self):
        x=np.array([[0,0,7],[2,1,7],[1000,1,7]],np.float32)
        valid=np.ones_like(x,bool);train=np.array([1,1,0],bool)
        p=fit_preprocessing(x,valid,train);xn,xb=transform(x,valid,p)
        self.assertEqual(p['center'][0],1)
        self.assertEqual(p['num_columns'].tolist(),[0]);self.assertEqual(p['binary_columns'].tolist(),[1])
        self.assertTrue(np.isin(xb,(0,1)).all());self.assertEqual(xn[-1,0],8)
        valid[0,0]=False;xn,_=transform(x,valid,p);self.assertEqual(xn[0,0],0)

    def test_all_backbones_joint_head_and_gradients(self):
        torch.set_num_threads(2)
        x=torch.randn(9,3);b=torch.randint(0,2,(9,2)).float();e=torch.tensor([0,1,2]*3);t=torch.tensor([365.,20.,90.]*3)
        bias=initial_bias(e.numpy());initial=[]
        for arm in ARMS:
            m=PublicModel(arm,3,2,[torch.tensor([-2.,0.,2.])]*3,bias)
            a,k=m(x,b);self.assertEqual(a.shape,(9,m.k,2));initial.append((a[:,0].detach(),k[:,0].detach()))
            loss=ensemble_nll(a,k,t,e);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(m.head.weight.grad.abs().sum()>0)
            curve=ensemble_curve(a.detach().numpy(),k.detach().numpy(),device='cpu')
            daily_probabilities(curve);self.assertTrue(np.all(np.diff(curve[:,:,1],axis=1)>=-1e-9))
        for a,k in initial:self.assertTrue(torch.equal(a,initial[0][0]) and torch.equal(k,initial[0][1]))

    @unittest.skipUnless(torch.cuda.is_available(),'GPU mixed precision test')
    def test_initial_gpu_mixed_precision_equal(self):
        initial=[]
        for arm in ARMS:
            m=PublicModel(arm,3,2,[torch.tensor([-2.,0.,2.])]*3,np.zeros(4,np.float32)).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,k=m(torch.randn(8,3,device='cuda'),torch.zeros(8,2,device='cuda'))
            initial.append((a[:,0],k[:,0]))
        for a,k in initial:
            self.assertTrue(torch.equal(a,initial[0][0]) and torch.equal(k,initial[0][1]))

    def test_sdpa_matches_official_forward_and_backward(self):
        torch.manual_seed(4);bins=[torch.tensor([-2.,0.,2.])]*3;bias=np.zeros(4,np.float32)
        original=PublicModel('ft_transformer',3,2,bins,bias,sdpa=False).double().eval()
        accelerated=PublicModel('ft_transformer',3,2,bins,bias).double().eval()
        accelerated.load_state_dict(original.state_dict())
        x=torch.randn(5,3,dtype=torch.float64);b=torch.randint(0,2,(5,2))
        # 零输出头会掩盖内部错误；直接核对骨干和梯度。
        y=original.backbone(x,b);yy=accelerated.backbone(x,b)
        torch.testing.assert_close(y,yy,rtol=1e-10,atol=1e-10)
        y.square().sum().backward();yy.square().sum().backward()
        for (name,p),(other,q) in zip(original.backbone.named_parameters(),accelerated.backbone.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-8)

    def test_ensemble_averages_probabilities_not_parameters(self):
        a=np.array([[[-3.,-4.],[1.,-2.]]]);k=np.array([[[.5,1.],[2.,.8]]])
        curve=ensemble_curve(a,k,device='cpu')
        separate=[ensemble_curve(a[:,i:i+1],k[:,i:i+1],device='cpu') for i in range(2)]
        np.testing.assert_allclose(curve,(separate[0]+separate[1])/2,atol=1e-12)
        wrong=ensemble_curve(a.mean(1,keepdims=True),k.mean(1,keepdims=True),device='cpu')
        self.assertGreater(np.abs(curve-wrong).max(),.01)


if __name__=='__main__':unittest.main()
