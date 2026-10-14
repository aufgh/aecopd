import unittest
import torch
from training.tests.test_v6_deep_interaction import example
from training.v6.model import HistoryBaseline
from training.v6.contextual_residual import TRIALS,build
from training.v6.risk import competing_nll


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_same_encoder_and_zero_initial_risk(self):
        base=HistoryBaseline();h=torch.randn(3,6);b=example()
        control=build(base,42,TRIALS[0],[[-1,1]]*132,text_width=8,device='cpu')
        for trial in TRIALS:
            m=build(base,42,trial,[[-1,1]]*132,text_width=8,device='cpu').eval()
            for key,value in control.encoder.state_dict().items():
                torch.testing.assert_close(value,m.encoder.state_dict()[key],atol=0,rtol=0)
            actual,expected=m(h,b),m.baseline(h)
            torch.testing.assert_close(actual.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(actual.shape,expected.shape,atol=0,rtol=0)

    def test_baseline_only_has_no_clinical_gradient(self):
        m=build(HistoryBaseline(),42,TRIALS[1],[[-1,1]]*132,text_width=8,device='cpu')
        torch.nn.init.normal_(m.delta_head[-1].weight,std=.01)
        out=m(torch.randn(3,6),example());competing_nll(out,torch.tensor([20.,50.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))

    def test_context_routes_backpropagate_to_clinical_not_frozen_base(self):
        for trial in TRIALS[2:]:
            m=build(HistoryBaseline(),42,trial,[[-1,1]]*132,text_width=8,device='cpu')
            torch.nn.init.normal_(m.delta_head[-1].weight,std=.01)
            out=m(torch.randn(3,6),example());competing_nll(out,torch.tensor([20.,50.,365.]),torch.tensor([1,2,0])).backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None),0.)
            self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))

    def test_context_changes_correction_not_just_base_prediction(self):
        base=HistoryBaseline()
        with torch.no_grad():base.amplitude.weight.fill_(.1)
        b=example();h=torch.zeros(3,6)
        for trial in TRIALS[1:]:
            m=build(base,42,trial,[[-1,1]]*132,text_width=8,device='cpu').eval()
            torch.nn.init.normal_(m.delta_head[-1].weight,std=.03)
            if trial.mode=='film':torch.nn.init.normal_(m.modulation[-1].weight,std=.03)
            left=m(h,b).delta;right=m(h+1,b).delta
            self.assertGreater(float((left-right).detach().abs().max()),1e-7)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU not available')
    def test_gpu_fit_keeps_baseline_and_starts_at_base(self):
        # 真实GPU小样本六轮入口核验，不是队列性能实验。
        from pathlib import Path
        import tempfile
        from training.v6.regularization_multiseed import fit
        import training.v6.regularization_multiseed as engine
        b=example()
        # 示例对象为嵌套batch，完整搬移每个张量。
        import dataclasses
        def cuda(x):
            if isinstance(x,torch.Tensor):return x.cuda()
            if dataclasses.is_dataclass(x):return type(x)(**{f.name:cuda(getattr(x,f.name)) for f in dataclasses.fields(x)})
            return x
        b=cuda(b);h=torch.randn(3,6,device='cuda');t=torch.tensor([20.,50.,365.],device='cuda');e=torch.tensor([1,2,0],device='cuda')
        cache=dict(train=[(None,h,b,t,e)],tune=[(None,h,b,t,e)])
        old=engine.EPOCHS;engine.EPOCHS=6
        try:
            with tempfile.TemporaryDirectory() as folder:
                info=fit(HistoryBaseline().cuda(),42,TRIALS[3],[[-1,1]]*132,cache,t.cpu().numpy(),e.cpu().numpy(),
                         Path(folder),model_factory=lambda base,seed,trial,bins:build(base,seed,trial,bins,text_width=8))
                self.assertTrue(info['frozen_verified']);self.assertEqual(info['completed_epochs'],6)
        finally:engine.EPOCHS=old


if __name__=='__main__':unittest.main()
