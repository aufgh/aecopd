import tempfile
from pathlib import Path
import unittest
import torch
from training.v6.contextual_residual_experiment import clinical_nomination
from training.v6.task_optimization_audit import load


class Tests(unittest.TestCase):
    def test_recalibration_winner_does_not_become_clinical_candidate(self):
        selected,names=clinical_nomination(dict(control=.10,baseline_only=.09,context_concat=.095))
        self.assertEqual(selected,'baseline_only');self.assertEqual(names,[])

    def test_clinical_candidate_must_beat_both_comparators(self):
        selected,names=clinical_nomination(dict(control=.10,baseline_only=.09,context_film=.085))
        self.assertEqual(selected,'context_film');self.assertEqual(names,['context_film'])
        self.assertEqual(clinical_nomination(dict(control=.10,baseline_only=.11,context_film=.10)),('control',[]))

    def test_load_uses_explicit_root_without_global_mutation(self):
        from training.v6.task_optimization_audit import ROOT
        from training.v6.contextual_residual import TRIALS
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            with self.assertRaises(FileNotFoundError) as error:load(None,42,TRIALS[0],None,root=path)
            self.assertIn(str(path),str(error.exception))
        from training.v6.task_optimization_audit import ROOT as after
        self.assertEqual(ROOT,after)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU not available')
    def test_real_100_epoch_toy_fits_reload_with_injected_factory(self):
        # 真实100轮小样本工程测试，不能冒称MIMIC队列训练或性能证据。
        import dataclasses
        import numpy as np
        from training.tests.test_v6_deep_interaction import example
        from training.v6.contextual_residual import TRIALS,build
        from training.v6.model import HistoryBaseline
        from training.v6.regularization_multiseed import fit
        from training.v6.module_surgery_experiment import validation
        def cuda(x):
            if isinstance(x,torch.Tensor):return x.cuda()
            if dataclasses.is_dataclass(x):return type(x)(**{f.name:cuda(getattr(x,f.name)) for f in dataclasses.fields(x)})
            return x
        torch.set_num_threads(1)
        b=cuda(example());h=torch.randn(3,6,device='cuda');t=torch.tensor([20.,50.,365.],device='cuda');e=torch.tensor([1,2,0],device='cuda')
        cache=dict(train=[(None,h,b,t,e)],tune=[(None,h,b,t,e)])
        factory=lambda base,seed,trial,bins:build(base,seed,trial,bins,text_width=8)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for trial in (TRIALS[1],TRIALS[3]):
                base=HistoryBaseline().cuda();path=root/'seed42'/trial.name
                info=fit(base,42,trial,[[-1,1]]*132,cache,t.cpu().numpy(),e.cpu().numpy(),path,model_factory=factory)
                m,reloaded,best=load(base,42,trial,[[-1,1]]*132,root=root,model_factory=factory)
                self.assertEqual(info,reloaded);self.assertEqual(info['completed_epochs'],100)
                actual=validation(m,cache['tune'],t.cpu().numpy(),e.cpu().numpy())
                self.assertAlmostEqual(actual['ibs'],best['tuning_ibs'],places=10)
                self.assertTrue(np.isfinite(actual['nll']))


if __name__=='__main__':unittest.main()
