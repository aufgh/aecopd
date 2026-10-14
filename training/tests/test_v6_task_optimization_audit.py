import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from training.v6.task_optimization_audit import check_curve,check_saved
from training.v6.module_surgery_experiment import save


class Tests(unittest.TestCase):
    def example(self):
        days=np.arange(366)[None,:,None]
        survival=np.exp(-days*.002)
        curve=np.concatenate([survival,(1-survival)*.7,(1-survival)*.3],2)
        curve=np.repeat(curve,2,0)
        d=SimpleNamespace(arrays=dict(hadm_ids=np.array([11,12]),subject_ids=np.array([8,8])),
                          times=np.array([21.,365.]),events=np.array([1,0]))
        return d,np.arange(2),np.zeros((2,2)),np.ones((2,2)),curve

    def test_saved_curve_daily_and_label_identity(self):
        d,ix,a,k,curve=self.example()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'predictions.npz';save(p,d,ix,a,k,curve)
            self.assertEqual(check_saved(p,d,ix,a,k,curve),0.)
            d.events[0]=2
            with self.assertRaises(AssertionError):check_saved(p,d,ix,a,k,curve)

    def test_probability_contract(self):
        *_,curve=self.example();check_curve(curve)
        broken=curve.copy();broken[0,0,0]=.5
        with self.assertRaises(AssertionError):check_curve(broken)
        broken=curve.copy();broken[0,100]=curve[0,50]
        with self.assertRaises(ValueError):check_curve(broken)

    def test_early_censoring_cannot_be_relabelled_by_save(self):
        # 存储会忠实保留时间/结局；模型评价另由WorldA校验阻断提前删失。
        from training.v6.evaluation import validate
        d,ix,a,k,curve=self.example();d.times[1]=60.
        with self.assertRaises(ValueError):validate(d.times,d.events,curve[:,:,1])


if __name__=='__main__':unittest.main()
