"""时钟读出精度及保存接口回归；无真实病历。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from training.v6.clock_audit import refined_teacher,save_or_check
from training.v6.anchored_clock import clock_curve
from training.v6.risk import predict_cif


class Tests(unittest.TestCase):
    def test_refined_early_grid_preserves_identity_and_converges(self):
        # 复现低shape高早期事件概率的数值边界，不是患者记录。
        a=torch.tensor([[.09016468,2.26372838]])
        k=torch.tensor([[.69631469,.62271988]])
        cache=dict(amplitude=a,shape=k)
        days=np.arange(366)
        daily=predict_cif(a.numpy(),k.numpy(),days,step_days=.5)
        grid,dense=refined_teacher(cache,daily)
        fine_grid,fine=refined_teacher(cache,daily,fine=True)
        unchanged=clock_curve(dense,grid,np.zeros((1,2)),days)
        np.testing.assert_allclose(unchanged,daily,atol=1e-12,rtol=0)
        clocks=np.array([[.066802,.228160]])
        curve=clock_curve(dense,grid,clocks,days)
        refined=clock_curve(fine,fine_grid,clocks,days)
        self.assertLess(float(np.abs(curve-refined).max()),1e-4)
        np.testing.assert_array_equal(curve[:,-1],daily[:,-1])

    def test_clock_is_saved_and_replayed_separately_from_teacher_parameters(self):
        cache=dict(amplitude=torch.zeros(2,2),shape=torch.ones(2,2))
        curve=predict_cif(np.zeros((2,2)),np.ones((2,2)),np.arange(366))
        d=SimpleNamespace(arrays=dict(hadm_ids=np.array([11,12]),subject_ids=np.array([1,2])),
            times=np.array([90.,365.]),events=np.array([1,0]))
        ix=np.arange(2);clock=np.zeros((2,2),dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'prediction.npz'
            self.assertIsNone(save_or_check(path,d,ix,cache,clock,curve))
            self.assertEqual(save_or_check(path,d,ix,cache,clock,curve),0.)
            with self.assertRaises(AssertionError):save_or_check(path,d,ix,cache,clock+.1,curve)


if __name__=='__main__':unittest.main()
