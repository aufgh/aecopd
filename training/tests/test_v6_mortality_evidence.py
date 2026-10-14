import unittest
import numpy as np
from training.v6.task_mortality_evidence import death_pair


class Tests(unittest.TestCase):
    def test_event_swap_is_only_evaluation_and_preserves_competing_controls(self):
        n=60;events=np.array([2]*20+[1]*20+[0]*20);original=events.copy()
        times=np.array([30.]*40+[365.]*20);risk=np.tile(np.linspace(0,.2,366),(n,1))
        p=death_pair(np.arange(n),times,events,risk,risk,seed=42)
        self.assertEqual(p['integrated_brier_difference'],0.);self.assertEqual(p['auc90_difference'],0.)
        np.testing.assert_array_equal(events,original)


if __name__=='__main__':unittest.main()
