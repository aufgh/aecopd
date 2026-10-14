import unittest
import numpy as np
import re
from training.v6.data_geometry_metrics import cosine_pairs
from training.v6.discharge_template_audit import PATTERNS


class Tests(unittest.TestCase):
    def test_independent_is_not_dependent(self):
        p=PATTERNS['dependence_or_limitation']
        self.assertIsNone(re.search(p,'Ambulatory - Independent',re.I))
        self.assertIsNotNone(re.search(p,'Dependent on assistance and walker',re.I))

    def test_cosine_is_scale_invariant(self):
        x=np.random.default_rng(4).normal(size=(60,10));ids=np.arange(60)
        a=cosine_pairs(x,ids);b=cosine_pairs(x*1.001,ids)
        np.testing.assert_allclose(a['mean'],b['mean'],atol=1e-12)
        np.testing.assert_allclose(a['quantiles'],b['quantiles'],atol=1e-12)


if __name__=='__main__':unittest.main()
