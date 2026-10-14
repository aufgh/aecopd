import unittest
from unittest.mock import patch
import numpy as np
from training.v6.resp_fact_set_experiment import lock,rebuild_data,SEEDS,ARMS


def report(scores):
    m=lambda v:{'readmission':{'integrated_brier_7_365':v}}
    return dict(fits={str(s):{a:{'metrics':m(scores[a])} for a in ARMS} for s in SEEDS},
                references={str(s):m(scores['original_V6']) for s in SEEDS})


class Tests(unittest.TestCase):
    def test_new_interface_four_return_values(self):
        arrays={'values':np.array([1.0])}
        with patch('training.v6.resp_fact_set_experiment.derive',return_value=(arrays,[],[],{})):
            rebuild_data(arrays)

    def test_tie_has_no_nomination(self):
        self.assertEqual(lock(report(dict(availability=.1,content=.1,original_V6=.1)))['nominated'],[])

    def test_availability_gain_not_content_gain(self):
        self.assertEqual(lock(report(dict(availability=.08,content=.09,original_V6=.1)))['nominated'],[])

    def test_content_must_beat_both(self):
        self.assertEqual(lock(report(dict(availability=.09,content=.08,original_V6=.1)))['nominated'],['content'])


if __name__=='__main__':unittest.main()
