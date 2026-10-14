import unittest
import numpy as np
import torch
from training.v6.model import HistoryBaseline
from training.v6.predict_research import scores


class ResearchPredictionTests(unittest.TestCase):
    def test_no_future_label_dependency(self):
        model=HistoryBaseline(); history=torch.ones(3,6)
        a,k=scores(model,[(np.arange(3),history,None,torch.ones(3),torch.ones(3,dtype=torch.long))])
        b,l=scores(model,[(np.arange(3),history,None,torch.ones(3)*365,torch.zeros(3,dtype=torch.long))])
        self.assertTrue(np.array_equal(a,b)); self.assertTrue(np.array_equal(k,l))


if __name__=='__main__': unittest.main()
