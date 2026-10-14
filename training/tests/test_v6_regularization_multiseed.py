"""五种子合同和GPU小样本训练检查，不作为临床性能证据。"""
from dataclasses import fields,is_dataclass,replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from training.tests.test_v6_deep_interaction import example
from training.v6.model import HistoryBaseline
from training.v6.regularization_model import Trial
from training.v6.regularization_multiseed import SEEDS,BASE_SEEDS,fit,build


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_five_clinical_seeds_and_explicit_foundation_mapping(self):
        self.assertEqual(len(SEEDS),5);self.assertEqual(len(set(SEEDS)),5)
        self.assertEqual(set(SEEDS),set(BASE_SEEDS))
        self.assertEqual(set(BASE_SEEDS.values()),{42,7})

    def test_gpu_two_epoch_fit_and_frozen_source(self):
        if not torch.cuda.is_available():self.skipTest('GPU unavailable')
        def cuda(x):
            if isinstance(x,torch.Tensor):return x.cuda()
            if is_dataclass(x):return replace(x,**{f.name:cuda(getattr(x,f.name)) for f in fields(x)})
            return x
        b=example();c=b.base.base.clinical
        c=replace(c,snippet_vectors=c.snippet_vectors.repeat(1,96))
        b=cuda(replace(b,base=replace(b.base,base=replace(b.base.base,clinical=c))))
        base=HistoryBaseline().cuda();h=torch.randn(3,6,device='cuda')
        times=torch.tensor([30.,90.,365.],device='cuda');events=torch.tensor([1,2,0],device='cuda')
        batch=(np.arange(3),h,b,times,events)
        trial=Trial('v6_control');bins=[[-1.,1.]]*132
        before={k:v.detach().clone() for k,v in base.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp,patch('training.v6.regularization_multiseed.EPOCHS',2):
            path=Path(tmp)/'trial'
            info=fit(base,42,trial,bins,dict(train=[batch],tune=[batch]),times.cpu().numpy(),events.cpu().numpy(),path)
            self.assertEqual(info['completed_epochs'],2)
            rows=json.loads((path/'history.json').read_text())
            self.assertEqual(len(rows),3)
            self.assertIsNone(rows[0]['tuning_auc90'])
            restored=build(base,42,trial,bins)
            restored.load_state_dict(torch.load(path/'best.pt',map_location='cuda',weights_only=True)['state_dict'])
            for k,v in before.items():torch.testing.assert_close(v,restored.baseline.state_dict()[k],atol=0,rtol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in restored.baseline.parameters()))


if __name__=='__main__':unittest.main()
