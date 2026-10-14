import unittest
import torch
from torch import nn
from training.v6.frozen_augmentation import FrozenAugmentation,AugmentBatch,mask


class FrozenAugmentationTests(unittest.TestCase):
    def test_zero_start_bound_and_frozen_shape(self):
        m=FrozenAugmentation(nn.Linear(6,2))
        b=AugmentBatch(torch.zeros(3,2),torch.ones(3,2),torch.ones(3,283))
        out=m(torch.zeros(3,34),b)
        torch.testing.assert_close(out.log_amplitude,b.amplitude)
        self.assertTrue(torch.equal(out.shape,b.shape)); self.assertTrue(all(not p.requires_grad for p in m.baseline.parameters()))
        with torch.no_grad(): m.correction.weight.fill_(1.)
        torch.testing.assert_close(m(torch.zeros(3,34),b).delta,torch.full((3,2),.35))

    def test_group_mask_and_no_input_gradient(self):
        b=AugmentBatch(torch.zeros(2,2),torch.ones(2,2),torch.ones(2,283))
        cache=[(None,torch.zeros(2,34),b,None,None)]
        self.assertEqual(int(mask(cache,'constant')[0][2].features.sum()),2)
        self.assertEqual(int(mask(cache,'new_labs')[0][2].features.sum()),182)
        self.assertEqual(int(b.features.sum()),566)


if __name__=='__main__': unittest.main()
