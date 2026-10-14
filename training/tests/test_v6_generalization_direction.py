import unittest
import numpy as np
from training.v6.generalization_direction import parameter_group,gradient_direction,mean_gradient_sum,scopes


class Tests(unittest.TestCase):
    def test_direction_same_opposite_and_zero(self):
        x=np.array([1.,2.])
        self.assertAlmostEqual(gradient_direction(x,x)['cosine'],1.)
        self.assertAlmostEqual(gradient_direction(x,-x)['cosine'],-1.)
        self.assertIsNone(gradient_direction(x,np.zeros(2))['cosine'])
        with self.assertRaises(ValueError):gradient_direction(x,np.ones(3))

    def test_full_fold_normalization_not_equal_batch_weight(self):
        # 两个batch样本数3/1；sum/n应为2，不是batch均值再平均得到3。
        self.assertEqual(float(mean_gradient_sum([np.array([3.]),np.array([5.])],4)[0]),2.)
        with self.assertRaises(ValueError):mean_gradient_sum([],0)

    def test_verified_semantic_paths_and_unknown_rejection(self):
        pairs={'encoder.base.lab_ids.weight':'labs','encoder.base.branch_norms.2.weight':'medications',
               'encoder.base.microbiology.0.weight':'microbiology','encoder.base.text_projection.1.bias':'text',
               'encoder.base.fusion.0.weight':'fusion','encoder.extra_labs.0.weight':'additional_labs',
               'encoder.extra_meds.0.weight':'inactive_fine_medications','delta_head.2.weight':'risk_head'}
        for n,g in pairs.items():self.assertEqual(parameter_group(n),g)
        with self.assertRaises(ValueError):parameter_group('baseline.raw_shape')
        with self.assertRaises(ValueError):parameter_group('encoder.base.unknown.weight')

    def test_exclusive_groups_cover_encoder(self):
        layout=[dict(name='a',group='labs',start=0,end=3,last_projection=False),
                dict(name='b',group='text',start=3,end=6,last_projection=False),
                dict(name='c',group='risk_head',start=6,end=8,last_projection=True)]
        groups=scopes(layout)
        np.testing.assert_array_equal(groups['labs']|groups['text'],groups['encoder'])
        self.assertFalse((groups['labs']&groups['text']).any())
        self.assertEqual(int(groups['all_clinical'].sum()),8)
        np.testing.assert_array_equal(groups['last_projection'],groups['risk_head'])


if __name__=='__main__':unittest.main()
