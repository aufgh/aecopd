import unittest
from patient_data.v6_detail_extension import dose_mg,molecule_matches,scale
import numpy as np


class DetailTests(unittest.TestCase):
    def row(self,ordinal,dose,unit='mg',product='prednisone'):
        return dict(parent_field_ordinal=ordinal,dose_given=str(dose),dose_given_unit=unit,product_description=product)

    def test_parent_not_added_and_duplicate_ordinal_not_recounted(self):
        x=[self.row(r'\N',40),self.row('1.1',20),self.row('1.2',20),self.row('1.2',20)]
        self.assertEqual(dose_mg(x,'prednisone'),(40.,'verified_component_sum'))

    def test_conflicting_or_unknown_dose(self):
        self.assertIsNone(dose_mg([self.row('1.1',20),self.row('1.1',30)],'prednisone')[0])
        self.assertIsNone(dose_mg([self.row('1.1',2,'NEB')],'prednisone')[0])
        self.assertIsNone(dose_mg([self.row('1.1',2,product='hydrocortisone')],'prednisone')[0])

    def test_units_and_combination(self):
        self.assertEqual(dose_mg([self.row('1.1',.02,'gm')],'prednisone')[0],20.)
        self.assertEqual(set(molecule_matches('albuterol / ipratropium')),{'albuterol','ipratropium'})
        self.assertEqual(molecule_matches('levalbuterol'),())

    def test_training_only_scale_zero_is_measured(self):
        v=np.array([[[0.]],[[2.]],[[1000.]]]); masks=np.ones_like(v,dtype=bool)
        a,p=scale(v,masks,np.array([True,True,False]))
        self.assertTrue(masks[0,0,0]); self.assertLess(p['0/0']['median'],1)
        self.assertEqual(a[2,0,0],8.)


if __name__=='__main__': unittest.main()
