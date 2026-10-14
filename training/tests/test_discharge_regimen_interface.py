import copy
import unittest
import numpy as np
from patient_data.discharge_regimen_interface import collect,encode,role


def record(key='a',product='tiotropium',action=None,status='listed'):
    return dict(hadm_id='1',entry_key=key,quote='example',reading=dict(canonical_identity=product,
        product_name='Spiriva',classes=['lama'],status=status,prefix_action=action,as_needed=False))


class RegimenInterfaceTests(unittest.TestCase):
    def base(self):
        return dict(hadm_ids=np.array([1,2]),subject_ids=np.array([1,2]),patient_splits=np.array(['train','test']))

    def test_hold_not_stop(self):
        self.assertEqual(role(record(action='held',status='explicit_stop')['reading']),'held')
        self.assertEqual(role(record(action='discontinued',status='explicit_stop')['reading']),'stopped')

    def test_unknown_action_rejected(self):
        with self.assertRaises(ValueError):role(record(action='maybe',status='explicit_stop')['reading'])

    def test_dedup_retains_references(self):
        a,b=record(),record('b');members=collect([a,b,a],{'1':0})
        self.assertEqual(len(members),1);self.assertEqual(len(members[0]['evidence']),2)

    def test_permutation(self):
        a,b=record(),record('b',action='held',status='explicit_stop')
        self.assertEqual(collect([a,b],{'1':0}),collect([b,a],{'1':0}))

    def test_joint_product_keeps_multiclass(self):
        a=record(product='umeclidinium_vilanterol');a['reading']['classes']=['laba','lama']
        m=collect([a],{'1':0});self.assertEqual(len(m),1);self.assertEqual(len(m[0]['classes']),2)

    def test_nontraining_vocabulary_unknown(self):
        a,b=record(),record('b','new_identity');b['hadm_id']='2'
        m=collect([a,b],{'1':0,'2':1});arr,meta=encode(m,self.base(),np.ones(2,bool),np.ones(2,bool))
        self.assertNotIn('new_identity',meta['product_vocabulary']);self.assertEqual(arr['products'][1],0)

    def test_source_absent_not_no_mention(self):
        arr,_=encode([],self.base(),np.array([True,False]),np.zeros(2,bool))
        self.assertEqual(arr['features'].sum(),0);self.assertNotEqual(*arr['section_available'])

    def test_evidence_frequency_not_feature(self):
        a,b=record(),record('b');x,_=encode(collect([a],{'1':0}),self.base(),np.ones(2,bool),np.array([True,False]))
        y,_=encode(collect([a,b],{'1':0}),self.base(),np.ones(2,bool),np.array([True,False]))
        np.testing.assert_array_equal(x['features'],y['features'])

    def test_unresolved_not_listed(self):
        a=record(status='unresolved');a['reading']['classes']=[]
        m=collect([a],{'1':0});x,_=encode(m,self.base(),np.ones(2,bool),np.array([True,False]))
        self.assertEqual(x['features'][0,:15].sum(),0);self.assertEqual(x['features'][0,19],1)

    def test_invalid_source_relation(self):
        with self.assertRaises(ValueError):encode([],self.base(),np.zeros(2,bool),np.ones(2,bool))


if __name__=='__main__':unittest.main()
