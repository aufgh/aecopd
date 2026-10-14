import unittest
from datetime import datetime,timedelta
import json
import numpy as np
from patient_data.oxygen_device import label,compress,summarize,derive,LOOKUP,available


class Tests(unittest.TestCase):
    def setUp(self):self.t=datetime(2020,1,1);self.end=self.t+timedelta(days=8)

    def test_same_time_set_order_and_duplicate_invariant(self):
        a,b=label('Nasal cannula'),label('Bipap mask ')
        records=[(self.t,a),(self.t,b),(self.t,a)]
        self.assertEqual(compress(records),compress(records[::-1]))
        runs,_=compress(records);self.assertEqual(len(runs),1);self.assertEqual(runs[0][2],frozenset([a,b]))

    def test_terminal_equal_set_keeps_actual_last_time(self):
        a=label('Nasal cannula');runs,_=compress([(self.t,a),(self.t+timedelta(days=2),a)])
        self.assertEqual(len(runs),1);self.assertEqual(runs[0][1],self.t+timedelta(days=2))
        hot,flags,raw,mask,_=summarize({'x':[(self.t,a),(self.t+timedelta(days=2),a)]},self.t,self.end)
        self.assertAlmostEqual(np.expm1(raw[2]),6);self.assertAlmostEqual(np.expm1(raw[4]),2)
        self.assertEqual(hot[3:].sum(),0);self.assertEqual(flags[1],1)

    def test_cross_stay_not_a_transition(self):
        a,b=label('Endotracheal tube'),label('Nasal cannula')
        hot,flags,_,_,_=summarize({'x':[(self.t,a)],'y':[(self.t+timedelta(days=1),b)]},self.t,self.end)
        self.assertEqual(hot[3:].sum(),0);self.assertEqual(flags[3],1)

    def test_same_stay_change_is_not_device_intensity(self):
        a,b=label('Nasal cannula'),label('None');hot,_,raw,_,_=summarize({'x':[(self.t,a),(self.t+timedelta(days=1),b)]},self.t,self.end)
        self.assertEqual(hot[3,b],1);self.assertEqual(hot[4,a],1);self.assertAlmostEqual(np.expm1(raw[5]),1)
        self.assertNotEqual(a,b);self.assertEqual(label(' NONE '),b)

    def test_missing_not_none_and_unknown_not_other(self):
        hot,flags,raw,mask,proc=summarize({},self.t,self.end)
        self.assertEqual(hot.sum(),0);self.assertEqual(flags.sum(),0);self.assertFalse(mask.any())
        self.assertIsNone(label(r'\N'))
        with self.assertRaises(ValueError):label('unverified new label')

    def test_actual_ultrasonic_nebulizer_has_its_own_slot(self):
        self.assertNotEqual(label('Ultrasonic neb'),label('Other'))
        self.assertEqual(label(' ULTRASONIC NEB '),17)

    def test_procedure_after_discharge_is_not_clipped_into_available(self):
        self.assertFalse(available(self.t,self.end,self.t,self.t,self.end+timedelta(days=1)))
        self.assertFalse(available(self.t,self.end,self.t+timedelta(hours=1),self.t,self.end))
        self.assertTrue(available(self.t,self.end,self.t,self.end,self.end))

    def test_procedure_pair_requires_same_stay_and_exact_interval(self):
        a=label('Endotracheal tube');obs={'x':[(self.t,a)]}
        wrong=[('y','225792',self.t,self.end)]
        self.assertEqual(summarize(obs,self.t,self.end,wrong)[4][6],0)
        right=[('x','225792',self.t,self.end)]
        self.assertEqual(summarize(obs,self.t,self.end,right)[4][6],1)
        self.assertEqual(summarize(obs,self.t,self.end,[('x','225468',self.t,self.t)])[4][2],1)

    def test_real_source_metadata_and_train_scale_are_finite(self):
        arr,report=derive();json.dumps(report,allow_nan=False)
        self.assertEqual(arr['features'].shape,(19401,115));self.assertTrue(np.isfinite(arr['features']).all())
        self.assertEqual(len(LOOKUP),18);self.assertEqual(len(report['dictionary']),7)
        self.assertTrue(np.all(arr['availability'][:,:90]==0));self.assertTrue(np.all(arr['timing'][:,:90]==0))


if __name__=='__main__':unittest.main()
