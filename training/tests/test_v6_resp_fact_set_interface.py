import unittest
import numpy as np
from patient_data.resp_fact_set_interface import collect,arrays_from

def base():
    return dict(hadm_ids=np.array([10,20,30]),subject_ids=np.array([1,2,3]),patient_splits=np.array(['train','train','validation']))

def record(**changes):
    r=dict(hadm_id='10',concept='dyspnea',role='explicit_discharge',polarity='affirmed',unit=None,
           activity_condition='exertion',value=None,note_id='N1',start=0,end=3,evidence_start_char=0,evidence_end_char=3,unit_key='a')
    r.update(changes);return r

def flow(value,**changes):
    fields=dict(concept='oxygen_flow_litres_notation',unit='L_oxygen_notation',activity_condition='unknown',value=value)
    fields.update(changes)
    return record(**fields)

def pack(records):
    b=base();facts,excluded=collect(records,{10:0,20:1,30:2});arrays,scales=arrays_from(facts,b)
    return facts,excluded,arrays,scales

class Interface(unittest.TestCase):
    def test_unknown_role_isolated(self):
        f,e,_,_=pack([record(role='unknown')]);self.assertFalse(f);self.assertEqual(e[0]['reason'],'unknown_time_role')

    def test_ambiguous_numeric_isolated(self):
        f,e,_,_=pack([flow(2,activity_condition='ambiguous')]);self.assertFalse(f);self.assertEqual(e[0]['reason'],'ambiguous_flow_condition')

    def test_nonnumeric_activity_not_consumed(self):
        f,_,_,_=pack([record(),record(activity_condition='rest',note_id='N2')])
        self.assertEqual(len(f),1);self.assertEqual(f[0]['condition'],'not_applicable');self.assertEqual(len(f[0]['evidence']),2)

    def test_roles_separate(self):
        f,_,_,_=pack([record(),record(role='inpatient_change')]);self.assertEqual(len(f),2)

    def test_polarities_separate(self):
        f,_,_,_=pack([record(),record(polarity='negated')]);self.assertEqual(len(f),2)

    def test_improvement_not_presence(self):
        f,_,_,_=pack([record(),record(concept='dyspnea_improved')]);self.assertEqual(len(f),2)

    def test_units_separate(self):
        f,_,_,_=pack([flow(2),flow(2,unit='L/min')]);self.assertEqual(len(f),2)

    def test_different_values_not_mean(self):
        f,_,_,_=pack([flow(1),flow(3)]);self.assertEqual([r['value'] for r in f],[1,3])

    def test_duplicate_not_frequency(self):
        f,_,a,_=pack([record(),record()]);self.assertEqual(len(f),1);self.assertEqual(len(f[0]['evidence']),1)
        self.assertEqual(a['aggregate'][0].tolist(),[1,0,0])

    def test_reference_preserved(self):
        f,_,_,_=pack([record(),record(note_id='N2',evidence_start_char=5,evidence_end_char=8)])
        self.assertEqual(len(f[0]['evidence']),2)

    def test_permutation_invariant(self):
        r=[record(),flow(3),flow(1),record(note_id='N2')]
        f,e,a,s=pack(r);g,x,b,t=pack(r[::-1]);self.assertEqual(f,g);self.assertEqual(e,x);self.assertEqual(s,t)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_only_training_scales(self):
        r=[flow(1),flow(3,hadm_id='20'),flow(999,hadm_id='30')]
        _,_,a,s=pack(r);r[-1]=flow(100000,hadm_id='30');_,_,b,t=pack(r)
        self.assertEqual(s,t);np.testing.assert_array_equal(a['values'][:2],b['values'][:2])

    def test_unit_without_training_invalid_scale(self):
        _,_,a,s=pack([flow(1),flow(4,hadm_id='30',unit='L/min')])
        self.assertEqual(s['L/min']['method'],'no_training_observation')
        self.assertTrue(a['numeric_present'][1]);self.assertFalse(a['numeric_valid'][1]);self.assertEqual(a['values'][1],0)

    def test_zero_distinct_missing(self):
        _,_,a,_=pack([record(),flow(0)])
        self.assertEqual(a['raw_values'].tolist(),[0,0]);self.assertEqual(a['numeric_present'].tolist(),[False,True])

    def test_empty_finite(self):
        _,_,a,_=pack([]);self.assertEqual(len(a['positions']),0);self.assertFalse(a['observed'].any())
        self.assertTrue(np.isfinite(a['values']).all())

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):pack([flow(float('nan'))])

    def test_unknown_not_rest(self):
        f,_,_,_=pack([flow(2),flow(2,activity_condition='rest')]);self.assertEqual(len(f),2)

    def test_no_source_row(self):
        _,_,a,_=pack([record()]);self.assertEqual(a['observed'].tolist(),[True,False,False])

if __name__=='__main__':unittest.main()
