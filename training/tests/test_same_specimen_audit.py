from datetime import datetime
import unittest
import numpy as np
from patient_data.same_specimen_audit import bind,pair_slot,latest_sets,collisions,AXES
from patient_data.preprocess_events import LABS


def record(j=0,**changes):
    r=dict(hadm_id='1',specimen_id='10',subject_id='3',axis=AXES[j],charttime='2020-01-01 01:00:00',
           storetime='2020-01-01 02:00:00',labevent_id=str(j+1),value=7.3 if j==0 else 60.,
           unit=LABS['50820' if j==0 else '50818'][1],lower=7.35 if j==0 else 35.,upper=7.45 if j==0 else 45.,
           severity=(7.35-7.3)/.1 if j==0 else 1.,day=1/24)
    return dict(r,**changes)


class SpecimenTests(unittest.TestCase):
    def test_pair(self):
        p,reason=pair_slot([record(),record(1)])
        self.assertEqual(reason,'paired');self.assertEqual(p['values'],[7.3,60.])

    def test_missing_specimen(self):
        self.assertEqual(pair_slot([record(specimen_id=''),record(1)])[1],'missing_specimen')

    def test_different_hospital_or_specimen(self):
        for changes in (dict(hadm_id='2'),dict(specimen_id='11')):
            self.assertEqual(pair_slot([record(),record(1,**changes)])[1],'different_admission_or_specimen')

    def test_missing_component(self):
        self.assertEqual(pair_slot([record()])[1],'missing_component')

    def test_duplicate_not_averaged(self):
        self.assertEqual(pair_slot([record(),record(value=7.4),record(1)])[1],'duplicate_component')

    def test_time_boundary(self):
        self.assertEqual(pair_slot([record(),record(1,charttime='2020-01-01 02:00:00')])[1],'paired')
        self.assertEqual(pair_slot([record(),record(1,charttime='2020-01-01 02:00:01')])[1],'chart_gap_over_one_hour')

    def event(self):
        r=record();return dict(kind='lab',concept='ph:arterial',specimen_type='arterial',patient_split='train',
            subject_id='3',hadm_id='1',source_id='1',specimen_id='10',value_unit=r['unit'],value_raw='7.3',
            charttime=r['charttime'],storetime=r['storetime'],elapsed_days=str(1/24))

    def test_binding(self):
        b=(datetime(2020,1,1),datetime(2020,1,2))
        self.assertEqual(bind(record(),self.event(),b,3)['value'],7.3)
        for k,v in (('specimen_type','venous'),('value_unit','pH'),('patient_split','validation'),('source_id','9')):
            with self.assertRaises(ValueError):bind(record(),dict(self.event(),**{k:v}),b,3)

    def test_post_discharge_not_bound(self):
        with self.assertRaises(ValueError):bind(record(),self.event(),(datetime(2020,1,1),datetime(2020,1,1,1,30)),3)

    def test_actual_inherited_ph_unit(self):
        from patient_data.preprocess_events import LABS
        unit=LABS['50820'][1]
        b=(datetime(2020,1,1),datetime(2020,1,2))
        bound=bind(record(unit=unit),dict(self.event(),value_unit=unit),b,3)
        self.assertEqual(bound['unit'],'units')

    def test_latest_same_time_set_keeps_multiple_pairs(self):
        p,_=pair_slot([record(),record(1)]);q=dict(p,values=[7.4,50.])
        self.assertEqual(latest_sets([p,q],[1])[0],((7.3,60.),(7.4,50.)))
        self.assertEqual(latest_sets([p,q],[2])[0],())

    def test_exact_signatures_and_patients(self):
        physical=[((7.3,60.),),((7.4,50.),),()];x=np.zeros((3,9))
        self.assertEqual(collisions(x,physical,np.array([1,2,3]))[0]['groups'],1)
        x[1,0]=1e-14
        self.assertEqual(collisions(x,physical,np.array([1,2,3]))[0]['groups'],0)
        x[:]=0
        self.assertEqual(collisions(x,physical,np.array([1,1,3]))[0]['groups'],0)


if __name__=='__main__':unittest.main()
