import unittest
from patient_data.text_time_coverage import coverage,span,time_flags


class Tests(unittest.TestCase):
    def test_union_and_single_chunk_are_different(self):
        self.assertEqual(coverage(0,10,[(0,7),(4,10)]),(1.,False))
        self.assertEqual(coverage(0,10,[(0,10)]),(1.,True))
        self.assertEqual(coverage(0,10,[(0,3),(7,10)]),(.6,False))
        self.assertEqual(coverage(0,10,[]),(0.,False))

    def test_overlap_not_double_counted(self):
        self.assertEqual(coverage(2,8,[(0,10),(0,10)]),(1.,True))

    def test_empty_reverse_offsets_rejected(self):
        for a,b in [(1,0),(1,1),(-1,2),(0,4)]:
            with self.assertRaises(ValueError):span('abc',a,b,'')

    def test_at_and_after_discharge_not_same(self):
        self.assertTrue(time_flags('At discharge she still required oxygen.','hospital_course')['explicit_discharge'])
        self.assertFalse(time_flags('After discharge oxygen was required.','hospital_course')['explicit_discharge'])
        self.assertFalse(time_flags('After discharge dyspnea improved.','hospital_course')['priority_candidate'])

    def test_HPI_not_discharge_state(self):
        f=time_flags('Dyspnea improved at discharge.','history_of_present_illness')
        self.assertFalse(f['priority_candidate'])

    def test_warning_not_current_residual(self):
        f=time_flags('If you develop shortness of breath call 911.','discharge_instructions')
        self.assertTrue(f['warning']);self.assertFalse(f['priority_candidate'])

    def test_plan_and_history_can_coexist_with_discharge(self):
        f=time_flags('At discharge she will continue her baseline oxygen.','hospital_course')
        self.assertTrue(f['priority_candidate']);self.assertTrue(f['plan']);self.assertTrue(f['historical'])

    def test_no_respiratory_keyword_is_not_normal(self):
        self.assertFalse(time_flags('Independent in ADLs.','discharge_condition')['respiratory'])
        self.assertTrue(time_flags('Wheezing resolved.','hospital_course')['priority_candidate'])


if __name__=='__main__':unittest.main()
