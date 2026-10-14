"""旧规则红复现/新规则绿回归，同一真实extract接口。"""
import importlib
import os
import unittest
from training.tests import test_v6_discharge_resp_rules as old_tests

rules=importlib.import_module(os.environ.get('RESP_RULES_MODULE','patient_data.discharge_resp_rules_r2'))

class RespiratoryR2(unittest.TestCase):
    def facts(self,text):return rules.extract(text,'hospital_course')[0]
    def names(self,text):return [r['concept'] for r in self.facts(text)]

    def test_steroid_not_oxygen_wean(self):
        self.assertNotIn('oxygen_failed_wean',self.names('At discharge unable to wean steroids.'))

    def test_oxygen_failure(self):
        self.assertIn('oxygen_failed_wean',self.names('At discharge unable to wean off oxygen.'))

    def test_unrelated_oxygen_not_object(self):
        self.assertNotIn('oxygen_failed_wean',self.names('Unable to wean steroids while receiving oxygen.'))

    def test_attempt_not_success(self):
        self.assertNotIn('oxygen_weaned_off',self.names('Attempted to wean off oxygen prior to discharge.'))

    def test_instruction_not_success(self):
        self.assertEqual(self.facts('At discharge instructed to wean off oxygen.'),[])

    def test_goal_not_success(self):
        self.assertEqual(self.facts('At discharge goal is to wean off oxygen.'),[])

    def test_was_weaned_success(self):
        self.assertIn('oxygen_weaned_off',self.names('At discharge she was weaned off oxygen.'))

    def test_base_wean_not_completed(self):
        self.assertNotIn('oxygen_weaned_off',self.names('At discharge wean off oxygen.'))

    def test_is_improved(self):
        self.assertEqual(self.names('At discharge dyspnea is improved.'),['dyspnea_improved'])

    def test_was_improved(self):
        self.assertEqual(self.names('At discharge dyspnea was significantly improved.'),['dyspnea_improved'])

    def test_improved_prefix(self):
        self.assertEqual(self.names('At discharge improved dyspnea.'),['dyspnea_improved'])

    def test_improvement_not_absence(self):
        self.assertEqual(self.facts('At discharge SOB improved.')[0]['polarity'],'affirmed')

    def test_unbound_improvement(self):
        self.assertNotIn('dyspnea',self.names('At discharge improved cough and shortness of breath.'))

    def test_termination_keeps_residual(self):
        self.assertEqual(self.names('At discharge cough improved but dyspnea persists.'),['dyspnea'])

    def test_direct_negation(self):
        self.assertEqual(self.facts('At discharge denies dyspnea.')[0]['polarity'],'negated')

    def test_no_fever_but_sob(self):
        self.assertEqual(self.facts('At discharge no fever but SOB persists.')[0]['polarity'],'affirmed')

    def test_ambiguous_negation(self):
        self.assertEqual(self.facts('At discharge denies fever, chills and SOB.'),[])

    def test_oxygen_requirement_narrow(self):
        self.assertNotIn('oxygen_required',self.names('At discharge required antibiotics and oxygen was weaned.'))

    def test_mixed_flow_after(self):
        f=[r for r in self.facts('At discharge 1L NC at rest and 3L NC with ambulation.') if r['value'] is not None]
        self.assertEqual([(r['value'],r['activity_condition']) for r in f],[(1.,'rest'),(3.,'exertion')])

    def test_mixed_flow_before(self):
        f=[r for r in self.facts('At discharge at rest 1L NC and with ambulation 3L NC.') if r['value'] is not None]
        self.assertEqual([(r['value'],r['activity_condition']) for r in f],[(1.,'rest'),(3.,'exertion')])

    def test_flow_condition_unknown(self):
        self.assertEqual(self.facts('At discharge 2L NC.')[0]['activity_condition'],'unknown')

    def test_single_flow_two_conditions(self):
        self.assertEqual(self.facts('At discharge 2L NC at rest and with ambulation.')[0]['activity_condition'],'ambiguous')

    def test_role_not_carried_across_sentence(self):
        self.assertEqual(self.facts('At discharge stable. Dyspnea improved.')[0]['role'],'inpatient_change')

    def test_flow_exact_evidence(self):
        text='At discharge at rest 1.5 L/min NC and with ambulation 3L NC.'
        f=self.facts(text)
        for r in f:self.assertIn(text[r['start']:r['end']],['1.5 L/min NC','3L NC'])

class PreviousBoundaries(old_tests.RespiratoryFacts):
    def facts(self,text,section='hospital_course'):return rules.extract(text,section)[0]

if __name__=='__main__':unittest.main()
