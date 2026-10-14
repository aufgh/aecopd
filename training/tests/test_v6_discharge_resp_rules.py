import unittest
from patient_data.discharge_resp_rules import extract, clauses

class RespiratoryFacts(unittest.TestCase):
    def facts(self, text, section='hospital_course'):
        return extract(text, section)[0]

    def test_discharge_flow(self):
        f=self.facts('At discharge she required 2 L NC with ambulation.')
        self.assertEqual(f[0]['value'],2);self.assertEqual(f[0]['role'],'explicit_discharge');self.assertTrue(f[0]['exertional'])

    def test_weaned(self):
        f=self.facts('He was weaned to room air prior to discharge.')
        self.assertEqual(f[0]['concept'],'oxygen_weaned_room_air');self.assertEqual(f[0]['role'],'explicit_discharge')

    def test_inpatient_not_discharge(self):
        self.assertEqual(self.facts('She was weaned off oxygen.')[0]['role'],'inpatient_change')

    def test_failed_wean(self):
        f=self.facts('At discharge he was unable to wean off oxygen.')
        self.assertEqual([x['concept'] for x in f],['oxygen_failed_wean'])

    def test_negated_dyspnea(self):
        f=self.facts('At discharge denies shortness of breath.')
        self.assertEqual(f[0]['polarity'],'negated')

    def test_improvement(self):
        self.assertEqual(self.facts('SOB improved with treatment.')[0]['concept'],'dyspnea_improved')

    def test_residual(self):
        f=self.facts('At discharge she still had dyspnea with walking.')
        self.assertEqual(f[0]['concept'],'dyspnea');self.assertTrue(f[0]['exertional'])

    def test_warning(self):
        self.assertEqual(self.facts('If you develop SOB call 911.','discharge_instructions'),[])

    def test_plan(self):
        self.assertEqual(self.facts('She will be discharged on oxygen.'),[])

    def test_post_discharge(self):
        self.assertEqual(self.facts('After discharge she required oxygen.'),[])

    def test_baseline(self):
        self.assertEqual(self.facts('At baseline she required oxygen.'),[])

    def test_no_baseline_delta(self):
        self.assertEqual(self.facts('Baseline 2L NC but at discharge 4L NC.'),[])

    def test_hpi(self):
        self.assertEqual(self.facts('At discharge SOB.','history_of_present_illness'),[])

    def test_admission(self):
        self.assertEqual(self.facts('On admission she had SOB.'),[])

    def test_other_person(self):
        self.assertEqual(self.facts('Her mother required oxygen.'),[])

    def test_possible(self):
        self.assertEqual(self.facts('She may need oxygen.'),[])

    def test_ra_unknown(self):
        self.assertEqual(self.facts('At discharge sat 94% RA.'),[])

    def test_flow_range(self):
        self.assertEqual(self.facts('At discharge 2-4L NC.'),[])

    def test_flow_bound(self):
        self.assertEqual(self.facts('At discharge >2L NC.'),[])
        self.assertEqual(self.facts('At discharge 2 to 4L NC.'),[])

    def test_decimal_offset(self):
        text='At discharge she required 1.5 L nasal cannula.'
        f=self.facts(text)[0];self.assertEqual(f['value'],1.5)
        self.assertEqual(text[f['start']:f['end']],'1.5 L nasal cannula')

    def test_unit_required(self):
        self.assertEqual(self.facts('At discharge 2 NC.'),[])

    def test_unknown_time(self):
        self.assertEqual(self.facts('SOB with activity.')[0]['role'],'unknown')

    def test_discharge_section(self):
        self.assertEqual(self.facts('On room air.','discharge_condition')[0]['role'],'discharge_section')

    def test_wean_from_not_last(self):
        f=self.facts('At discharge weaned from 4L NC to room air.')
        self.assertNotIn('oxygen_flow_litres_notation',[x['concept'] for x in f])

    def test_negation_termination(self):
        self.assertEqual(self.facts('At discharge no fever but SOB persists.')[0]['polarity'],'affirmed')

    def test_ambiguous_negation_list(self):
        self.assertEqual(self.facts('At discharge denies fever, chills and SOB.'),[])

    def test_explicit_rate_unit(self):
        self.assertEqual(self.facts('At discharge 2L/min NC.')[0]['unit'],'L/min')
        self.assertEqual(self.facts('At discharge 2L NC.')[0]['unit'],'L_oxygen_notation')

    def test_no_cross_sentence_discharge(self):
        f=self.facts('At discharge stable. SOB improved.')
        self.assertEqual(f[0]['role'],'inpatient_change')

    def test_clause_offsets(self):
        text='A; 1.5 L NC. B';parts=list(clauses(text))
        self.assertTrue(all(text[a:b]==s for a,b,s in parts));self.assertIn(' 1.5 L NC',[s for _,_,s in parts])

if __name__=='__main__':unittest.main()
