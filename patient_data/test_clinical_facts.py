import unittest
from patient_data.clinical_facts.imaging import modality, sections
from patient_data.clinical_facts.llm import validate, LocalBonsai
from patient_data.clinical_facts.cxr_trajectory import aggregate, SELECTED
from patient_data.clinical_facts.tasks import blocks
from patient_data.clinical_facts.guards import guard

class SourceTests(unittest.TestCase):
    def test_current_not_comparison(self):
        self.assertEqual(modality('FINDINGS: Compared to a chest radiograph from yesterday, no change.')[0],'unknown')
        self.assertEqual(modality('HISTORY: COPD and CT chest\nEXAM: CHEST PA AND LATERAL\nFINDINGS: No edema.')[0],'cxr')
        self.assertEqual(modality('TECHNIQUE: CT of the chest obtained.\nFINDINGS: AP chest shows prior effusion.')[0],'chest_ct')
        self.assertEqual(modality('FINDINGS: AP and lateral chest demonstrate no edema.')[0],'cxr')
        self.assertEqual(modality('EXAM: CT HEAD\nFINDINGS: Chest radiograph showed edema.')[0],'other')

    def test_exact_sections(self):
        text='HISTORY: cough\nFINDINGS:  No edema.\nIMPRESSION: Emphysema.\n'
        result=sections(text)
        self.assertEqual(len(result),2)
        for span in result: self.assertEqual(text[span['start_char']:span['end_char']],span['text'])

    def test_quote_failure_removes_fact(self):
        facts,errors=validate({'facts':[['emphysema','present','Invented quote']]},'chest_ct','No emphysema.')
        self.assertEqual(facts,[]); self.assertEqual(errors,['nonverbatim_evidence'])
        facts,errors=validate({'facts':[['emphysema','absent','No emphysema.']]},'chest_ct','No emphysema.')
        self.assertEqual(len(facts),1); self.assertEqual(errors,[])

    def test_no_remote(self):
        with self.assertRaises(ValueError): LocalBonsai('https://example.com')

    def test_temporal_negation(self):
        def record(t,state):
            return {'charttime':t,'labels14':{k:state for k in SELECTED},'chunk_conflicts':{}}
        same=aggregate([record('01',1),record('01',2)])
        self.assertEqual(same[0,10],0); self.assertEqual(same[0,12],1)
        later=aggregate([record('01',1),record('02',0)])
        self.assertEqual(later[0,10],0)
        later=aggregate([record('01',1),record('02',2)])
        self.assertEqual(later[0,10],1)
        self.assertTrue((aggregate([])==0).all())

    def test_whitespace_only_evidence_equivalence(self):
        source='No pleural\r\n  effusion.'
        facts,errors=validate({'facts':[['pleural_effusion','absent','No pleural effusion.']]},'chest_ct',source)
        self.assertEqual(facts[0]['quote'],source); self.assertEqual(errors,[])
        facts,errors=validate({'facts':[['pleural_effusion','absent','No pleural effusions.']]},'chest_ct',source)
        self.assertEqual(facts,[])

    def test_complete_source_chunks(self):
        text=('Clinical narrative. '*1000)+'END'
        result=blocks([{'role':'hospital_course','text':text,'start_char':50,'end_char':50+len(text)}],limit=1000)
        recovered=''.join(b['text'][m['input_start']:m['input_end']] for b in result for m in b['source_maps'])
        self.assertEqual(recovered,text)

    def test_concept_mismatch_rejected(self):
        source='No intraluminal filling defect in the pulmonary artery.'
        fact={'field':'enlarged_pulmonary_artery','value':'absent','quote':source,'local_start':0,'local_end':len(source)}
        task={'text':source,'source_maps':[{'role':'FINDINGS','input_start':0,'input_end':len(source),'original_start':20}]}
        self.assertEqual(guard(fact,task)[1],'pulmonary_artery_patency_not_size')
        source='Bronchial wall thickening.'
        fact=dict(fact,field='bronchiectasis',value='present',quote=source,local_end=len(source))
        task={'text':source,'source_maps':[{'role':'FINDINGS','input_start':0,'input_end':len(source),'original_start':20}]}
        self.assertEqual(guard(fact,task)[1],'unsupported_CT_concept')

    def test_semantically_wrong_quote_not_enough(self):
        cases=[('oxygen_at_discharge','room_air','discharge_condition','Activity Status: Ambulatory - Independent.'),
               ('ventilation_at_discharge','weaned','hospital_course','Unable to wean him off O2.'),
               ('care_limits','dnr','hospital_course','Code Status: Full (confirmed)'),
               ('living_alone','present','discharge_disposition','Home'),
               ('steroid_discharge_course','completed','hospital_course','Resolved with antibiotic therapy.'),
               ('residual_dyspnea','absent','discharge_physical_exam','No acute distress.')]
        for field,value,role,text in cases:
            fact={'field':field,'value':value,'quote':text,'local_start':0,'local_end':len(text)}
            task={'text':text,'source_maps':[{'role':role,'input_start':0,'input_end':len(text),'original_start':0}]}
            self.assertIsNone(guard(fact,task)[0],field)

if __name__=='__main__': unittest.main()
