import unittest
from patient_data.cxr_report_selection import identify, report_section, usable, choose


class SelectionTests(unittest.TestCase):
    def test_cxr_exam_without_confusion_from_comparison(self):
        self.assertTrue(identify('EXAMINATION: CHEST (PORTABLE AP)\nCOMPARISON: CT chest yesterday.\nIMPRESSION: Clear.')[0].startswith('cxr_'))
        self.assertFalse(identify('EXAMINATION: CT CHEST W/O CONTRAST\nCOMPARISON: chest x-ray\nIMPRESSION: opacity.')[0].startswith('cxr_'))
        self.assertFalse(identify('HISTORY: Evaluate pneumonia.\nCOMPARISON: Chest radiograph yesterday.')[0].startswith('cxr_'))

    def test_section_exact_and_impression_preferred(self):
        source='CHEST RADIOGRAPH\nFINDINGS:\nOld finding.\nIMPRESSION:\n  No pneumonia. \nNOTIFICATION: told team.'
        text, kind, start, end=report_section(source)
        self.assertEqual((text,kind),('No pneumonia.','impression'))
        self.assertEqual(source[start:end],text)
        self.assertEqual(report_section('CHEST RADIOGRAPH\nHISTORY: pneumonia')[0],'')

    def test_future_store_or_exam_missing_times_and_subject_rejected(self):
        adm={'subject_id':'1','admittime':'2020-01-01 00:00:00','dischtime':'2020-01-05 00:00:00'}
        r={'subject_id':'1','charttime':'2020-01-02 00:00:00','storetime':'2020-01-03 00:00:00'}
        self.assertTrue(usable(r,adm)[0])
        for update in ({'charttime':'2020-01-06 00:00:00'}, {'storetime':'2020-01-06 00:00:00'},
                       {'charttime':'2019-12-31 00:00:00'}, {'storetime':'\\N'}, {'subject_id':'2'}):
            self.assertFalse(usable(dict(r,**update),adm)[0])
        self.assertIsNone(choose([]))

    def test_latest_selection_and_tie_determinism(self):
        a=dict(charttime='2020-01-02',storetime='2020-01-03',note_id='a')
        b=dict(charttime='2020-01-03',storetime='2020-01-03',note_id='b')
        self.assertEqual(choose([b,a]),b)
        self.assertEqual(choose([a,b]),b)

    def test_reason_for_exam_does_not_override_mdct_technique(self):
        text='REASON FOR EXAMINATION: New opacity on chest radiograph.\nCOMPARISON: yesterday.\nTECHNIQUE: MDCT of the chest with axial and coronal reformats.\nFINDINGS: lung nodule.'
        self.assertFalse(identify(text)[0].startswith('cxr_'))
        self.assertFalse(identify('REASON FOR EXAM: Preop chest x-ray.\nFINDINGS: opacity.')[0].startswith('cxr_'))


if __name__=='__main__': unittest.main()
