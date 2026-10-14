"""完整章节比较的路由/空输入/排列不变性，不是临床性能证据。"""
import unittest
from dataclasses import replace
import torch
from patient_data.align_saved_sections import section_inputs, ROLES, FOUR
from training.v6.saved_section_experiment import SectionEncoder
from training.v6.model import ClinicalBatch


class SavedSectionTests(unittest.TestCase):
    def test_raw_section_whitespace(self):
        raw={'note_type':'DS','text':'History of Present Illness: hello  \nworld\nDischarge Condition:\nwell'}
        result=section_inputs(raw)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0][0].text,'hello\nworld')
        self.assertEqual(tuple(ROLES[i] for i in FOUR),('history_of_present_illness','past_medical_history','hospital_course','discharge_condition'))

    def batch(self, vectors, index):
        return ClinicalBatch(torch.zeros(3,14),torch.zeros(3,14,dtype=torch.bool),torch.zeros(3,22,9),
            torch.zeros(3,7),torch.zeros(3,7,dtype=torch.bool),vectors,index)

    def test_text_permutation_empty_and_no_duplicate_weight(self):
        torch.manual_seed(42); model=SectionEncoder().eval()
        vectors=torch.randn(5,768); ix=torch.tensor([[0,1],[0,1],[0,9],[1,2],[1,7]])
        a,p=model._text(self.batch(vectors,ix),3,mask_pmh=False)
        order=torch.tensor([4,2,0,3,1])
        b,q=model._text(self.batch(vectors[order],ix[order]),3,mask_pmh=False)
        torch.testing.assert_close(a,b); self.assertTrue(torch.equal(p,q)); self.assertTrue(torch.equal(a[2],torch.zeros(128)))
        empty,present=model._text(self.batch(vectors[:0],ix[:0]),3,mask_pmh=False)
        self.assertTrue(torch.equal(empty,torch.zeros(3,128))); self.assertFalse(present.any())

    def test_pmh_mask_and_invalid_field(self):
        model=SectionEncoder().eval(); v=torch.randn(1,768); ix=torch.tensor([[0,2]])
        a,p=model._text(self.batch(v,ix),3,mask_pmh=True)
        self.assertFalse(p.any()); self.assertEqual(float(a.abs().sum()),0.)
        with self.assertRaises(ValueError): model._text(self.batch(v,torch.tensor([[0,10]])),3,mask_pmh=False)


if __name__=='__main__': unittest.main()
