import copy
from dataclasses import replace
from datetime import datetime,timedelta
import unittest
import numpy as np
import torch
from patient_data.v6_expanded_candidates import identity,medication_catalog,summarize
from patient_data import medication_identity as old
from patient_data.v6_expanded_medication_fix import identity as fixed_identity
from training.tests.test_v6_compact_model import compact
from training.v6.compact_model import CompactAnchoredModel
from training.v6.model import HistoryBaseline
from training.v6.detail_extension_experiment import DetailBatch,DetailEncoder
from training.v6.original_interaction import ExpandedBatch,OriginalInteractionEncoder


def sample():
    c=compact();c=replace(c,clinical=replace(c.clinical,
       snippet_vectors=torch.randn(len(c.clinical.snippet_vectors),8)))
    labs=torch.randn(3,90);extra=torch.rand(3,10,10);extra[:,:,5:]=1
    return ExpandedBatch(DetailBatch(c,labs,torch.zeros(3,192)),extra,torch.rand(3,38,23),torch.ones(3,38,dtype=torch.bool))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def encoder(self,**kwargs):
        base=CompactAnchoredModel(HistoryBaseline(),8).encoder
        return OriginalInteractionEncoder(DetailEncoder(base),**kwargs).eval()

    def test_identity_and_scoped_restore(self):
        self.assertEqual(identity('Breo Ellipta'), 'fluticasone_vilanterol')
        self.assertEqual(identity('Levalbuterol'), 'levalbuterol')
        self.assertEqual(identity('amoxicillin-clavulanate'), 'amoxicillin_clavulanate')
        self.assertIsNone(identity('Trelegy Ellipta'))
        self.assertIsNone(identity('guaifenesin-codeine'))
        original=old.identity
        with medication_catalog():self.assertEqual(old.identity('cefepime'),'cefepime')
        self.assertIs(old.identity,original)

    def test_existing_combinations_are_not_shadowed(self):
        for name,expected in (('umeclidinium / vilanterol','umeclidinium_vilanterol'),
           ('Anoro Ellipta','umeclidinium_vilanterol'),('albuterol / ipratropium','albuterol_ipratropium'),
           ('fluticasone / salmeterol','fluticasone_salmeterol'),('budesonide / formoterol','budesonide_formoterol')):
            self.assertEqual(fixed_identity(name),expected)
        self.assertEqual(fixed_identity('umeclidinium'), 'umeclidinium')
        self.assertEqual(fixed_identity('fluticasone / vilanterol'), 'fluticasone_vilanterol')
        self.assertIsNone(fixed_identity('Trelegy Ellipta'))
        self.assertIsNone(fixed_identity('fluticasone / umeclidinium / vilanterol'))

    def test_summary_conflicts_zero_and_latest(self):
        t=datetime(2020,1,1);x,m=summarize([(t,0),(t,0),(t+timedelta(days=1),4),(t+timedelta(days=1),5)],t+timedelta(days=3))
        self.assertTrue(m.all());np.testing.assert_array_equal(x[:4],0);self.assertAlmostEqual(float(x[4]),np.log1p(3),places=6)

    def test_control_and_zero_interaction_preserve_original(self):
        b=sample();e=self.encoder(interaction=True);a,_=e.original(b.base);h,_=e(b)
        torch.testing.assert_close(a,h,atol=2e-6,rtol=1e-6)
        e=self.encoder();a,_=e.original(b.base);h,_=e(b);torch.testing.assert_close(a,h,rtol=0,atol=0)

    def test_interaction_changes_and_gradient(self):
        b=sample();e=self.encoder(labs=True,medications=True,interaction=True)
        e.interaction_gate.data.fill_(.1);h,_=e(b);h.square().mean().backward()
        self.assertGreater(float(e.attention.self_attn.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(e.drug_extra[0].weight.grad.abs().sum()),0)
        e.interaction_gate.data.zero_();zero,_=e(b);self.assertGreater(float((zero-h).detach().abs().max()),1e-5)

    def test_text_permutation_invariant(self):
        b=sample();e=self.encoder(interaction=True);e.interaction_gate.data.fill_(.1);a,_=e(b)
        c=b.base.base;index=c.clinical.snippet_index;p=torch.arange(len(index)-1,-1,-1)
        changed=replace(c.clinical,snippet_vectors=c.clinical.snippet_vectors[p],snippet_index=index[p])
        h,_=e(replace(b,base=replace(b.base,base=replace(c,clinical=changed))))
        torch.testing.assert_close(a,h,atol=3e-6,rtol=2e-6)

    def test_empty_inputs_and_unused_features(self):
        b=sample();e=self.encoder(interaction=True);e.interaction_gate.data.fill_(.2)
        c=b.base.base;z=replace(c.clinical,lab_axis_inputs=torch.zeros_like(c.clinical.lab_axis_inputs),
           current_masks=torch.zeros_like(c.clinical.current_masks),snippet_vectors=c.clinical.snippet_vectors[:0],snippet_index=c.clinical.snippet_index[:0])
        empty=replace(c,clinical=z,med_features=torch.zeros_like(c.med_features),micro_masks=torch.zeros_like(c.micro_masks),support_masks=torch.zeros_like(c.support_masks))
        h,_=e(replace(b,base=replace(b.base,base=empty,labs=torch.zeros_like(b.base.labs))))
        self.assertTrue(torch.isfinite(h).all())
        a,_=e(b);changed,_=e(replace(b,extra_labs=b.extra_labs+100,medication_details=b.medication_details+100))
        torch.testing.assert_close(a,changed,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
