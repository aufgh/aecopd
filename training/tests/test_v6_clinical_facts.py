from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.model import HistoryBaseline
from training.v6.clinical_facts_context import FactsBatch,FactsAnchoredModel,WIDTHS,mask
from training.v6.risk import competing_nll
from patient_data.clinical_facts.features import ct_vector,discharge_vector,CT_NAMES,DISCHARGE_NAMES,read_results

def batch(device='cpu'):
    return FactsBatch(compact(device),tuple(torch.zeros(3,width,device=device) for width in WIDTHS),torch.ones(3,3,dtype=torch.bool,device=device))

class FactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_zero_initial_and_frozen_history(self):
        model=FactsAnchoredModel(HistoryBaseline(),8,width=16); b=batch(); h=torch.zeros(3,6)
        torch.testing.assert_close(model(h,b).log_amplitude,model.baseline(h).log_amplitude,rtol=0,atol=0)
        torch.nn.init.normal_(model.delta_head[-1].weight,std=.1)
        b.values[0][:,0]=1
        competing_nll(model(h,b),torch.tensor([3.,50.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        self.assertGreater(model.encoder.fact_branches[0][0].weight.grad.abs().sum(),0)

    def test_source_content_and_missing_masks(self):
        model=FactsAnchoredModel(HistoryBaseline(),8,width=16).eval(); b=batch()
        changed=replace(b,values=tuple(v.clone() for v in b.values)); changed.values[1][0,0]=1
        original,_=model.encoder(b); other,_=model.encoder(changed)
        self.assertGreater((other-original).abs().sum(),0)
        a,_=model.encoder(mask(b,'availability_only')); c,_=model.encoder(mask(changed,'availability_only'))
        torch.testing.assert_close(a,c,rtol=0,atol=0)
        _,branches=model.encoder(mask(b,'without_new'))
        for key in ('cxr','ct','discharge'): self.assertEqual(branches[key].abs().sum(),0)
        self.assertIs(mask(b,'without_new').compact,b.compact)

    def test_same_init_other_text_width(self):
        states=[]
        for d in (8,20):
            torch.manual_seed(42); m=FactsAnchoredModel(HistoryBaseline(),d,width=16)
            states.append({k:v for k,v in m.state_dict().items() if 'text_projection' not in k})
        for k in states[0]: torch.testing.assert_close(states[0][k],states[1][k],rtol=0,atol=0)

    def test_fact_conflicts_and_missing(self):
        self.assertEqual(len(ct_vector([])),len(CT_NAMES)); self.assertEqual(ct_vector([]).sum(),0)
        fs=[{'field':'emphysema','value':v,'charttime':'01'} for v in ('present','absent')]
        output=ct_vector(fs)
        self.assertEqual(output[CT_NAMES.index('emphysema:uncertain')],1)
        self.assertEqual(output[CT_NAMES.index('emphysema:conflict')],1)
        fs=[{'field':'care_limits','value':v,'section_role':'hospital_course'} for v in ('dnr','dni')]
        output=discharge_vector(fs)
        self.assertEqual(output[DISCHARGE_NAMES.index('care_limits:conflict')],0)
        self.assertEqual(output[DISCHARGE_NAMES.index('care_limits:dnr')],1)
        self.assertEqual(output[DISCHARGE_NAMES.index('care_limits:dni')],1)

    def test_partial_extraction_not_trainable(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'status.json').write_text(json.dumps({'status':'running'}))
            (p/'contract.json').write_text('{}')
            with self.assertRaises(ValueError): read_results(p,[])

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_cuda_forward_backward(self):
        m=FactsAnchoredModel(HistoryBaseline(),8,width=16).cuda()
        loss=competing_nll(m(torch.zeros(3,6,device='cuda'),batch('cuda')),torch.tensor([3.,50.,365.],device='cuda'),torch.tensor([1,2,0],device='cuda'))
        loss.backward(); self.assertTrue(torch.isfinite(loss))

if __name__=='__main__': unittest.main()
