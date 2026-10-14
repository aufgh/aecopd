from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from training.tests.test_v6_compact_model import compact
from training.v6.model import HistoryBaseline
from training.v6.cxr_context import CXRBatch,CXRAnchoredModel,mask_cxr,attach,SELECTED,STATES
from training.v6.risk import competing_nll
from patient_data.export_mimiciv import sha256


def batch(device='cpu'):
    codes=torch.tensor([[0]*7,[1]*7,[2]*7],device=device)
    return CXRBatch(compact(device),torch.nn.functional.one_hot(codes,4).float().reshape(3,28),
                    torch.ones(3,dtype=torch.bool,device=device))


class CXRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_missing_and_blank_report_distinct(self):
        model=CXRAnchoredModel(HistoryBaseline(),8,width=16).eval(); b=batch()
        _,with_image=model.encoder(b); omitted=mask_cxr(b,'without_cxr'); _,without=model.encoder(omitted)
        self.assertEqual(without['xray'].abs().sum(),0)
        self.assertGreater(with_image['xray'][0].abs().sum(),0)
        for key in ('structured','lab','medication','microbiology','support','text'):
            torch.testing.assert_close(with_image[key],without[key],rtol=0,atol=0)

    def test_cxr_content_matters_and_ablation_removes_it(self):
        model=CXRAnchoredModel(HistoryBaseline(),8,width=16).eval(); b=batch()
        changed=replace(b,cxr_onehot=b.cxr_onehot.roll(1,1))
        a,_=model.encoder(b); c,_=model.encoder(changed)
        self.assertGreater((a-c).abs().max(),1e-6)
        a,_=model.encoder(mask_cxr(b,'without_cxr')); c,_=model.encoder(mask_cxr(changed,'without_cxr'))
        torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_zero_initial_history_and_active_gradient(self):
        model=CXRAnchoredModel(HistoryBaseline(),8,width=16); h=torch.zeros(3,6); b=batch()
        torch.testing.assert_close(model(h,b).log_amplitude,model.baseline(h).log_amplitude,rtol=0,atol=0)
        torch.nn.init.normal_(model.delta_head[-1].weight,std=.1)
        competing_nll(model(h,b),torch.tensor([4.,40.,365.]),torch.tensor([1,2,0])).backward()
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        self.assertGreater(model.encoder.xray[0].weight.grad.abs().sum(),0)

    def test_invalid_contract_rejected(self):
        model=CXRAnchoredModel(HistoryBaseline(),8,width=16)
        with self.assertRaises(TypeError): model.encoder(compact())
        b=batch(); b.cxr_onehot[0,0]=.5
        with self.assertRaises(ValueError): model.encoder(b)

    def test_initialization_not_affected_by_text_dimension(self):
        states=[]
        for dimension in (8,20):
            torch.manual_seed(42); m=CXRAnchoredModel(HistoryBaseline(),dimension,width=16)
            states.append({k:v for k,v in m.state_dict().items() if 'text_projection' not in k})
        for key in states[0]: torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)

    def test_disk_bridge_checks_hash_members_and_onehot(self):
        ids=dict(hadm_ids=np.array([1,2,3]),subject_ids=np.array([1,2,3]),patient_splits=np.array(['train','validation','test']))
        c=compact()
        base=SimpleNamespace(arrays=ids,provenance={'state_manifest_sha256':'parent'})
        data=SimpleNamespace(base=base,provenance={'compact_manifest_sha256':'compact'},
            collate=lambda pos,device: (torch.zeros(3,6,device=device),c,torch.ones(3,device=device),torch.zeros(3,dtype=torch.long,device=device)))
        arrays=dict(ids,cxr_states=np.zeros((3,7),np.int8),cxr_onehot=batch().cxr_onehot.numpy(),
                    cxr_present=np.ones(3,bool),condition_names=np.array(SELECTED),state_names=np.array(STATES))
        arrays['cxr_onehot']=np.eye(4,dtype=np.float32)[arrays['cxr_states']].reshape(3,28)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            def save():
                np.savez(path/'cxr_context.npz',**arrays)
                manifest={'status':'chexbert_labeled_temporally_checked_not_prediction_validated',
                          'parent_state_manifest_sha256':'parent','reports':3,
                          'output_sha256':{'cxr_context.npz':sha256(path/'cxr_context.npz')}}
                (path/'manifest.json').write_text(json.dumps(manifest))
            save(); received=attach(data,path); _,b,_,_=received.collate(np.arange(3))
            self.assertEqual(tuple(b.cxr_onehot.shape),(3,28))
            arrays['hadm_ids']=np.array([3,2,1]); save()
            with self.assertRaises(ValueError): attach(data,path)
            arrays['hadm_ids']=ids['hadm_ids']; arrays['cxr_onehot'][0,0]=0; save()
            with self.assertRaises(ValueError): attach(data,path)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_availability_control_has_no_content_variation(self):
        from training.v6.cxr_experiment import prepared
        b=batch('cuda'); changed=replace(b,cxr_onehot=b.cxr_onehot.roll(1,1))
        def data_for(value):
            return SimpleNamespace(collate=lambda pos,device:(torch.zeros(3,6,device=device),value,
                                  torch.ones(3,device=device),torch.zeros(3,dtype=torch.long,device=device)))
        a=prepared(data_for(b),np.arange(3),3,'availability_only')[0][2]
        c=prepared(data_for(changed),np.arange(3),3,'availability_only')[0][2]
        torch.testing.assert_close(a.cxr_onehot,c.cxr_onehot,rtol=0,atol=0)
        self.assertIs(a.compact,b.compact)
        self.assertTrue(a.cxr_present.all())
        omitted=prepared(data_for(b),np.arange(3),3,'without_cxr')[0][2]
        self.assertFalse(omitted.cxr_present.any())
        self.assertEqual(omitted.cxr_onehot.abs().sum(),0)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_cuda_forward_backward(self):
        model=CXRAnchoredModel(HistoryBaseline(),8,width=16).cuda()
        loss=competing_nll(model(torch.zeros(3,6,device='cuda'),batch('cuda')),torch.tensor([4.,40.,365.],device='cuda'),torch.tensor([1,2,0],device='cuda'))
        loss.backward(); self.assertTrue(torch.isfinite(loss))


if __name__=='__main__': unittest.main()
