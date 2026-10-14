"""补充来源的顺序/身份/缺失合同，不启动训练。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from patient_data.export_mimiciv import sha256
from training.v6.source_context_data import attach,checked_arrays,SourceContextData
from training.tests.test_v6_core import batch


class SourceTests(unittest.TestCase):
    def test_member_order_and_hash_checked(self):
        ids=np.array([1,2]); base=SimpleNamespace(arrays={'hadm_ids':ids,'subject_ids':ids,'patient_splits':np.array(['train','test'])},
                                                 provenance={'state_manifest_sha256':'parent'})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); np.savez(p/'data.npz',**base.arrays)
            meta={'status':'ready','parent_state_manifest_sha256':'parent','output_sha256':{'data.npz':sha256(p/'data.npz')}}
            (p/'manifest.json').write_text(json.dumps(meta))
            checked_arrays(p,'ready','data.npz',base)
            changed=SimpleNamespace(arrays={**base.arrays,'hadm_ids':ids[::-1]},provenance=base.provenance)
            with self.assertRaises(ValueError): checked_arrays(p,'ready','data.npz',changed)
            meta['output_sha256']['data.npz']='wrong'; (p/'manifest.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError): checked_arrays(p,'ready','data.npz',base)
    def test_corrected_mask_does_not_mutate_base(self):
        original=batch(); positions=np.array([0,1,2])
        base=SimpleNamespace(collate=lambda positions,device:(torch.zeros(3,6),original,torch.ones(3),torch.zeros(3)))
        eos={'eos_features':np.zeros((3,2,20),np.float32),'med_source_mask':np.array([[False]*7,[True]*7,[True]*7])}
        rm={k:np.zeros(s,dtype=bool if 'mask' in k else np.float32) for k,s in (
            ('micro_values',(3,6)),('micro_masks',(3,6)),('resp_numeric_scaled',(3,8,5)),('resp_numeric_masks',(3,8,5)),
            ('support_values',(3,9)),('support_masks',(3,9)),('recorded_duration_hours',(3,2)),('recorded_duration_masks',(3,2)))}
        before=original.med_source_mask.clone()
        _,result,_,_=SourceContextData(base,eos,rm,{}).collate(positions)
        self.assertFalse(result.clinical.med_source_mask[0].any())
        torch.testing.assert_close(original.med_source_mask,before)
        self.assertEqual(tuple(result.eos_features.shape),(3,2,20))
        self.assertFalse(hasattr(result,'event_type'))


if __name__=='__main__': unittest.main()
