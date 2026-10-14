import unittest
import numpy as np
import torch
from training.v6.admission_context import raw_row,encode,category,feature_view,Correction

class Tests(unittest.TestCase):
    def row(self,**kwargs):
        return dict(dict(admittime='2026-01-01',dischtime='2026-01-02',edregtime='2025-12-31T23:00:00',edouttime='2026-01-01T01:00:00',
                         admission_type='EW EMER.',admission_location='EMERGENCY ROOM'),**kwargs)
    def test_days_hours_and_overlap(self):
        v,m,c,r=raw_row(self.row());np.testing.assert_array_equal(v,[1,2]);self.assertTrue(m.all());self.assertFalse(r)
    def test_future_ed_not_current(self):
        v,m,c,r=raw_row(self.row(edouttime='2026-01-02T01:00:00'));self.assertFalse(m[1]);self.assertEqual(v[1],0);self.assertIn('ed_after_discharge',r)
    def test_missing_ed_is_not_zero(self):
        v,m,_,_=raw_row(self.row(edregtime='\\N',edouttime='\\N'));self.assertFalse(m[1]);self.assertEqual(v[1],0)
    def test_true_zero_ed_preserved(self):
        v,m,_,_=raw_row(self.row(edregtime='2026-01-01',edouttime='2026-01-01'));self.assertTrue(m[1]);self.assertEqual(v[1],0)
    def test_reverse_ed(self):
        v,m,_,r=raw_row(self.row(edregtime='2026-01-01T02:00:00'));self.assertFalse(m[1]);self.assertIn('ed_reverse_time',r)
    def test_unknown_category(self):
        for s in ('','\\N','INFORMATION NOT AVAILABLE',None):self.assertIsNone(category(s))
    def dataset(self):
        raw=np.column_stack((np.arange(101,dtype=float),np.arange(101,dtype=float)));m=np.ones((101,4),bool)
        cats=[('EW EMER.','EMERGENCY ROOM')]*100+[('NEW HELDOUT CATEGORY','UNSEEN LOCATION')];folds=np.asarray(['train']*100+['validation'])
        return raw,m,cats,folds
    def test_training_only_scale_vocab(self):
        raw,m,c,folds=self.dataset();a,meta=encode(raw,m,c,folds);raw[-1]=1e9;b,meta2=encode(raw,m,c,folds)
        self.assertEqual(meta,meta2);np.testing.assert_array_equal(a['features'][:-1],b['features'][:-1]);self.assertEqual(meta['vocab']['type'],['UNKNOWN','EW EMER.']);self.assertEqual(a['codes'][-1,0],0)
    def test_missing_onehot_zero(self):
        raw,m,c,folds=self.dataset();m[-1,2:]=False;a,_=encode(raw,m,c,folds);self.assertTrue((a['features'][-1,6:]==0).all())
    def test_content_truly_hidden(self):
        x=np.random.default_rng(42).normal(size=(8,32)).astype(np.float32);other=x.copy();other[:,:2]+=99;other[:,6:]*=5
        np.testing.assert_array_equal(feature_view(x,'availability'),feature_view(other,'availability'));self.assertFalse(np.array_equal(feature_view(x,'content'),feature_view(other,'content')))
    def test_model_capacity_fixed_initialization(self):
        a,b=Correction(),Correction();self.assertEqual(sum(p.numel() for p in a.parameters()),1140)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

if __name__=='__main__':unittest.main()
