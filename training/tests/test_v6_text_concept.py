import unittest
from dataclasses import asdict
import numpy as np
import torch
from training.v6.text_concept import targets,masked_bce,task_split,Reader,replay_early,AXES
from patient_data.v6_lab_state import NUMERIC,FLAGS
from training.v6.early_stopping import EarlyStopping

class Tests(unittest.TestCase):
    def source(self):
        raw=np.zeros((3,5,9));raw[0,:,6]=1;raw[0,:,1]=1;raw[1,:,6]=1;raw[1,:,2]=2
        return dict(lab_axis_raw=raw,axis_names=np.asarray(AXES),feature_names=np.asarray(NUMERIC+FLAGS))
    def test_targets_and_missing(self):
        y,mask,age=targets(self.source());self.assertTrue(mask[:2].all());self.assertFalse(mask[2].any())
        np.testing.assert_array_equal(y[0],np.tile([1,0],5));np.testing.assert_array_equal(y[1],np.tile([0,1],5))
    def test_axis_order_not_assumed(self):
        a=self.source();order=[4,1,3,2,0];b=dict(a,axis_names=a['axis_names'][order],lab_axis_raw=a['lab_axis_raw'][:,order])
        for x,y in zip(targets(a),targets(b)):np.testing.assert_array_equal(x,y)
    def test_wrong_columns_rejected(self):
        a=self.source();a['feature_names']=a['feature_names'][::-1]
        with self.assertRaises(ValueError):targets(a)
    def test_unobserved_not_supervised(self):
        a=torch.randn(4,10,requires_grad=True);y=torch.zeros_like(a);mask=torch.ones_like(a,dtype=torch.bool);mask[:,0]=False
        loss=masked_bce(a,y,mask);loss.backward();self.assertTrue(a.grad[:,0].eq(0).all())
        b=a.detach().clone();b[:,0]=100.;torch.testing.assert_close(masked_bce(b,y,mask),loss.detach())
    def test_empty_loss(self):
        a=torch.randn(3,10,requires_grad=True);loss=masked_bce(a,torch.zeros_like(a),torch.zeros_like(a,dtype=torch.bool))
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(a.grad.eq(0).all())
    def test_heads_equally_weighted(self):
        a=torch.zeros(3,2);y=torch.zeros_like(a);mask=torch.tensor([[1,1],[1,0],[1,0]],dtype=torch.bool)
        a[:,0]=2.;expected=(torch.nn.functional.softplus(torch.tensor(2.))+torch.log(torch.tensor(2.)))/2
        torch.testing.assert_close(masked_bce(a,y,mask),expected)
    def test_reader_init_fixed(self):
        a,b=Reader(),Reader();self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
    def test_patient_boundary(self):
        subjects=np.repeat(np.arange(1000),2);mask=np.ones((2000,10),bool);y=np.tile(np.arange(2000)[:,None]%2,(1,10)).astype(np.float32)
        fit,hold,active=task_split(subjects,np.arange(2000),np.ones(2000,bool),mask,y)
        self.assertFalse(set(subjects[fit])&set(subjects[hold]));self.assertTrue(active.all())
    def test_degenerate_head_disabled(self):
        subjects=np.repeat(np.arange(1000),2);mask=np.ones((2000,10),bool);y=np.tile(np.arange(2000)[:,None]%2,(1,10)).astype(np.float32);y[:,0]=0
        _,_,active=task_split(subjects,np.arange(2000),np.ones(2000,bool),mask,y);self.assertFalse(active[0]);self.assertTrue(active[1:].all())
    def test_early_replay(self):
        e=EarlyStopping(4,5,1e-4);rows=[]
        for epoch in range(6):e.observe(epoch,1.);rows.append(dict(epoch=epoch,internal_holdout_bce=1.))
        replay_early(rows,dict(early=asdict(e),actual_epochs=5,stop_reason='patience_exhausted'),True)

if __name__=='__main__':unittest.main()
