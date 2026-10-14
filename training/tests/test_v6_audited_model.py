import copy
import numpy as np
import torch
from dataclasses import replace
from patient_data.v6_input_audit_compact import scalar_dose, scale_per_unit
from training.v6.audited_model import AuditedBatch, AuditedEncoder, AuditedModel, flat_features
from training.v6.model import HistoryBaseline


def batch():
    z=lambda *shape:torch.zeros(*shape)
    return AuditedBatch(z(3,4,24),torch.ones(3,4,dtype=torch.bool),z(3,2,11),torch.ones(3,2,dtype=torch.bool),
      torch.ones(3,10,2),z(3,8),torch.ones(3,dtype=torch.bool),z(3,6),torch.ones(3,dtype=torch.bool),
      torch.randn(3,5,768),torch.tensor([[0,0,1,2,3]]*3),torch.ones(3,5,dtype=torch.bool))


def test_scalar_units_do_not_parse_ranges_or_concentrations():
    assert scalar_dose('500','mcg')==(0.5,1)
    assert scalar_dose('2','puffs')==(2.,2)
    assert scalar_dose('1-2','mg') is None
    assert scalar_dose('5','mg/mL') is None


def test_scale_fits_train_only_and_separates_units():
    raw=np.array([[[1.,2.]],[[2.,3.]],[[9999.,9999.]]],np.float32)
    mask=np.ones_like(raw,bool);units=np.ones((3,1),int)
    a,p=scale_per_unit(raw,mask,units,np.array([True,True,False]),['a'])
    assert p['a/0/1']['median']<2
    assert np.isfinite(a).all() and a[2,0,0]==8


def test_epoch_zero_matches_baseline_and_frozen_parameters():
    base=HistoryBaseline();m=AuditedModel(copy.deepcopy(base),[0,0,1,2],np.eye(2,10),mode='interaction_cls').eval()
    h=torch.zeros(3,6);out=m(h,batch());expected=base(h)
    assert torch.equal(out.log_amplitude,expected.log_amplitude)
    assert torch.equal(out.shape,expected.shape)
    assert not any(p.requires_grad for p in m.baseline.parameters())


def test_no_early_pooling_keeps_all_tokens_and_permutation_invariance():
    m=AuditedEncoder([0,0,1,2],np.eye(2,10),mode='interaction_cls').eval();b=batch()
    x,p=m.tokens(b)
    assert x.shape==(3,30,128)  # 4labs+2drug+10parent+7labparent+2context+5text
    permutation=torch.tensor([4,1,3,0,2]);alt=replace(b,text=b.text[:,permutation],fields=b.fields[:,permutation],text_present=b.text_present[:,permutation])
    assert torch.allclose(m(b)[0],m(alt)[0],atol=2e-6)


def test_empty_clinical_is_finite_and_flat_matches_width():
    b=batch();b=replace(b,lab_present=b.lab_present&False,med_present=b.med_present&False,coarse=b.coarse*0,
      micro_present=b.micro_present&False,support_present=b.support_present&False,text_present=b.text_present&False)
    base=HistoryBaseline()
    for mode in ('pool','interaction_pool','interaction_cls','flat'):
        m=AuditedModel(copy.deepcopy(base),[0,0,1,2],np.eye(2,10),mode=mode).eval()
        assert torch.isfinite(m(torch.zeros(3,6),b).log_amplitude).all()
    assert flat_features(b).shape==(3,4*25+2*12+20+8+6+4*768+4)


def test_tree_cif_competition_and_mass_conservation():
    from training.v6.audited_simple_controls import tree_curve
    class ConstantRisk:
        def predict(self,x,output_margin=False):return np.zeros(len(x))
    models=[ConstantRisk(),ConstantRisk()];x=np.zeros((2,3))
    no_death=[(np.array([10.]),np.array([.2])),(np.array([]),np.array([]))]
    with_death=[no_death[0],(np.array([5.]),np.array([.4]))]
    a=tree_curve(models,no_death,x);b=tree_curve(models,with_death,x)
    assert np.allclose(b.sum(2),1,atol=1e-12)
    assert np.all(np.diff(b[:,:,1:],axis=1)>=-1e-12)
    assert np.all(b[:,90,1]<a[:,90,1])
    assert np.all(b[:,0,0]==1) and np.all(b[:,0,1:]==0)


def test_breslow_competing_patient_leaves_risk_set_at_death():
    from training.v6.audited_simple_controls import breslow
    t=np.array([2.,5.,10.]);e=np.array([2,1,0]);risk=np.array([100.,1.,1.])
    times,increments=breslow(t,e,risk,1)
    assert np.array_equal(times,[5.]) and np.allclose(increments,[.5])
