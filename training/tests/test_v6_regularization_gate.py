"""操作门槛的逻辑测试，不证明实际临床收益。"""
import copy
import unittest
from training.v6.regularization_multiseed import SEEDS
from training.v6.regularization_multiseed_verify import audit_gate


def scores(auc,ibs):
    return dict(readmission=dict(integrated_brier_7_365=ibs,
        points={str(day):dict(auc=auc,brier=.08) for day in (30,90,180,365)}),
        death=dict(integrated_brier_7_365=.09))


def models():
    return {str(s):dict(candidate=dict(metrics=scores(.80,.095)),v6_control=dict(metrics=scores(.79,.1)),
                       official_V6=dict(metrics=scores(.79,.1))) for s in SEEDS}


class Tests(unittest.TestCase):
    def test_pass_is_not_completion(self):
        x=audit_gate(models(),['candidate'])['candidate']
        self.assertTrue(x['basic_gate_passed'])
        self.assertFalse(x['stable_superiority_proven'])
        self.assertTrue(x['remaining_requirements'])

    def test_cannot_only_beat_weaker_retrained_control(self):
        data=models()
        for seed in SEEDS:data[str(seed)]['official_V6']['metrics']=scores(.81,.09)
        x=audit_gate(data,['candidate'])['candidate']
        self.assertTrue(x['comparisons']['v6_control']['predeclared_basic_gate_passed'])
        self.assertFalse(x['basic_gate_passed'])

    def test_missing_auc_does_not_become_gain(self):
        data=models();data['42']['candidate']['metrics']['readmission']['points']['90']['auc']=None
        x=audit_gate(data,['candidate'])['candidate']
        self.assertFalse(x['basic_gate_passed'])
        self.assertIsNone(x['comparisons']['official_V6']['mean_auc90_difference'])

    def test_large_death_or_timepoint_degradation_rejected(self):
        data=models()
        for s in SEEDS:data[str(s)]['candidate']['metrics']['death']['integrated_brier_7_365']=.10
        self.assertFalse(audit_gate(data,['candidate'])['candidate']['basic_gate_passed'])
        data=models()
        for s in SEEDS:data[str(s)]['candidate']['metrics']['readmission']['points']['365']['auc']=.75
        self.assertFalse(audit_gate(data,['candidate'])['candidate']['basic_gate_passed'])


if __name__=='__main__':unittest.main()
