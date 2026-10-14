import unittest
from training.v6.task_optimization_experiment import SEEDS
from training.v6.task_optimization_finish import material_signal


class Tests(unittest.TestCase):
    def models(self,gain=.01,ibs=.098):
        def score(auc,brier):return dict(metrics=dict(readmission=dict(points={'90':dict(auc=auc)},integrated_brier_7_365=brier),death=dict(integrated_brier_7_365=.10)))
        return {str(s):dict(control=score(.8,.10),official_V6=score(.8,.10),candidate=score(.8+gain,ibs)) for s in SEEDS}

    def test_tiny_gain_is_not_material_and_never_goal_complete(self):
        r=material_signal(self.models(gain=.00004,ibs=.099999),'candidate')
        self.assertFalse(r['worth_five_seed_confirmation']);self.assertFalse(r['goal_achieved'])

    def test_material_signal_requires_both_references(self):
        m=self.models();self.assertTrue(material_signal(m,'candidate')['worth_five_seed_confirmation'])
        for s in SEEDS:m[str(s)]['official_V6']['metrics']['readmission']['points']['90']['auc']=.82
        self.assertFalse(material_signal(m,'candidate')['worth_five_seed_confirmation'])

    def test_context_candidate_must_also_beat_baseline_only(self):
        m=self.models()
        import copy
        for s in SEEDS:
            m[str(s)]['baseline_only']=copy.deepcopy(m[str(s)]['candidate'])
        result=material_signal(m,'candidate',comparators=('control','baseline_only','official_V6'))
        self.assertFalse(result['worth_five_seed_confirmation'])
        self.assertFalse(result['comparators']['baseline_only']['worth_five_seed_confirmation'])
