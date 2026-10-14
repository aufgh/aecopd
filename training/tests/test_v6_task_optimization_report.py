import copy
import unittest
import numpy as np
from training.v6.task_optimization_report import averages
from training.v6.task_optimization_experiment import SEEDS
from training.v6.task_optimization_diagnostics import analyze


class Tests(unittest.TestCase):
    def test_unavailable_auc_not_filled_with_zero(self):
        points={str(d):dict(auc=.8,brier=.1,mean_predicted=.2,observed_world_a_fraction=.2) for d in (30,90,180,365)}
        data={str(s):dict(control=dict(metrics={c:dict(integrated_brier_7_365=.1,points=copy.deepcopy(points))
              for c in ('readmission','death')})) for s in SEEDS}
        data[str(SEEDS[0])]['control']['metrics']['readmission']['points']['30']['auc']=None
        result=averages(data,['control'])['control']['readmission']
        self.assertIsNone(result['points']['30']['auc']);self.assertAlmostEqual(result['points']['90']['auc'],.8)

    def test_sparse_strata_preserve_metrics_and_report_no_auc_evidence(self):
        n=12;t=np.full(n,365.);e=np.zeros(n,dtype=int);t[:3]=20.;e[:3]=1
        days=np.arange(366)[None,:,None];survival=np.exp(-days*.002)
        curve=np.repeat(np.concatenate([survival,(1-survival)*.7,(1-survival)*.3],2),n,0)
        models,counts=analyze(t,e,np.arange(n),np.array([0]*4+[1]*4+[4]*4),
                            dict(control=curve,official_V6=curve,candidate=curve),seed=42)
        self.assertEqual(counts,dict(prior0=4,prior1_2=4,prior3plus=4))
        self.assertIsNone(models['candidate']['strata']['prior0']['points']['90']['auc'])
        self.assertIn('insufficient',models['candidate']['paired_strata']['control']['prior0']['status'])
        self.assertIn('2',models['candidate']['calibration'])

    def test_extra_baseline_is_a_comparator_not_clinical_candidate(self):
        n=12;t=np.full(n,365.);e=np.zeros(n,dtype=int);t[:3]=20.;e[:3]=1
        days=np.arange(366)[None,:,None];s=np.exp(-days*.002)
        curve=np.repeat(np.concatenate([s,(1-s)*.7,(1-s)*.3],2),n,0)
        names=('control','official_V6','baseline_only','candidate')
        models,_=analyze(t,e,np.arange(n),np.zeros(n,dtype=int),{name:curve for name in names},seed=42,
                         comparators=('control','baseline_only','official_V6'))
        self.assertNotIn('paired_strata',models['baseline_only'])
        self.assertEqual(set(models['candidate']['paired_strata']),{'control','baseline_only','official_V6'})


if __name__=='__main__':unittest.main()
