import unittest
from dataclasses import replace
import torch
from training.v6.curve_adaptation import Correction,FrozenBatch,objective
from training.v6.risk import RiskOutput
from training.v6.curve_score import daily_cif
from training.v6.curve_adaptation_audit import choose,NAMES


class Tests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(12)
        return FrozenBatch(torch.randn(5,32),torch.randn(5,4)*.02,torch.randn(5,2)-2,
                           torch.tensor([-.4,.2]),torch.tensor([0.,30.,100.,365.,365.]),torch.tensor([1,2,1,0,0]))

    def test_exact_teacher_restore_all_arms(self):
        b=self.batch();r=b.raw.reshape(-1,2,2)
        expected=RiskOutput((b.prior+r[:,:,0]).clamp(-10,6),torch.nn.functional.softplus(b.shape_raw+r[:,:,1])+.1)
        for arm,n in [('clinical_nll',132),('clinical_curve',132),('global_curve',4)]:
            m=Correction(arm);out=m(b)
            self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            torch.testing.assert_close(out.log_amplitude,expected.log_amplitude,atol=0,rtol=0)
            torch.testing.assert_close(out.shape,expected.shape,atol=0,rtol=0)
            torch.testing.assert_close(daily_cif(out),daily_cif(expected),atol=0,rtol=0)

    def test_global_does_not_read_current_hidden(self):
        b=self.batch();m=Correction('global_curve')
        with torch.no_grad():m.global_raw.add_(.2)
        a=m(b);c=m(replace(b,hidden=torch.full_like(b.hidden,999)))
        torch.testing.assert_close(a.log_amplitude,c.log_amplitude,atol=0,rtol=0)
        torch.testing.assert_close(a.shape,c.shape,atol=0,rtol=0)

    def test_finite_first_steps_and_no_source_mutation(self):
        b=self.batch();original=[v.clone() for v in (b.hidden,b.raw,b.prior,b.shape_raw)]
        for arm in ('clinical_nll','clinical_curve','global_curve'):
            m=Correction(arm);opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=1e-4)
            loss=objective(m,b);loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            opt.step();self.assertTrue(any(torch.count_nonzero(p) for p in m.parameters()))
        for a,c in zip(original,(b.hidden,b.raw,b.prior,b.shape_raw)):torch.testing.assert_close(a,c,atol=0,rtol=0)

    def test_nonclinical_and_tied_winners_never_nominate(self):
        scores={n:.1 for n in reversed(NAMES)}
        self.assertEqual(choose(scores),('original_teacher',[]))
        scores['global_curve']=.09
        self.assertEqual(choose(scores),('global_curve',[]))
        scores['clinical_curve']=.08
        self.assertEqual(choose(scores),('clinical_curve',['clinical_curve']))
        scores['clinical_nll']=float('nan')
        with self.assertRaises(ValueError):choose(scores)


if __name__=='__main__':unittest.main()
