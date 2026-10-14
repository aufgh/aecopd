"""新增交互层：缺失/空输入、文本排列、共同初值、共享深度和真实梯度。"""
import copy
from dataclasses import replace
import unittest
import torch
from training.tests.test_v6_module_surgery import sample
from training.v6.model import HistoryBaseline
from training.v6.deep_interaction import CANDIDATES, DeepModel, gradient_groups
from training.v6.risk import competing_nll


def example():
    x = sample()
    c = x.base.base
    clinical = replace(c.clinical, current_values=c.clinical.current_values*0,
                       current_masks=c.clinical.current_masks & False)
    labs = x.base.labs.clone()
    labs[:, :50].reshape(3, 5, 10)[:, :, 5:] = 1
    labs[:, 50:].reshape(3, 2, 20)[:, :, 6:12] = 1
    return replace(x, base=replace(x.base, base=replace(c, clinical=clinical), labs=labs))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)
        self.base, self.batch = HistoryBaseline(), example()

    def model(self, arm):
        return DeepModel(self.base, 42, arm, text_width=8, width=32)

    def test_common_initialization_and_exact_zero_start(self):
        ref = self.model('late_single')
        history = torch.randn(3, 6)
        for arm in CANDIDATES:
            m = self.model(arm).eval()
            for part in ('tokens', 'score', 'output'):
                for k, v in getattr(ref.encoder, part).state_dict().items():
                    torch.testing.assert_close(v, getattr(m.encoder, part).state_dict()[k], atol=0, rtol=0)
            a, b = m(history, self.batch), m.baseline(history)
            torch.testing.assert_close(a.log_amplitude, b.log_amplitude, atol=0, rtol=0)
            torch.testing.assert_close(a.shape, b.shape, atol=0, rtol=0)

    def test_all_branches_and_interaction_have_gradient(self):
        for arm in ('late_single', 'A_deep', 'B_shared', 'C_latent'):
            m = self.model(arm)
            torch.nn.init.normal_(m.delta_head[-1].weight, std=.03)
            competing_nll(m(torch.randn(3, 6), self.batch), torch.tensor([10., 70., 365.]), torch.tensor([1, 2, 0])).backward()
            self.assertTrue(all(v > 0 for v in gradient_groups(m).values()), (arm, gradient_groups(m)))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.baseline.parameters()))

    def test_zero_gate_blocks_interaction_not_direct_content(self):
        m = self.model('late_gated')
        torch.nn.init.normal_(m.delta_head[-1].weight, std=.03)
        competing_nll(m(torch.randn(3, 6), self.batch), torch.tensor([10., 70., 365.]), torch.tensor([1, 2, 0])).backward()
        self.assertEqual(gradient_groups(m)['interaction'], 0.)
        self.assertGreater(gradient_groups(m)['text'], 0.)
        self.assertGreater(float(m.encoder.gate.grad.abs()), 0.)

    def test_no_text_chunk_order_is_invented(self):
        c = self.batch.base.base.clinical
        order = torch.arange(len(c.snippet_index)-1, -1, -1)
        changed = replace(c, snippet_vectors=c.snippet_vectors[order], snippet_index=c.snippet_index[order])
        other = replace(self.batch, base=replace(self.batch.base, base=replace(self.batch.base.base, clinical=changed)))
        for arm in CANDIDATES:
            m = self.model(arm).eval()
            a, _ = m.encoder(self.batch); b, _ = m.encoder(other)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)

    def test_text_and_lab_content_change_latent(self):
        c = self.batch.base.base.clinical
        vectors = c.snippet_vectors.clone(); vectors[:, 0] += 2
        other = replace(self.batch, base=replace(self.batch.base, base=replace(self.batch.base.base,
            clinical=replace(c, snippet_vectors=vectors))))
        for arm in CANDIDATES:
            m = self.model(arm).eval()
            a, _ = m.encoder(self.batch); b, _ = m.encoder(other)
            self.assertGreater(float((a-b).detach().abs().max()), 1e-5)

    def test_absent_values_ignored_and_empty_stable(self):
        c = self.batch.base.base
        clinical = replace(c.clinical, lab_axis_inputs=c.clinical.lab_axis_inputs*0,
                           snippet_vectors=c.clinical.snippet_vectors[:0], snippet_index=c.clinical.snippet_index[:0])
        empty = replace(c, clinical=clinical, med_features=c.med_features*0, micro_values=c.micro_values*0,
                        micro_masks=c.micro_masks&False, support_values=c.support_values*0, support_masks=c.support_masks&False)
        d = replace(self.batch.base, base=empty, labs=self.batch.base.labs*0)
        a = replace(self.batch, base=d)
        junk = clinical.lab_axis_inputs.clone(); junk[..., :6] = float('nan')
        b = replace(a, base=replace(d, base=replace(empty, clinical=replace(clinical, lab_axis_inputs=junk))))
        for arm in CANDIDATES:
            m = self.model(arm).eval()
            x, _ = m.encoder(a); y, _ = m.encoder(b)
            self.assertTrue(torch.isfinite(x).all())
            torch.testing.assert_close(x, y, atol=0, rtol=0)
            self.assertEqual(float(x.detach().abs().sum()), 0.)

    def test_shared_depth_does_not_duplicate_parameters(self):
        single, shared, deep = [self.model(a) for a in ('late_single', 'B_shared', 'A_deep')]
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(single), count(shared))
        self.assertGreater(count(deep), count(shared))

    def test_rejects_history_reentry(self):
        c = self.batch.base.base
        wrong = replace(self.batch, base=replace(self.batch.base, base=replace(c,
            clinical=replace(c.clinical, current_masks=torch.ones_like(c.clinical.current_masks)))))
        with self.assertRaises(ValueError):
            self.model('A_deep').encoder(wrong)

    def test_gpu_forward_backward_and_padding(self):
        if not torch.cuda.is_available():
            self.skipTest('GPU runtime not available')
        from dataclasses import fields, is_dataclass
        def cuda(x):
            if isinstance(x, torch.Tensor):
                return x.cuda()
            if is_dataclass(x):
                return replace(x, **{f.name:cuda(getattr(x,f.name)) for f in fields(x)})
            return x
        b = cuda(self.batch)
        for arm in CANDIDATES:
            m = self.model(arm).cuda().eval()
            torch.nn.init.normal_(m.delta_head[-1].weight, std=.03)
            out = m(torch.randn(3, 6, device='cuda'), b)
            loss = competing_nll(out, torch.tensor([10.,70.,365.],device='cuda'),torch.tensor([1,2,0],device='cuda'))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(gradient_groups(m)['text'],0.)


if __name__ == '__main__':
    unittest.main()
