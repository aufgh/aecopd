"""合成输入：新增分支确实可影响表示/获得梯度；删组不改变其他输入。"""
import copy
from dataclasses import replace
import unittest
import torch
from training.tests.test_v6_core import batch
from training.v6.compact_context_data import CompactContextBatch
from training.v6.compact_model import ARMS, CompactAnchoredModel, mask_group
from training.v6.model import HistoryBaseline
from training.v6.risk import competing_nll


def compact(device='cpu'):
    clinical = batch(device=device)
    med = torch.zeros(3, 7, 4, device=device); med[:, :, 0] = 1; med[:, :, 2:] = 1
    return CompactContextBatch(clinical, med, torch.ones(3, 5, device=device),
                               torch.ones(3, 5, dtype=torch.bool, device=device), torch.ones(3, 3, device=device),
                               torch.ones(3, 3, dtype=torch.bool, device=device))


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_rejects_legacy_batch(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16)
        with self.assertRaises(TypeError): model.encoder(batch())

    def test_zero_start_and_frozen_history(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16)
        h = torch.randn(3, 6)
        expected, actual = model.baseline(h), model(h, compact())
        torch.testing.assert_close(expected.log_amplitude, actual.log_amplitude, rtol=0, atol=0)
        torch.testing.assert_close(expected.shape, actual.shape, rtol=0, atol=0)
        competing_nll(actual, torch.tensor([5., 50., 365.]), torch.tensor([1, 2, 0])).backward()
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))

    def test_new_content_not_only_presence_changes_representation(self):
        torch.manual_seed(5)
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16).eval()
        b = compact(); expected, _ = model.encoder(b)
        changed = []
        med = b.med_features.clone(); med[:, :, 0] = 0; med[:, :, 1] = 1
        changed.append(replace(b, med_features=med))
        changed.append(replace(b, micro_values=torch.zeros_like(b.micro_values)))
        changed.append(replace(b, support_values=torch.zeros_like(b.support_values)))
        for other in changed:
            actual, _ = model.encoder(other)
            self.assertGreater((actual - expected).abs().max().item(), 1e-5)

    def test_all_new_branches_receive_gradient_after_head_unblocks(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16)
        torch.nn.init.normal_(model.delta_head[-1].weight, std=.1)
        output = model(torch.randn(3, 6), compact())
        competing_nll(output, torch.tensor([5., 50., 365.]), torch.tensor([1, 2, 0])).backward()
        for branch in ('lab', 'med', 'microbiology', 'support'):
            self.assertGreater(getattr(model.encoder, branch)[0].weight.grad.abs().sum().item(), 0)

    def test_masks_remove_entire_group_and_preserve_other_branches(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16).eval()
        b = compact(); _, reference = model.encoder(b)
        keys = dict(without_medications='medication', without_microbiology='microbiology', without_support='support', without_labs='lab')
        for arm in ARMS[1:]:
            omitted = mask_group(b, arm); _, result = model.encoder(omitted)
            self.assertEqual(result[keys[arm]].abs().sum().item(), 0)
            for key in ('structured', 'lab', 'medication', 'microbiology', 'support', 'text'):
                if key != keys[arm]: torch.testing.assert_close(result[key], reference[key], rtol=0, atol=0)
            # 仅改变被删组原值；再次屏蔽后预测表示应完全不变。
            if arm == 'without_labs':
                changed = replace(b, clinical=replace(b.clinical, lab_axis_inputs=b.clinical.lab_axis_inputs + 10))
            elif arm == 'without_medications':
                changed = replace(b, med_features=torch.zeros_like(b.med_features))
            elif arm == 'without_microbiology':
                changed = replace(b, micro_values=torch.zeros_like(b.micro_values), micro_masks=torch.zeros_like(b.micro_masks))
            else:
                changed = replace(b, support_values=torch.zeros_like(b.support_values), support_masks=torch.zeros_like(b.support_masks))
            a, _ = model.encoder(omitted); c, _ = model.encoder(mask_group(changed, arm))
            torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_all_missing_and_learned_norm_bias_stay_zero(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16).eval()
        for norm in model.encoder.branch_norms: torch.nn.init.constant_(norm.bias, 1.)
        b = compact()
        for arm in ARMS[1:]: b = mask_group(b, arm)
        clinical = replace(b.clinical, current_masks=torch.zeros_like(b.clinical.current_masks),
                           snippet_vectors=b.clinical.snippet_vectors[:0], snippet_index=b.clinical.snippet_index[:0])
        hidden, branches = model.encoder(replace(b, clinical=clinical))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue(all(v.abs().sum() == 0 for v in branches.values()))

    def test_nontext_initialization_invariant_and_arm_capacity(self):
        states = []
        for dim in (8, 20):
            torch.manual_seed(42)
            model = CompactAnchoredModel(HistoryBaseline(), dim, width=16)
            states.append({k: v for k, v in model.state_dict().items() if 'text_projection' not in k})
        for k in states[0]: torch.testing.assert_close(states[0][k], states[1][k], rtol=0, atol=0)
        torch.manual_seed(42); model = CompactAnchoredModel(HistoryBaseline(), 8, width=16)
        before = copy.deepcopy(model.state_dict())
        for arm in ARMS: mask_group(compact(), arm)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_invalid_sources_and_binary_values(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16)
        b = compact(); b.med_features[:, :, 2] = 0
        with self.assertRaises(ValueError): model.encoder(b)
        b = compact(); b.micro_values[0, 0] = .5
        with self.assertRaises(ValueError): model.encoder(b)

    @unittest.skipUnless(torch.cuda.is_available(), 'GPU unavailable')
    def test_cuda_forward_backward(self):
        model = CompactAnchoredModel(HistoryBaseline(), 8, width=16).cuda()
        output = model(torch.zeros(3, 6, device='cuda'), compact('cuda'))
        loss = competing_nll(output, torch.tensor([5., 50., 365.], device='cuda'), torch.tensor([1, 2, 0], device='cuda'))
        loss.backward(); self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__': unittest.main()
