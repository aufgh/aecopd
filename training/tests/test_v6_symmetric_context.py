import unittest
import torch
from training.v6.symmetric_context import visit_tokens


class SymmetricContextTests(unittest.TestCase):
    def test_symmetric_current_and_past_same_background(self):
        cur = torch.tensor([[1., 2.]])
        bg = torch.tensor([[3., 4.]])
        result = visit_tokens(cur, cur[:, None], bg, bg[:, None], torch.zeros(1, 1, 2), symmetric=True)
        torch.testing.assert_close(result[:, 0], result[:, 1], rtol=0, atol=0)

    def test_asymmetric_current_not_changed(self):
        cur = torch.ones(2, 3); past = torch.ones(2, 4, 3)
        result = visit_tokens(cur, past, cur * 2, past * 3, past * 4, symmetric=False)
        torch.testing.assert_close(result[:, 0], cur, rtol=0, atol=0)
        torch.testing.assert_close(result[:, 1:], past * 8, rtol=0, atol=0)

    def test_only_current_token_changes(self):
        cur = torch.ones(1, 3); past = torch.ones(1, 2, 3)
        a = visit_tokens(cur, past, cur * 2, past * 3, past * 4, symmetric=False)
        b = visit_tokens(cur, past, cur * 2, past * 3, past * 4, symmetric=True)
        torch.testing.assert_close(a[:, 1:], b[:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(b[:, 0] - a[:, 0], cur * 2, rtol=0, atol=0)

    def test_current_background_has_gradient(self):
        cur = torch.zeros(1, 2); past = torch.zeros(1, 1, 2)
        bg = torch.ones(1, 2, requires_grad=True)
        visit_tokens(cur, past, bg, past, past, symmetric=True)[:, 0].sum().backward()
        torch.testing.assert_close(bg.grad, torch.ones_like(bg), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
