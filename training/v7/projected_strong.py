"""完整强V6冻结、V7端到端修正，加入每轮训练集固定岭去重。"""
import numpy as np
import torch
from training.v7.strong_anchor_experiment import StrongV7


def ridge_operator(context):
    """仅传入训练患者H34；截距不惩罚，岭系数固定1。"""
    if context.ndim != 2 or context.shape[1] != 34 or not len(context) or not np.isfinite(context).all():
        raise ValueError('finite nonempty training H34 required')
    design = np.column_stack((np.ones(len(context)), context)).astype(np.float64)
    penalty = np.eye(35); penalty[0, 0] = 0
    return design, np.linalg.solve(design.T @ design + penalty, design.T)


def fit_coefficients(operator, hidden):
    if hidden.ndim != 2 or hidden.shape != (operator.shape[1], 256) or not np.isfinite(hidden).all():
        raise ValueError('training hidden rows do not match fixed H34 operator')
    return (operator @ hidden.astype(np.float64)).astype(np.float32)


class ProjectedStrongV7(StrongV7):
    def __init__(self, base, parent, anchors, seed=42, *, enabled=True):
        super().__init__(base, parent, anchors, seed)
        self.register_buffer('projection_coefficients', torch.zeros(35, 256))
        self.projection_enabled = enabled
        self._capture = False
        self._captured = None
        self._current_context = None
        self.delta_head.register_forward_pre_hook(self._project_read)

    def _project_read(self, module, arguments):
        hidden = arguments[0]
        if self._capture:
            self._captured = hidden.detach()
            return None
        if not self.projection_enabled:
            return None
        context = self._current_context
        design = torch.cat((torch.ones_like(context[:, :1]), context), 1)
        residual = hidden - design @ self.projection_coefficients
        return (residual, *arguments[1:])

    def forward(self, batch):
        self._current_context = batch.original.context[batch.original.current]
        return super().forward(batch)

    @torch.no_grad()
    def representations(self, batch):
        if self.training:
            raise ValueError('projection fits require deterministic evaluation mode')
        self._capture = True
        try:
            self(batch)
            hidden = self._captured
            if hidden is None:
                raise RuntimeError('risk representation was not consumed')
            return hidden
        finally:
            self._capture = False
            self._captured = None


@torch.no_grad()
def estimate_projection(model, batches, operator, design):
    model.eval()
    hidden = np.concatenate([model.representations(b.to('cuda')).cpu().numpy() for _, b in batches])
    coefficients = fit_coefficients(operator, hidden)
    residual = hidden.astype(np.float64) - design @ coefficients.astype(np.float64)
    diagnostics = dict(training_rows=len(hidden), hidden_mean_square=float(np.mean(hidden.astype(np.float64)**2)),
        residual_mean_square=float(np.mean(residual**2)),
        coefficient_l2=float(np.linalg.norm(coefficients)),
        residual_cross_l2=float(np.linalg.norm(design.T @ residual)/len(hidden)))
    return coefficients, diagnostics


@torch.no_grad()
def refresh_projection(model, batches, operator, design):
    coefficients, diagnostics = estimate_projection(model, batches, operator, design)
    model.projection_coefficients.copy_(torch.as_tensor(coefficients, device=model.projection_coefficients.device))
    return diagnostics
