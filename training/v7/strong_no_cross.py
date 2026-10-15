"""冻结完整 V6，仅关闭 V7 药物/影像跨源读取；不改历史模型。"""
from training.v7.strong_anchor_experiment import StrongV7
from training.v7.sequential_variants import Variant


class StrongV7NoCross(StrongV7):
    def __init__(self, base, parent, anchors, seed=42):
        super().__init__(base, parent, anchors, seed=seed)
        # 父类内部固定了变体；须同时更换实际前向开关和优化器可见参数。
        self.variant = Variant('strong_anchor_no_cross', cross=False)
        self.drug_cross.requires_grad_(False)
        self.rad_cross.requires_grad_(False)
