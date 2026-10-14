"""B3：索引住院相互独立的分层模型（直接架构对照，脚手架）。

设计对齐 docs/BASELINE_COMPARISON_PLAN.md B3。这是与主模型 M1 最"贴身"的
对照：**保留完全相同的住院内多模态编码（aligned_model）与相同的章节向量、
相同的条件区间/尾部头，唯一去掉患者纵向因果 Transformer**——每次索引住院
独立产生损失，不加入跨住院因果序列。

因此 B3 与 M1 的差异集中在两点：
1. 是否有患者纵向建模（longitudinal_model 的 causal Transformer）；
2. 患者等权损失是否还需要（独立住院时可退化为住院等权，需在报告中说明选择）。

它回答的问题："把同一患者多次住院串成因果序列，到底额外贡献多少？"

复用策略（尽量不新写模型）：
- 直接用 ConditionalIntervalAecopdModel，但令 longitudinal 编码变为恒等/单住院，
  或用 aligned_model.TimeAlignedAecopdRiskModel.encode_admission 得到住院状态后
  直接接区间/尾部头，绕过 longitudinal_model 的序列重建。
- 标签、损失、评估全部复用主项目模块，保证唯一变量是"有无纵向"。

当前状态：脚手架，不训练。它需要训练（是唯一需要 GPU 的基线），因此在获得
研究者明确训练授权前，run_b3 只定义流程不执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .shared_contract import BaselineEvalResult, SplitContract


@dataclass(frozen=True)
class NoLongitudinalConfig:
    """B3 配置。除"去纵向"外，其余超参必须与主模型 h256 正式运行一致，确保可比。"""

    artifacts_path: Path
    section_embedding_dir: Path
    split: SplitContract
    # 与主模型对齐的超参（来自 run_state.json config）
    hidden_size: int = 256
    attention_heads: int = 4
    temporal_layers: int = 2
    within_window_layers: int = 1
    text_layers: int = 1
    fusion_layers: int = 1
    dropout: float = 0.1
    # 损失口径：独立住院下的两个候选，需在报告中固定其一
    loss_reduction: str = "admission_equal"   # 备选: admission_equal / patient_equal
    # 训练控制（与主模型一致，便于公平对照）
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 12
    selection_metric: str = "aecopd_interval_mean_brier"
    seed: int = 2026


def build_admission_only_model(config: NoLongitudinalConfig):
    """构造"住院内编码 + 区间/尾部头"的模型，不含患者纵向序列。

    实现建议：复用 aligned_model.TimeAlignedAecopdRiskModel.encode_admission 得到
    [admissions, hidden]，再接 conditional_model 的 interval_head / tail_head。
    不要重写编码器，只是断开 longitudinal_model 的序列重建这一段。
    """
    raise NotImplementedError("scaffold: 复用住院内编码器 + 条件区间/尾部头，去掉纵向序列")


def run_b3(config: NoLongitudinalConfig) -> BaselineEvalResult:
    """B3 端到端入口。

    ⚠ 需要训练（GPU）。在研究者明确授权前不执行，仅定义流程：
    build model -> 训练(validation 早停) -> 在 best 检查点评估 test 一次 ->
    shared_contract.evaluate_with_project_metrics。
    """
    raise NotImplementedError(
        "scaffold: B3 需要训练；获得授权后再执行，评估复用 conditional_metrics"
    )
