"""当前主模型的直接条件区间与开放尾部输出层。

本模块只复用住院内多模态编码器和患者纵向编码器。旧版 365 天逐日
竞争风险头会被显式移除，避免未使用参数进入优化器或新检查点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .aligned_model import AlignedModelConfig
from .longitudinal_model import LongitudinalAecopdRiskModel


@dataclass(frozen=True)
class ConditionalIntervalOutput:
    """一次前向传播的四任务输出及共享患者纵向表示。

    ``interval_*`` 形状为 ``[住院数, 3 个区间, 3 个类别]``；
    ``tail_*`` 形状为 ``[住院数, 3 个类别]``。
    """

    interval_logits: Tensor
    interval_probabilities: Tensor
    tail_logits: Tensor
    tail_probabilities: Tensor
    representation: Tensor


class ConditionalIntervalAecopdModel(nn.Module):
    """带直接条件区间头和开放尾部头的患者纵向模型。"""

    def __init__(
        self,
        admission_config: AlignedModelConfig,
        *,
        longitudinal_layers: int = 1,
        max_admissions: int = 128,
        disable_longitudinal: bool = False,
    ) -> None:
        super().__init__()
        # disable_longitudinal 为 B3 消融开关（opt-in，默认 False）：透传给编码器，
        # 开启后每次住院独立编码、不进入患者因果序列；区间/尾部头保持不变。
        self.disable_longitudinal = disable_longitudinal
        self.encoder = LongitudinalAecopdRiskModel(
            admission_config,
            admission_layers=longitudinal_layers,
            max_admissions=max_admissions,
            disable_longitudinal=disable_longitudinal,
        )
        # 当前架构只复用旧模型的住院内表示。显式注销所有逐日输出模块，
        # 防止它们作为“未使用但仍可训练”的参数进入优化器或新检查点。
        admission_model = self.encoder.admission_model
        admission_model.risk_head = None
        admission_model.aecopd_representation = None
        admission_model.death_representation = None
        admission_model.no_event_head = None
        admission_model.aecopd_head = None
        admission_model.death_head = None

        hidden = admission_config.hidden_size
        self.interval_embedding = nn.Embedding(3, hidden)
        self.interval_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(admission_config.dropout),
            nn.Linear(hidden, 3),
        )
        self.tail_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(admission_config.dropout),
            nn.Linear(hidden, 3),
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        """以“无事件”占优初始化区间头，尾部头保持均匀初始化。

        区间偏置 [2, 0, 0] 的 softmax 约为 [0.787, 0.107, 0.107]。
        这会降低类别不平衡数据上的初始交叉熵，但不代表模型已学到特征。
        """
        nn.init.normal_(self.interval_embedding.weight, mean=0.0, std=0.02)
        interval_output = self.interval_head[-1]
        tail_output = self.tail_head[-1]
        assert isinstance(interval_output, nn.Linear)
        assert isinstance(tail_output, nn.Linear)
        nn.init.normal_(interval_output.weight, mean=0.0, std=1e-3)
        nn.init.normal_(tail_output.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            interval_output.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
            tail_output.bias.zero_()

    def forward(self, batch: Mapping[str, Tensor]) -> ConditionalIntervalOutput:
        """编码每次住院并同时输出三个区间任务和一个尾部任务。"""
        representation = self.encoder.encode_patient_admissions(batch)
        # 三个区间共享预测网络，但各自加入可训练区间嵌入，以表达时间段差异。
        interval_ids = torch.arange(3, device=representation.device)
        interval_values = representation.unsqueeze(1) + self.interval_embedding(interval_ids).unsqueeze(0)
        interval_logits = self.interval_head(interval_values)
        tail_logits = self.tail_head(representation)
        return ConditionalIntervalOutput(
            interval_logits=interval_logits,
            interval_probabilities=interval_logits.softmax(dim=-1),
            tail_logits=tail_logits,
            tail_probabilities=tail_logits.softmax(dim=-1),
            representation=representation,
        )
