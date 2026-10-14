"""Causal patient-level encoder over index-admission multimodal states."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .aligned_model import AlignedModelConfig, TimeAlignedAecopdRiskModel
from .model import ModelOutput


class LongitudinalAecopdRiskModel(nn.Module):
    """Predict every discharge using only that admission and earlier admissions.

    Admission inputs remain flattened for the existing multimodal collator. Four
    longitudinal tensors map those rows into padded patient sequences:

    - ``longitudinal_patient_indices``: patient row for every flattened admission;
    - ``longitudinal_position_indices``: zero-based causal position;
    - ``longitudinal_gap_days``: days since the previous discharge (zero at position 0);
    - ``longitudinal_attention_mask``: valid positions in ``[patients, admissions]``.
    """

    def __init__(
        self,
        admission_config: AlignedModelConfig,
        *,
        admission_layers: int = 2,
        max_admissions: int = 128,
        disable_longitudinal: bool = False,
    ) -> None:
        super().__init__()
        if admission_layers < 1:
            raise ValueError("admission_layers must be positive")
        if max_admissions < 1:
            raise ValueError("max_admissions must be positive")
        self.admission_model = TimeAlignedAecopdRiskModel(admission_config)
        self.max_admissions = max_admissions
        # B3 消融开关（opt-in，默认 False = 主模型 M1 行为不变）。开启后完全去掉
        # 患者纵向因果 Transformer：每次索引住院只用自身住院内多模态编码，
        # 不读取同一患者更早住院。为不让未用参数进入优化器/新检查点，开启时
        # 直接不构造位置嵌入/间隔编码/序列 Transformer 三个子模块。
        self.disable_longitudinal = disable_longitudinal
        hidden = admission_config.hidden_size
        if disable_longitudinal:
            self.position_embedding = None
            self.gap_encoder = None
            self.admission_encoder = None
            return
        self.position_embedding = nn.Embedding(max_admissions + 1, hidden, padding_idx=0)
        self.gap_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=admission_config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=admission_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.admission_encoder = nn.TransformerEncoder(
            layer,
            num_layers=admission_layers,
            norm=nn.LayerNorm(hidden),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        """为每次展平住院生成因果纵向状态，再调用旧兼容风险接口。"""
        contextual_states = self.encode_patient_admissions(batch)
        return self.admission_model.predict_from_representation(contextual_states)

    def encode_patient_admissions(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Return causal admission states without applying the legacy daily risk head."""
        admission_states = self.admission_model.encode_admission(batch)
        if self.disable_longitudinal:
            # 去纵向消融：直接返回逐住院编码，保持 [住院数, hidden] 形状与展平顺序
            # 不变（区间/尾部头与损失完全复用），唯一变量即"有无跨住院因果序列"。
            return admission_states
        return self._encode_patient_sequences(batch, admission_states)

    def _encode_patient_sequences(
        self,
        batch: Mapping[str, Tensor],
        admission_states: Tensor,
    ) -> Tensor:
        """把展平住院重建为患者序列并返回原展平顺序的因果状态。"""
        patient_indices = batch["longitudinal_patient_indices"].long()
        position_indices = batch["longitudinal_position_indices"].long()
        gap_days = batch["longitudinal_gap_days"].to(admission_states.dtype)
        attention = batch["longitudinal_attention_mask"].bool()
        admission_count = admission_states.shape[0]
        if patient_indices.shape != (admission_count,):
            raise ValueError("longitudinal_patient_indices must have shape [admissions]")
        if position_indices.shape != (admission_count,):
            raise ValueError("longitudinal_position_indices must have shape [admissions]")
        if gap_days.shape != (admission_count,):
            raise ValueError("longitudinal_gap_days must have shape [admissions]")
        if attention.ndim != 2 or not bool(attention.any(dim=1).all().item()):
            raise ValueError("every patient sequence must contain at least one admission")
        patient_count, sequence_width = attention.shape
        if sequence_width > self.max_admissions:
            raise ValueError(
                f"patient sequence width {sequence_width} exceeds max_admissions={self.max_admissions}"
            )
        if torch.any(patient_indices < 0) or torch.any(patient_indices >= patient_count):
            raise ValueError("longitudinal patient index is out of range")
        if torch.any(position_indices < 0) or torch.any(position_indices >= sequence_width):
            raise ValueError("longitudinal position index is out of range")
        if torch.any(~torch.isfinite(gap_days)) or torch.any(gap_days < 0):
            raise ValueError("longitudinal gap days must be finite and non-negative")
        linear_indices = patient_indices * sequence_width + position_indices
        if torch.unique(linear_indices).numel() != admission_count:
            raise ValueError("duplicate patient/position pair in longitudinal mapping")
        if attention.sum().item() != admission_count or not bool(
            attention[patient_indices, position_indices].all().item()
        ):
            raise ValueError("longitudinal attention mask does not match flattened admissions")

        # 住院内编码按住院展平执行以复用现有 collator；此处才按
        # patient_index/position_index 重建 [患者, 序列位置, hidden]。
        padded = admission_states.new_zeros((patient_count, sequence_width, admission_states.shape[1]))
        padded[patient_indices, position_indices] = admission_states
        gap_values = gap_days.new_zeros((patient_count, sequence_width))
        gap_values[patient_indices, position_indices] = gap_days
        gap_known = attention & (
            torch.arange(sequence_width, device=attention.device).unsqueeze(0) > 0
        )
        gap_features = torch.stack(
            [torch.log1p(gap_values), gap_known.to(gap_values.dtype)],
            dim=-1,
        )
        positions = torch.arange(1, sequence_width + 1, device=attention.device).unsqueeze(0)
        values = padded + self.position_embedding(positions) + self.gap_encoder(gap_features)
        # True 表示禁止注意；严格上三角确保第 k 次住院只能读取 1..k。
        causal_mask = torch.triu(
            torch.ones(
                (sequence_width, sequence_width),
                dtype=torch.bool,
                device=attention.device,
            ),
            diagonal=1,
        )
        encoded = self.admission_encoder(
            values,
            mask=causal_mask,
            src_key_padding_mask=~attention,
        )
        return encoded[patient_indices, position_indices]
