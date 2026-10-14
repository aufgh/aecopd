"""一次索引住院内的时间对齐多模态编码器。

所有输入分支最终都进入统一 ``hidden_size`` 空间；Qwen 的 2,560 维仅是
冻结源向量宽度，不会以 2,560 维直接与 25 通道结构化特征拼接竞争。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .model import ModelOutput, _competing_risk_curves, _masked_attention_pool


@dataclass(frozen=True)
class AlignedModelConfig:
    """住院内多模态编码器的词表大小、张量宽度和层数配置。"""

    concept_vocab_size: int
    domain_vocab_size: int
    window_vocab_size: int
    unit_vocab_size: int
    route_vocab_size: int
    evidence_vocab_size: int
    summary_vocab_size: int
    static_category_sizes: tuple[int, ...]
    timeline_feature_count: int = 25
    summary_feature_count: int = 11
    static_continuous_count: int = 16
    static_binary_count: int = 2
    reliable_window_count: int = 15
    hidden_size: int = 256
    attention_heads: int = 4
    within_window_layers: int = 1
    text_layers: int = 1
    fusion_layers: int = 1
    temporal_layers: int = 2
    patient_layers: int = 2
    dropout: float = 0.1
    horizon_days: int = 365
    use_text: bool = True
    text_embedding_dim: int = 2560
    text_role_vocab_size: int = 6
    text_section_role_vocab_size: int = 18
    text_modality_vocab_size: int = 9
    text_feature_count: int = 3
    text_chunk_position_vocab_size: int = 129
    text_section_position_vocab_size: int = 64
    text_section_instance_vocab_size: int = 32
    text_section_layers: int = 1
    cause_specific_heads: bool = False
    initial_readmission_logit: float = -6.5
    initial_death_logit: float = -7.0

    @classmethod
    def from_artifacts(cls, path: str | Path, **overrides: object) -> "AlignedModelConfig":
        artifacts = json.loads(Path(path).read_text(encoding="utf-8"))
        values: dict[str, object] = {
            "concept_vocab_size": len(artifacts["event_vocabulary"]),
            "domain_vocab_size": len(artifacts["event_domain_vocabulary"]),
            "window_vocab_size": 18,
            "unit_vocab_size": len(artifacts["event_unit_vocabulary"]),
            "route_vocab_size": len(artifacts["event_route_vocabulary"]),
            "evidence_vocab_size": len(artifacts["event_evidence_vocabulary"]),
            "summary_vocab_size": len(artifacts["numeric_summary_vocabulary"]),
            "static_category_sizes": tuple(
                len(value["vocabulary"])
                for value in artifacts["static_categorical"].values()
            ),
            **overrides,
        }
        return cls(**values)  # type: ignore[arg-type]


def _encoder(hidden: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden,
        nhead=heads,
        dim_feedforward=hidden * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(hidden))


def _group_and_pad(
    values: Tensor,
    mask: Tensor,
    group_ids: Tensor,
    group_count: int,
) -> tuple[Tensor, Tensor]:
    """Pack tokens into [batch, group, within-group position, hidden]."""
    sequences: list[Tensor] = []
    sequence_masks: list[Tensor] = []
    for batch_index in range(values.shape[0]):
        for group_index in range(1, group_count + 1):
            selected = mask[batch_index] & (group_ids[batch_index] == group_index)
            sequence = values[batch_index, selected]
            if sequence.shape[0] == 0:
                sequence = values.new_zeros((1, values.shape[-1]))
                sequence_mask = mask.new_zeros((1,))
            else:
                sequence_mask = mask.new_ones((sequence.shape[0],))
            sequences.append(sequence)
            sequence_masks.append(sequence_mask)
    packed = pad_sequence(sequences, batch_first=True)
    packed_mask = pad_sequence(sequence_masks, batch_first=True)
    return (
        packed.reshape(values.shape[0], group_count, packed.shape[1], values.shape[-1]),
        packed_mask.reshape(values.shape[0], group_count, packed.shape[1]),
    )


def _safe_transformer(encoder: nn.Module, values: Tensor, mask: Tensor) -> Tensor:
    safe_mask = mask.clone()
    empty = ~safe_mask.any(dim=1)
    safe_mask[empty, 0] = True
    return encoder(values, src_key_padding_mask=~safe_mask)


class TimeAlignedAecopdRiskModel(nn.Module):
    """Time-aligned structured/text model with a daily competing-risk head."""

    def __init__(self, config: AlignedModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if config.use_text and config.text_embedding_dim <= 0:
            raise ValueError("time-aligned model requires frozen text embeddings")
        self.config = config
        hidden = config.hidden_size

        self.concept_embedding = nn.Embedding(config.concept_vocab_size, hidden, padding_idx=0)
        self.domain_embedding = nn.Embedding(config.domain_vocab_size, hidden, padding_idx=0)
        self.window_embedding = nn.Embedding(config.window_vocab_size, hidden, padding_idx=0)
        self.unit_embedding = nn.Embedding(config.unit_vocab_size, hidden, padding_idx=0)
        self.route_embedding = nn.Embedding(config.route_vocab_size, hidden, padding_idx=0)
        self.evidence_embedding = nn.Embedding(config.evidence_vocab_size, hidden, padding_idx=0)
        self.encoder_flag_embedding = nn.Embedding(2, hidden)
        self.timeline_numeric = nn.Sequential(
            # 25 个统计量和 25 个可用性 mask 拼成 50 维，再投影到 hidden。
            nn.Linear(config.timeline_feature_count * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.timeline_norm = nn.LayerNorm(hidden)
        self.within_window_encoder = _encoder(
            hidden, config.attention_heads, config.within_window_layers, config.dropout
        )
        self.structured_scorer = nn.Linear(hidden, 1)

        # 冻结 Qwen 向量先统一投影到 hidden；后续融合不保留 2,560 维宽度优势。
        # use_text=False 时不构造任何文本参数，避免“零向量仍携带
        # 文档角色/位置嵌入”造成伪无文本对照。
        if config.use_text:
            self.text_projection: nn.Linear | None = nn.Linear(config.text_embedding_dim, hidden)
            self.text_role_embedding: nn.Embedding | None = nn.Embedding(
                config.text_role_vocab_size, hidden, padding_idx=0
            )
            self.text_section_role_embedding: nn.Embedding | None = nn.Embedding(
                config.text_section_role_vocab_size, hidden, padding_idx=0
            )
            self.text_modality_embedding: nn.Embedding | None = nn.Embedding(
                config.text_modality_vocab_size, hidden, padding_idx=0
            )
            self.text_metadata: nn.Module | None = nn.Sequential(
                nn.Linear(config.text_feature_count, hidden), nn.GELU()
            )
            self.text_chunk_position: nn.Embedding | None = nn.Embedding(
                config.text_chunk_position_vocab_size, hidden, padding_idx=0
            )
            self.text_section_position: nn.Embedding | None = nn.Embedding(
                config.text_section_position_vocab_size, hidden, padding_idx=0
            )
            self.text_section_instance: nn.Embedding | None = nn.Embedding(
                config.text_section_instance_vocab_size, hidden, padding_idx=0
            )
            self.text_norm: nn.LayerNorm | None = nn.LayerNorm(hidden)
            self.text_section_encoder: nn.Module | None = _encoder(
                hidden, config.attention_heads, config.text_section_layers, config.dropout
            )
            self.text_section_scorer: nn.Linear | None = nn.Linear(hidden, 1)
            self.text_encoder: nn.Module | None = _encoder(
                hidden, config.attention_heads, config.text_layers, config.dropout
            )
            self.text_scorer: nn.Linear | None = nn.Linear(hidden, 1)
        else:
            self.text_projection = None
            self.text_role_embedding = None
            self.text_section_role_embedding = None
            self.text_modality_embedding = None
            self.text_metadata = None
            self.text_chunk_position = None
            self.text_section_position = None
            self.text_section_instance = None
            self.text_norm = None
            self.text_section_encoder = None
            self.text_section_scorer = None
            self.text_encoder = None
            self.text_scorer = None

        self.day_fusion_encoder = _encoder(
            hidden, config.attention_heads, config.fusion_layers, config.dropout
        )
        self.day_fusion_scorer = nn.Linear(hidden, 1)
        self.temporal_encoder = _encoder(
            hidden, config.attention_heads, config.temporal_layers, config.dropout
        )

        self.summary_embedding = nn.Embedding(config.summary_vocab_size, hidden, padding_idx=0)
        self.summary_numeric = nn.Sequential(
            nn.Linear(config.summary_feature_count * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.summary_scorer = nn.Linear(hidden, 1)

        category_dim = max(4, hidden // 16)
        self.static_category_embeddings = nn.ModuleList(
            nn.Embedding(size, category_dim, padding_idx=0) for size in config.static_category_sizes
        )
        static_input = (
            config.static_continuous_count * 2
            + config.static_binary_count
            + category_dim * len(config.static_category_sizes)
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

        self.patient_cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.patient_type_embedding = nn.Embedding(7, hidden)
        self.patient_encoder = _encoder(
            hidden, config.attention_heads, config.patient_layers, config.dropout
        )
        if config.cause_specific_heads:
            self.aecopd_representation = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(config.dropout), nn.LayerNorm(hidden)
            )
            self.death_representation = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(config.dropout), nn.LayerNorm(hidden)
            )
            self.no_event_head = nn.Linear(hidden, config.horizon_days)
            self.aecopd_head = nn.Linear(hidden, config.horizon_days)
            self.death_head = nn.Linear(hidden, config.horizon_days)
            self.risk_head: nn.Module | None = None
        else:
            self.aecopd_representation = None
            self.death_representation = None
            self.no_event_head = None
            self.aecopd_head = None
            self.death_head = None
            self.risk_head = nn.Linear(hidden, config.horizon_days * 3)
        nn.init.normal_(self.patient_cls, mean=0.0, std=0.02)
        self._initialize_risk_head()

    def _initialize_risk_head(self) -> None:
        if self.config.cause_specific_heads:
            assert self.no_event_head is not None
            assert self.aecopd_head is not None
            assert self.death_head is not None
            for head in (self.no_event_head, self.aecopd_head, self.death_head):
                nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            with torch.no_grad():
                self.no_event_head.bias.zero_()
                self.aecopd_head.bias.fill_(self.config.initial_readmission_logit)
                self.death_head.bias.fill_(self.config.initial_death_logit)
            return
        assert self.risk_head is not None
        nn.init.normal_(self.risk_head.weight, mean=0.0, std=1e-3)
        daily_bias = self.risk_head.bias.view(self.config.horizon_days, 3)
        with torch.no_grad():
            daily_bias[:, 0].zero_()
            daily_bias[:, 1].fill_(self.config.initial_readmission_logit)
            daily_bias[:, 2].fill_(self.config.initial_death_logit)

    def forward(self, batch: Mapping[str, Tensor]) -> ModelOutput:
        return self.predict_from_representation(self.encode_admission(batch))

    def encode_admission(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Return one multimodal state per index admission before the risk head."""

        structured_tokens = self._structured_tokens(batch)
        text_documents: Tensor | None = None
        if self.config.use_text:
            text_sections = self._text_tokens(batch)
            text_documents = self._text_documents(batch, text_sections)
        trajectory, trajectory_present, retrospective, retrospective_present = self._trajectory(
            batch, structured_tokens, text_documents
        )
        if text_documents is None:
            discharge = structured_tokens.new_zeros(
                (structured_tokens.shape[0], self.config.hidden_size)
            )
            history = discharge.clone()
            discharge_present = torch.zeros(
                structured_tokens.shape[0], dtype=torch.bool, device=structured_tokens.device
            )
            history_present = discharge_present.clone()
        else:
            discharge, discharge_present = self._text_subset(
                batch, text_documents, batch["text_document_role_ids"] == 1
            )
            history, history_present = self._text_subset(
                batch,
                text_documents,
                (batch["text_document_role_ids"] == 3)
                | (batch["text_document_role_ids"] == 4),
            )
        summary, summary_present = self._summaries(batch)
        static = self._static(batch)

        batch_size = structured_tokens.shape[0]
        cls = self.patient_cls.expand(batch_size, -1, -1)
        branch_values = torch.stack(
            [trajectory, retrospective, discharge, history, summary, static], dim=1
        )
        # 六个分支均为 hidden 维，再与 CLS 组成 [B, 7, hidden]；这里不是
        # 将原始 2,560 维文本与 25/16 维结构化张量直接拼接。
        values = torch.cat([cls, branch_values], dim=1)
        type_ids = torch.arange(7, device=values.device).unsqueeze(0)
        values = values + self.patient_type_embedding(type_ids)
        mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=values.device),
                torch.stack(
                    [
                        trajectory_present,
                        retrospective_present,
                        discharge_present,
                        history_present,
                        summary_present,
                        torch.ones_like(trajectory_present),
                    ],
                    dim=1,
                ),
            ],
            dim=1,
        )
        return self.patient_encoder(values, src_key_padding_mask=~mask)[:, 0]

    def predict_from_representation(self, patient: Tensor) -> ModelOutput:
        """Apply the competing-risk head to admission or longitudinal states."""

        if patient.ndim != 2 or patient.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"patient representation must have shape [batch, {self.config.hidden_size}]"
            )
        batch_size = patient.shape[0]
        if self.config.cause_specific_heads:
            assert self.aecopd_representation is not None
            assert self.death_representation is not None
            assert self.no_event_head is not None
            assert self.aecopd_head is not None
            assert self.death_head is not None
            logits = torch.stack(
                [
                    self.no_event_head(patient),
                    self.aecopd_head(self.aecopd_representation(patient)),
                    self.death_head(self.death_representation(patient)),
                ],
                dim=-1,
            )
        else:
            assert self.risk_head is not None
            logits = self.risk_head(patient).reshape(batch_size, self.config.horizon_days, 3)
        probabilities, survival, incidence = _competing_risk_curves(logits)
        return ModelOutput(logits, probabilities, survival, incidence, patient)

    def _structured_tokens(self, batch: Mapping[str, Tensor]) -> Tensor:
        """把类别嵌入与连续统计/掩码相加为统一结构化 token。"""
        features = batch["timeline_features"]
        masks = batch["timeline_feature_masks"]
        return self.timeline_norm(
            self.concept_embedding(batch["timeline_concept_ids"])
            + self.domain_embedding(batch["timeline_domain_ids"])
            + self.window_embedding(batch["timeline_window_ids"])
            + self.unit_embedding(batch["timeline_unit_ids"])
            + self.route_embedding(batch["timeline_route_ids"])
            + self.evidence_embedding(batch["timeline_evidence_ids"])
            + self.encoder_flag_embedding(batch["timeline_encoder_flags"].long())
            + self.timeline_numeric(
                torch.cat([features * masks.to(features.dtype), masks.to(features.dtype)], dim=-1)
            )
        )

    def _text_tokens(self, batch: Mapping[str, Tensor]) -> Tensor:
        """投影冻结文本向量并加入角色、章节、模态和位置元数据。"""
        if any(
            module is None
            for module in (
                self.text_projection,
                self.text_role_embedding,
                self.text_section_role_embedding,
                self.text_modality_embedding,
                self.text_metadata,
                self.text_chunk_position,
                self.text_section_position,
                self.text_section_instance,
                self.text_norm,
            )
        ):
            raise RuntimeError("text encoder is disabled")
        assert self.text_projection is not None
        assert self.text_role_embedding is not None
        assert self.text_section_role_embedding is not None
        assert self.text_modality_embedding is not None
        assert self.text_metadata is not None
        assert self.text_chunk_position is not None
        assert self.text_section_position is not None
        assert self.text_section_instance is not None
        assert self.text_norm is not None
        chunk_positions = batch["text_chunk_position_ids"].clamp_max(
            self.config.text_chunk_position_vocab_size - 1
        )
        section_positions = batch["text_section_position_ids"].clamp_max(
            self.config.text_section_position_vocab_size - 1
        )
        section_instances = batch["text_section_instance_ids"].clamp_max(
            self.config.text_section_instance_vocab_size - 1
        )
        return self.text_norm(
            self.text_projection(batch["text_embeddings"])
            + self.text_role_embedding(batch["text_role_ids"])
            + self.text_section_role_embedding(batch["text_section_role_ids"])
            + self.text_modality_embedding(batch["text_modality_ids"])
            + self.text_metadata(batch["text_features"])
            + self.text_chunk_position(chunk_positions)
            + self.text_section_position(section_positions)
            + self.text_section_instance(section_instances)
        )

    def _text_documents(self, batch: Mapping[str, Tensor], sections: Tensor) -> Tensor:
        """Encode ordered sections within each source clinical document."""

        if self.text_section_encoder is None or self.text_section_scorer is None:
            raise RuntimeError("text encoder is disabled")

        document_mask = batch["text_document_attention_mask"]
        document_count = document_mask.shape[1]
        grouped, grouped_mask = _group_and_pad(
            sections,
            batch["text_attention_mask"],
            batch["text_document_ids"],
            document_count,
        )
        batch_size, _, section_width, hidden = grouped.shape
        flat = grouped.reshape(-1, section_width, hidden)
        flat_mask = grouped_mask.reshape(-1, section_width)
        encoded = _safe_transformer(self.text_section_encoder, flat, flat_mask)
        pooled = _masked_attention_pool(encoded, flat_mask, self.text_section_scorer)
        return pooled.reshape(batch_size, document_count, hidden)

    def _trajectory(
        self, batch: Mapping[str, Tensor], structured: Tensor, text_documents: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        window_count = self.config.reliable_window_count
        structured_grouped, structured_mask = _group_and_pad(
            structured,
            batch["timeline_attention_mask"],
            batch["timeline_window_ids"],
            window_count,
        )
        batch_size, _, structured_width, hidden = structured_grouped.shape
        structured_flat = structured_grouped.reshape(-1, structured_width, hidden)
        structured_mask_flat = structured_mask.reshape(-1, structured_width)
        structured_encoded = _safe_transformer(
            self.within_window_encoder, structured_flat, structured_mask_flat
        )
        structured_days = _masked_attention_pool(
            structured_encoded, structured_mask_flat, self.structured_scorer
        ).reshape(batch_size, window_count, hidden)
        structured_present = structured_mask.any(dim=-1)

        if text_documents is None:
            fusion_values = structured_days.reshape(-1, 1, hidden)
            fusion_mask = structured_present.reshape(-1, 1)
            day_presence = structured_present
        else:
            radiology_mask = batch["text_document_attention_mask"] & (
                batch["text_document_role_ids"] == 2
            )
            radiology_grouped, radiology_group_mask = _group_and_pad(
                text_documents,
                radiology_mask,
                batch["text_document_window_ids"],
                window_count,
            )
            radiology_width = radiology_grouped.shape[2]
            radiology_flat = radiology_grouped.reshape(-1, radiology_width, hidden)
            radiology_mask_flat = radiology_group_mask.reshape(-1, radiology_width)
            fusion_values = torch.cat(
                [structured_days.reshape(-1, 1, hidden), radiology_flat], dim=1
            )
            fusion_mask = torch.cat(
                [structured_present.reshape(-1, 1), radiology_mask_flat], dim=1
            )
            day_presence = structured_present | radiology_group_mask.any(dim=-1)
        fused = _safe_transformer(self.day_fusion_encoder, fusion_values, fusion_mask)
        day_values = _masked_attention_pool(fused, fusion_mask, self.day_fusion_scorer).reshape(
            batch_size, window_count, hidden
        )
        day_values = day_values + self.window_embedding(
            torch.arange(1, window_count + 1, device=day_values.device).unsqueeze(0)
        )
        encoded_days = _safe_transformer(self.temporal_encoder, day_values, day_presence)
        trajectory = (
            encoded_days * day_presence.unsqueeze(-1).to(encoded_days.dtype)
        ).sum(dim=1) / day_presence.sum(dim=1, keepdim=True).clamp_min(1).to(encoded_days.dtype)

        retrospective_mask = batch["timeline_attention_mask"] & (
            batch["timeline_window_ids"] > window_count
        )
        retrospective = _masked_attention_pool(
            structured, retrospective_mask, self.structured_scorer
        )
        return (
            trajectory,
            day_presence.any(dim=1),
            retrospective,
            retrospective_mask.any(dim=1),
        )

    def _text_subset(
        self, batch: Mapping[str, Tensor], values: Tensor, subset_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.text_encoder is None or self.text_scorer is None:
            raise RuntimeError("text encoder is disabled")
        mask = batch["text_document_attention_mask"] & subset_mask
        group_ids = torch.ones_like(batch["text_document_role_ids"])
        grouped, grouped_mask = _group_and_pad(values, mask, group_ids, 1)
        sequence = grouped[:, 0]
        sequence_mask = grouped_mask[:, 0]
        encoded = _safe_transformer(self.text_encoder, sequence, sequence_mask)
        return _masked_attention_pool(encoded, sequence_mask, self.text_scorer), mask.any(dim=1)

    def _summaries(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = batch["global_summary_features"]
        masks = batch["global_summary_feature_masks"]
        values = self.summary_embedding(batch["global_summary_concept_ids"]) + self.summary_numeric(
            torch.cat([features * masks.to(features.dtype), masks.to(features.dtype)], dim=-1)
        )
        attention = batch["global_summary_attention_mask"]
        return _masked_attention_pool(values, attention, self.summary_scorer), attention.any(dim=1)

    def _static(self, batch: Mapping[str, Tensor]) -> Tensor:
        continuous = batch["static_continuous"]
        masks = batch["static_continuous_masks"]
        categorical = batch["static_categorical"]
        categories = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.static_category_embeddings)
        ]
        return self.static_encoder(
            torch.cat(
                [
                    continuous * masks.to(continuous.dtype),
                    masks.to(continuous.dtype),
                    batch["static_binary"],
                    *categories,
                ],
                dim=-1,
            )
        )
