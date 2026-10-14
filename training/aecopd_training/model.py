from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
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
    hidden_size: int = 128
    attention_heads: int = 4
    temporal_layers: int = 2
    dropout: float = 0.1
    horizon_days: int = 365
    text_embedding_dim: int = 0
    text_role_vocab_size: int = 6
    text_feature_count: int = 3
    initial_readmission_logit: float = -6.5
    initial_death_logit: float = -7.0

    @classmethod
    def from_artifacts(
        cls,
        path: str | Path,
        **overrides: object,
    ) -> "ModelConfig":
        artifacts = json.loads(Path(path).read_text(encoding="utf-8"))
        category_sizes = tuple(
            len(value["vocabulary"])
            for value in artifacts["static_categorical"].values()
        )
        values: dict[str, object] = {
            "concept_vocab_size": len(artifacts["event_vocabulary"]),
            "domain_vocab_size": len(artifacts["event_domain_vocabulary"]),
            "window_vocab_size": 18,
            "unit_vocab_size": len(artifacts["event_unit_vocabulary"]),
            "route_vocab_size": len(artifacts["event_route_vocabulary"]),
            "evidence_vocab_size": len(artifacts["event_evidence_vocabulary"]),
            "summary_vocab_size": len(artifacts["numeric_summary_vocabulary"]),
            "static_category_sizes": category_sizes,
            **overrides,
        }
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    probabilities: Tensor
    survival: Tensor
    cumulative_incidence: Tensor
    representation: Tensor

    def risk_at(self, day: int) -> Tensor:
        """Return [AECOPD readmission, all-cause death] cumulative risk."""
        if day < 1 or day > self.cumulative_incidence.shape[1]:
            raise ValueError(
                f"day must be between 1 and {self.cumulative_incidence.shape[1]}"
            )
        return self.cumulative_incidence[:, day - 1]


def _masked_attention_pool(values: Tensor, mask: Tensor, scorer: nn.Module) -> Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[-1]))
    scores = scorer(values).squeeze(-1)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=1) * mask.to(scores.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.sum(values * weights.unsqueeze(-1), dim=1)


def _competing_risk_curves(logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    probabilities = logits.softmax(dim=-1)
    event_free = probabilities[..., 0]
    survival_before = torch.cat(
        [
            torch.ones_like(event_free[:, :1]),
            torch.cumprod(event_free[:, :-1], dim=1),
        ],
        dim=1,
    )
    incidence = torch.cumsum(
        survival_before.unsqueeze(-1) * probabilities[..., 1:],
        dim=1,
    )
    survival = torch.cumprod(event_free, dim=1)
    return probabilities, survival, incidence


class AecopdRiskModel(nn.Module):
    """Hierarchical multimodal encoder with a daily competing-risk head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
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
            nn.Linear(config.timeline_feature_count * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.timeline_norm = nn.LayerNorm(hidden)
        self.token_scorer = nn.Linear(hidden, 1)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            norm=nn.LayerNorm(hidden),
        )
        self.structured_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.summary_embedding = nn.Embedding(config.summary_vocab_size, hidden, padding_idx=0)
        self.summary_numeric = nn.Sequential(
            nn.Linear(config.summary_feature_count * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.summary_scorer = nn.Linear(hidden, 1)

        category_dim = max(4, hidden // 16)
        self.static_category_embeddings = nn.ModuleList(
            nn.Embedding(size, category_dim, padding_idx=0)
            for size in config.static_category_sizes
        )
        static_input_size = (
            config.static_continuous_count * 2
            + config.static_binary_count
            + category_dim * len(config.static_category_sizes)
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        if config.text_embedding_dim > 0:
            self.text_projection: nn.Module | None = nn.Linear(config.text_embedding_dim, hidden)
            self.text_role_embedding: nn.Module | None = nn.Embedding(
                config.text_role_vocab_size, hidden, padding_idx=0
            )
            self.text_metadata: nn.Module | None = nn.Sequential(
                nn.Linear(config.text_feature_count, hidden),
                nn.GELU(),
            )
            self.text_norm: nn.Module | None = nn.LayerNorm(hidden)
            self.text_scorer: nn.Module | None = nn.Linear(hidden, 1)
        else:
            self.text_projection = None
            self.text_role_embedding = None
            self.text_metadata = None
            self.text_norm = None
            self.text_scorer = None
        fusion_branches = 4 if self.text_projection is not None else 3
        self.fusion = nn.Sequential(
            nn.Linear(hidden * fusion_branches, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )
        self.risk_head = nn.Linear(hidden, config.horizon_days * 3)
        self._initialize_risk_head()

    def _initialize_risk_head(self) -> None:
        # Start with a low daily hazard rather than a near-uniform three-class
        # distribution, which would numerically exhaust survival within days.
        nn.init.normal_(self.risk_head.weight, mean=0.0, std=1e-3)
        daily_bias = self.risk_head.bias.view(self.config.horizon_days, 3)
        with torch.no_grad():
            daily_bias[:, 0].zero_()
            daily_bias[:, 1].fill_(self.config.initial_readmission_logit)
            daily_bias[:, 2].fill_(self.config.initial_death_logit)

    def forward(
        self,
        batch: Mapping[str, Tensor],
        *,
        text_embeddings: Tensor | None = None,
    ) -> ModelOutput:
        structured = self._encode_timeline(batch)
        summary = self._encode_summaries(batch)
        static = self._encode_static(batch)
        branches = [structured, summary, static]
        if self.text_projection is not None:
            if text_embeddings is None:
                candidate = batch.get("text_embeddings")
                if candidate is None:
                    raise ValueError("text_embeddings are required when text_embedding_dim > 0")
                text_embeddings = candidate
            branches.append(self._encode_text(batch, text_embeddings))
        representation = self.fusion(torch.cat(branches, dim=-1))
        logits = self.risk_head(representation).reshape(
            representation.shape[0], self.config.horizon_days, 3
        )
        probabilities, survival, incidence = _competing_risk_curves(logits)
        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            survival=survival,
            cumulative_incidence=incidence,
            representation=representation,
        )

    def _encode_timeline(self, batch: Mapping[str, Tensor]) -> Tensor:
        features = batch["timeline_features"]
        feature_masks = batch["timeline_feature_masks"]
        token_values = self.timeline_norm(
            self.concept_embedding(batch["timeline_concept_ids"])
            + self.domain_embedding(batch["timeline_domain_ids"])
            + self.window_embedding(batch["timeline_window_ids"])
            + self.unit_embedding(batch["timeline_unit_ids"])
            + self.route_embedding(batch["timeline_route_ids"])
            + self.evidence_embedding(batch["timeline_evidence_ids"])
            + self.encoder_flag_embedding(batch["timeline_encoder_flags"].long())
            + self.timeline_numeric(
                torch.cat(
                    [features * feature_masks.to(features.dtype), feature_masks.to(features.dtype)],
                    dim=-1,
                )
            )
        )
        attention = batch["timeline_attention_mask"]
        window_ids = batch["timeline_window_ids"]
        windows: list[Tensor] = []
        presence: list[Tensor] = []
        for window_id in range(1, self.config.reliable_window_count + 1):
            mask = attention & (window_ids == window_id)
            windows.append(_masked_attention_pool(token_values, mask, self.token_scorer))
            presence.append(mask.any(dim=1))
        window_values = torch.stack(windows, dim=1)
        window_presence = torch.stack(presence, dim=1)
        safe_presence = window_presence.clone()
        no_reliable_window = ~safe_presence.any(dim=1)
        safe_presence[no_reliable_window, 0] = True
        encoded_windows = self.temporal_encoder(
            window_values,
            src_key_padding_mask=~safe_presence,
        )
        reliable = (
            encoded_windows * window_presence.unsqueeze(-1).to(encoded_windows.dtype)
        ).sum(dim=1) / window_presence.sum(dim=1, keepdim=True).clamp_min(1).to(encoded_windows.dtype)
        retrospective_mask = attention & (window_ids > self.config.reliable_window_count)
        retrospective = _masked_attention_pool(token_values, retrospective_mask, self.token_scorer)
        return self.structured_fusion(torch.cat([reliable, retrospective], dim=-1))

    def _encode_summaries(self, batch: Mapping[str, Tensor]) -> Tensor:
        features = batch["global_summary_features"]
        masks = batch["global_summary_feature_masks"]
        values = self.summary_embedding(batch["global_summary_concept_ids"]) + self.summary_numeric(
            torch.cat([features * masks.to(features.dtype), masks.to(features.dtype)], dim=-1)
        )
        return _masked_attention_pool(
            values,
            batch["global_summary_attention_mask"],
            self.summary_scorer,
        )

    def _encode_static(self, batch: Mapping[str, Tensor]) -> Tensor:
        continuous = batch["static_continuous"]
        masks = batch["static_continuous_masks"]
        categorical = batch["static_categorical"]
        category_vectors = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.static_category_embeddings)
        ]
        values = [
            continuous * masks.to(continuous.dtype),
            masks.to(continuous.dtype),
            batch["static_binary"],
            *category_vectors,
        ]
        return self.static_encoder(torch.cat(values, dim=-1))

    def _encode_text(self, batch: Mapping[str, Tensor], embeddings: Tensor) -> Tensor:
        if (
            self.text_projection is None
            or self.text_role_embedding is None
            or self.text_metadata is None
            or self.text_norm is None
            or self.text_scorer is None
        ):
            raise RuntimeError("Text encoder is not configured.")
        if embeddings.ndim == 2:
            return self.text_norm(self.text_projection(embeddings))
        if embeddings.ndim != 3:
            raise ValueError("text_embeddings must have shape [batch, documents, dimensions].")
        required = ("text_role_ids", "text_features", "text_attention_mask")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(f"Missing text batch fields: {missing}")
        values = self.text_norm(
            self.text_projection(embeddings)
            + self.text_role_embedding(batch["text_role_ids"])
            + self.text_metadata(batch["text_features"])
        )
        return _masked_attention_pool(values, batch["text_attention_mask"], self.text_scorer)
