"""PyTorch modules for AECOPD competing-risk modelling."""

from .batching import BatchingConfig, LengthBucketBatcher, load_timeline_lengths
from .aligned_model import AlignedModelConfig, TimeAlignedAecopdRiskModel
from .conditional_losses import conditional_admission_equal_loss, conditional_patient_equal_loss
from .conditional_model import ConditionalIntervalAecopdModel, ConditionalIntervalOutput
from .conditional_outcomes import ConditionalOutcome, ConditionalOutcomeStore
from .losses import discrete_competing_risk_nll, patient_equal_competing_risk_nll
from .longitudinal_batching import LongitudinalBatchingConfig, PatientSequenceBatcher
from .longitudinal_collator import LongitudinalBatchCollator, PatientSequenceInput
from .longitudinal_model import LongitudinalAecopdRiskModel
from .patient_sequence_cache import PatientSequenceCache, build_patient_sequence_cache
from .model import AecopdRiskModel, ModelConfig, ModelOutput
from .multimodal_collator import (
    FrozenTextBatch,
    FrozenTextEmbeddingStore,
    MultimodalBatchCollator,
    StructuredBatchCollator,
)
from .patient_bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    MetricInterval,
    patient_clustered_bootstrap,
)

__all__ = [
    "AecopdRiskModel",
    "AlignedModelConfig",
    "BatchingConfig",
    "LengthBucketBatcher",
    "ModelConfig",
    "ModelOutput",
    "TimeAlignedAecopdRiskModel",
    "ConditionalIntervalAecopdModel",
    "ConditionalIntervalOutput",
    "ConditionalOutcome",
    "ConditionalOutcomeStore",
    "FrozenTextBatch",
    "FrozenTextEmbeddingStore",
    "MultimodalBatchCollator",
    "StructuredBatchCollator",
    "discrete_competing_risk_nll",
    "patient_equal_competing_risk_nll",
    "conditional_patient_equal_loss",
    "conditional_admission_equal_loss",
    "LongitudinalBatchingConfig",
    "PatientSequenceBatcher",
    "LongitudinalBatchCollator",
    "PatientSequenceInput",
    "LongitudinalAecopdRiskModel",
    "PatientSequenceCache",
    "build_patient_sequence_cache",
    "load_timeline_lengths",
    "BootstrapConfig",
    "BootstrapResult",
    "MetricInterval",
    "patient_clustered_bootstrap",
]
