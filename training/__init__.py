"""Distributed Training, SFT Reasoning, Adaptive Compute, and GRPO RLVR Scripts for Cadbury-Chocolate 70B."""

from .pretrain_fsdp import (
    setup_distributed,
    init_weights_fn,
    get_cosine_lr,
    PretrainDatasetFactory,
    BinaryMemoryMappedDataset,
    StreamingJSONLTokenDataset,
)
from .sft_reasoning_trainer import (
    setup_sft_reasoning_trainer,
    CompletionOnlyReasoningCollator,
    NEFTuneHook,
    SFTReasoningDataset,
)
from .train_qlora import (
    load_70b_quantized_model,
    attach_lora_adapters,
    CADBURY_LORA_TARGET_MODULES,
)
from .adaptive_compute import (
    MultiTokenPredictionModule,
    EntropyReasoningGate,
    AdaptiveInferenceWrapper,
)
from .rl_grpo_verifier import (
    VerificationTier,
    SandboxExecutionResult,
    TierVerificationReport,
    CodeASTValidator,
    validate_tier_1_static,
    SandboxExecutionEngine,
    AntiRewardHackingGuard,
    ReasoningAndExplorationRewardManager,
    MultiTierVerificationPipeline,
    GRPOTrainingEngine,
)

# Backward-compatible alias
AdaptiveBudgetPenalizer = ReasoningAndExplorationRewardManager

__all__ = [
    "setup_distributed",
    "init_weights_fn",
    "get_cosine_lr",
    "PretrainDatasetFactory",
    "BinaryMemoryMappedDataset",
    "StreamingJSONLTokenDataset",
    "setup_sft_reasoning_trainer",
    "CompletionOnlyReasoningCollator",
    "NEFTuneHook",
    "SFTReasoningDataset",
    "load_70b_quantized_model",
    "attach_lora_adapters",
    "CADBURY_LORA_TARGET_MODULES",
    "MultiTokenPredictionModule",
    "EntropyReasoningGate",
    "AdaptiveInferenceWrapper",
    "VerificationTier",
    "SandboxExecutionResult",
    "TierVerificationReport",
    "CodeASTValidator",
    "validate_tier_1_static",
    "SandboxExecutionEngine",
    "AntiRewardHackingGuard",
    "ReasoningAndExplorationRewardManager",
    "AdaptiveBudgetPenalizer",
    "MultiTierVerificationPipeline",
    "GRPOTrainingEngine",
]
