"""Distributed Training, SFT Reasoning, Adaptive Compute, and GRPO RLVR Scripts for Cadbury-Chocolate 70B."""
from .pretrain_fsdp import build_70b_config
from .sft_reasoning_trainer import setup_sft_reasoning_trainer, CompletionOnlyReasoningCollator, NEFTuneHook
from .adaptive_compute import MultiTokenPredictionModule, EntropyReasoningGate, AdaptiveInferenceWrapper
from .rl_grpo_verifier import (
    VerificationTier,
    SandboxExecutionResult,
    TierVerificationReport,
    CodeASTValidator,
    validate_tier_1_static,
    SandboxExecutionEngine,
    AntiRewardHackingGuard,
    AdaptiveBudgetPenalizer,
    MultiTierVerificationPipeline,
    GRPOTrainingEngine,
    format_reward_fn,
    syntax_and_ast_reward_fn,
    execution_sandbox_reward_fn,
    adaptive_thinking_budget_reward_fn,
    information_gain_tool_reward_fn,
    anti_reward_hacking_fn,
)

__all__ = [
    "build_70b_config",
    "setup_sft_reasoning_trainer",
    "CompletionOnlyReasoningCollator",
    "NEFTuneHook",
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
    "AdaptiveBudgetPenalizer",
    "MultiTierVerificationPipeline",
    "GRPOTrainingEngine",
    "format_reward_fn",
    "syntax_and_ast_reward_fn",
    "execution_sandbox_reward_fn",
    "adaptive_thinking_budget_reward_fn",
    "information_gain_tool_reward_fn",
    "anti_reward_hacking_fn",
]
