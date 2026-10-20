"""Distributed Training, SFT Reasoning, and GRPO RLVR Scripts for Cadbury-Chocolate 70B."""
from .pretrain_fsdp import build_70b_config
from .sft_reasoning_trainer import setup_sft_reasoning_trainer, CompletionOnlyReasoningCollator, NEFTuneHook
from .rl_grpo_verifier import format_reward_fn, syntax_and_ast_reward_fn, execution_sandbox_reward_fn

__all__ = [
    "build_70b_config",
    "setup_sft_reasoning_trainer",
    "CompletionOnlyReasoningCollator",
    "NEFTuneHook",
    "format_reward_fn",
    "syntax_and_ast_reward_fn",
    "execution_sandbox_reward_fn",
]
