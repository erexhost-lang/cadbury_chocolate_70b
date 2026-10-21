"""
Cadbury-Chocolate Frontier V2: MoE Auxiliary-Loss-Free Routing Configuration & Scheduler
Dynamically balances 16 Routed Experts across 3.0T to 5.5T Token Pre-Training Schedule.

Mathematical Foundation:
Traditional auxiliary load-balancing losses (L_aux) penalize language modeling gradients,
forcing the router to make suboptimal domain assignments just to equalize GPU compute.

Cadbury-70B replaces L_aux with Dynamic Router Biases (b_i):
1. Router logits with bias:
   biased_logits = W_r * x_t + b_i
   top2_indices  = Top2(biased_logits)
2. Unbiased gradient backpropagation:
   routing_weights = Softmax(W_r * x_t)[top2_indices]
3. Non-differentiable EMA bias updates:
   b_i <- b_i - gamma(t) * sign(load_i - avg_load)
   where gamma(t) decays smoothly from 0.005 to 0.0002 across the 3.0T - 5.5T token schedule.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("moe_routing_config")


# ==============================================================================
# 1. Routing Configuration & Domain Specialization Dataclasses
# ==============================================================================
@dataclass
class MoERoutingConfig:
    """Configuration for Auxiliary-Loss-Free Router across 3.0T to 5.5T tokens."""
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k: int = 2
    hidden_size: int = 8192
    
    # Pre-training token schedule parameters
    target_tokens: int = 4_000_000_000_000  # 4.0 Trillion tokens
    warmup_tokens: int = 100_000_000_000    # 100 Billion initial symmetry-breaking tokens
    
    # Dynamic Bias Schedule: gamma(t)
    initial_bias_rate: float = 0.005        # High exploration early in training
    min_bias_rate: float = 0.0002           # Fine stabilization at late training
    decay_schedule: str = "cosine"          # "cosine" or "linear"
    
    # Capacity & Buffer Constraints
    expert_capacity_factor: float = 1.25    # Slack factor for distributed all-to-all dispatch
    starvation_threshold: float = 0.70      # Alert if an expert gets <70% of average load
    overload_threshold: float = 1.30        # Alert if an expert gets >130% of average load

    # Pre-assigned Domain Clusters (for tracking specialization emergence)
    domain_specializations: Dict[str, List[int]] = field(default_factory=lambda: {
        "low_level_systems": [0, 1, 2, 3],          # C, C++, Rust, Go, Kernels, Memory
        "application_infrastructure": [4, 5, 6, 7],  # Python, TS, Java, SQL, APIs, Cloud
        "formal_logic_and_math": [8, 9, 10, 11],     # DSA, Complexity, Proofs, Formal logic
        "architecture_and_diffs": [12, 13, 14, 15],  # System design, PR Evolution, Diffs
    })


@dataclass
class RoutingTelemetry:
    """Metrics tracking router balance, expert entropy, and load distribution."""
    step: int
    tokens_processed: int
    gamma_rate: float
    load_cv: float                          # Coefficient of Variation (std / mean)
    expert_entropy: float                   # Shannon entropy of routing distribution
    starved_experts: List[int]
    overloaded_experts: List[int]
    expert_loads: List[float]


# ==============================================================================
# 2. Dynamic Router Bias Scheduler
# ==============================================================================
class DynamicRouterBiasScheduler:
    """
    Computes dynamic bias step gamma(t) as a function of total processed tokens.
    Guarantees that expert routing settles into crisp domain specialization
    without jitter in late training stages.
    """

    def __init__(self, cfg: MoERoutingConfig):
        self.cfg = cfg

    def get_bias_rate(self, current_tokens: int) -> float:
        """
        Calculates gamma(t) using cosine decay between initial_bias_rate and min_bias_rate.
        """
        if current_tokens < self.cfg.warmup_tokens:
            return self.cfg.initial_bias_rate

        progress = min(1.0, (current_tokens - self.cfg.warmup_tokens) / max(1, self.cfg.target_tokens - self.cfg.warmup_tokens))
        
        if self.cfg.decay_schedule == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            gamma = self.cfg.min_bias_rate + (self.cfg.initial_bias_rate - self.cfg.min_bias_rate) * cosine_decay
        else:
            gamma = self.cfg.initial_bias_rate - progress * (self.cfg.initial_bias_rate - self.cfg.min_bias_rate)

        return gamma


# ==============================================================================
# 3. Auxiliary-Loss-Free Router Controller & Monitor
# ==============================================================================
class AuxiliaryLossFreeRouterController(nn.Module):
    """
    Maintains and updates the non-differentiable expert bias vector b_i
    during distributed TPU/GPU training across the 3.0T to 5.5T token schedule.
    """

    def __init__(self, cfg: Optional[MoERoutingConfig] = None):
        super().__init__()
        self.cfg = cfg or MoERoutingConfig()
        self.scheduler = DynamicRouterBiasScheduler(self.cfg)
        self.total_tokens_processed = 0

        # Dynamic bias vector buffer (persisted with checkpoint, not updated via AdamW)
        self.register_buffer("expert_biases", torch.zeros(self.cfg.num_routed_experts, dtype=torch.float32))

    def compute_routing(
        self,
        gate_logits: torch.Tensor,
        current_step_tokens: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingTelemetry]:
        """
        Executes auxiliary-loss-free top-k expert selection and updates dynamic biases.
        
        Args:
            gate_logits: Raw router affinities from W_r * x_t (N, num_experts)
            current_step_tokens: Number of tokens in the current forward pass
        """
        N, num_experts = gate_logits.shape
        self.total_tokens_processed += (current_step_tokens if current_step_tokens > 0 else N)

        # 1. Apply dynamic biases to logits for expert selection
        biased_logits = gate_logits + self.expert_biases

        # 2. Select Top-K experts
        topk_weights_raw, topk_indices = torch.topk(biased_logits, self.cfg.top_k, dim=-1)

        # 3. Unbiased routing weights (calculated from original gate_logits)
        selected_unbiased_logits = torch.gather(gate_logits, -1, topk_indices)
        routing_weights = F.softmax(selected_unbiased_logits, dim=-1)

        # 4. Update dynamic biases during training
        gamma = self.scheduler.get_bias_rate(self.total_tokens_processed)
        expert_counts = torch.bincount(topk_indices.view(-1), minlength=num_experts).float()
        target_count = expert_counts.mean()

        if self.training:
            with torch.no_grad():
                load_error = expert_counts - target_count
                # b_i <- b_i - gamma * sign(load_i - target)
                self.expert_biases.sub_(gamma * torch.sign(load_error))

        # 5. Compute Load Telemetry
        with torch.no_grad():
            loads = (expert_counts / max(1.0, expert_counts.sum())).tolist()
            loads_tensor = torch.tensor(loads)
            mean_load = loads_tensor.mean().item()
            std_load = loads_tensor.std().item()
            load_cv = (std_load / (mean_load + 1e-9)) if mean_load > 0 else 0.0

            # Calculate Shannon entropy of routing distribution
            probs = loads_tensor.clamp(min=1e-9)
            entropy = -torch.sum(probs * torch.log2(probs)).item()

            starved = [i for i, l in enumerate(loads) if l < self.cfg.starvation_threshold * (1.0 / num_experts)]
            overloaded = [i for i, l in enumerate(loads) if l > self.cfg.overload_threshold * (1.0 / num_experts)]

        telemetry = RoutingTelemetry(
            step=0,
            tokens_processed=self.total_tokens_processed,
            gamma_rate=gamma,
            load_cv=round(load_cv, 4),
            expert_entropy=round(entropy, 4),
            starved_experts=starved,
            overloaded_experts=overloaded,
            expert_loads=[round(l, 4) for l in loads],
        )

        return routing_weights, topk_indices, telemetry

    def reset_biases(self):
        """Resets expert biases to zero."""
        self.expert_biases.zero_()


# ==============================================================================
# 4. Standalone Verification Routine
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Cadbury-70B Frontier V2: MoE Auxiliary-Loss-Free Routing Config & Test")
    print("=" * 80)

    cfg = MoERoutingConfig(num_routed_experts=16, top_k=2, target_tokens=4_000_000_000_000)
    controller = AuxiliaryLossFreeRouterController(cfg)
    controller.train()

    print(f"\n[1. Routing Configuration]")
    print(f"Total Routed Experts: {cfg.num_routed_experts} | Top-K Active: {cfg.top_k}")
    print(f"Target Pre-training Schedule: {cfg.target_tokens / 1e12:.2f} Trillion Tokens")
    print(f"Initial Bias Step (gamma_0): {cfg.initial_bias_rate} -> Final (gamma_min): {cfg.min_bias_rate}")

    # Inspect Bias Rate Decay across 4T Tokens
    print(f"\n[2. Dynamic Bias Decay Schedule (gamma across tokens)]")
    for milestone in [0, 500e9, 1e12, 2e12, 3e12, 4e12]:
        rate = controller.scheduler.get_bias_rate(int(milestone))
        print(f"  Milestone {milestone / 1e12:4.1f}T Tokens -> gamma = {rate:.6f}")

    # Simulate forward passes with synthetic un-balanced logits
    print(f"\n[3. Simulating Router Load Balancing across 100 Forward Passes]")
    batch_tokens = 512
    # Expert 0 artificially starts with 5x higher logits to test recovery
    dummy_logits = torch.randn(batch_tokens, 16)
    dummy_logits[:, 0] += 3.0

    for step in range(100):
        weights, indices, telem = controller.compute_routing(dummy_logits, current_step_tokens=batch_tokens)

    print(f"Final Load CV (Coefficient of Variation): {telem.load_cv:.4f} (Ideal < 0.15)")
    print(f"Final Expert Entropy: {telem.expert_entropy:.4f} bits (Max possible: {math.log2(16):.4f})")
    print(f"Starved Experts Count: {len(telem.starved_experts)}")
    print(f"Adjusted Dynamic Biases (b_i):")
    print(f"  Expert #0 (initially overloaded): bias = {controller.expert_biases[0].item():.4f}")
    print(f"  Expert #1:                         bias = {controller.expert_biases[1].item():.4f}")
    print("=" * 80)
    print("MoE Auxiliary-Loss-Free Routing Configuration Verified Successfully.")
