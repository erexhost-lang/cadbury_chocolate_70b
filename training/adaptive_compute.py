"""
Adaptive Compute Architecture for 70B Code LLM
1. Multi-Token Prediction (MTP) Head:
   Auxiliary heads predicting k future tokens simultaneously (DeepSeek-V3 / Meta MTP style)
   to enforce structural lookahead planning during pretraining & SFT.
2. Entropy-Aware Reasoning Gate (Anti-Rumination Early Exit):
   Monitors Shannon entropy H(p) = -sum(p log p) of next-token distribution during <think>.
   When confidence is high (entropy < threshold), forces early exit out of thinking
   so simple tasks bypass rumination and answer immediately.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("adaptive_compute")


# ==============================================================================
# 1. Multi-Token Prediction (MTP) Auxiliary Head Module
# ==============================================================================
class MTPPredictionBlock(nn.Module):
    """
    Lightweight prediction block for the i-th future token.
    Combines transformer representation h_t with embedded token t_{i}
    and projects into next-token prediction space using shared LM head.
    """
    def __init__(self, hidden_size: int = 8192, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        # Small projection layer preserving dimension
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, h_prev: torch.Tensor, emb_curr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: (B, T, D) hidden states from previous token or base model
            emb_curr: (B, T, D) embedding of current token
        """
        fused = torch.cat([h_prev, emb_curr], dim=-1)
        h_next = self.proj(fused)
        return self.norm(h_next)


class MultiTokenPredictionModule(nn.Module):
    """
    Manages k auxiliary future token predictions.
    DeepSeek-V3 style: Forces base model representations to encode future trajectory.
    """
    def __init__(
        self,
        hidden_size: int = 8192,
        vocab_size: int = 128268,
        num_future_tokens: int = 2,
        loss_weight: float = 0.3,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_future_tokens = num_future_tokens
        self.loss_weight = loss_weight
        self.vocab_size = vocab_size

        # k lightweight blocks for predicting t+2, t+3, ..., t+k+1
        self.blocks = nn.ModuleList([
            MTPPredictionBlock(hidden_size=hidden_size, rms_norm_eps=rms_norm_eps)
            for _ in range(num_future_tokens)
        ])

    def compute_mtp_loss(
        self,
        base_hidden_states: torch.Tensor,
        token_embeddings: nn.Embedding,
        shared_lm_head: nn.Linear,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes auxiliary cross-entropy loss for k future tokens.
        
        Args:
            base_hidden_states: (B, T, D) output of the last transformer layer
            token_embeddings: embedding layer of the base model
            shared_lm_head: weight-tied or shared LM head projection
            input_ids: (B, T) input token IDs
            labels: (B, T) target labels with -100 masking
        """
        B, T, D = base_hidden_states.shape
        total_aux_loss = torch.tensor(0.0, device=base_hidden_states.device)
        metrics = {}

        curr_h = base_hidden_states

        for k_idx, block in enumerate(self.blocks):
            # Target is shifted forward by (k_idx + 2)
            shift = k_idx + 2
            if T <= shift:
                break

            # Current token embeddings: tokens at t .. T - shift
            curr_tokens = input_ids[:, 1 : T - shift + 1]
            curr_emb = token_embeddings(curr_tokens)

            # Hidden states for positions corresponding to inputs
            h_slice = curr_h[:, : T - shift]

            # Project through MTP block
            projected_h = block(h_slice, curr_emb)

            # Predict logits for token at position (t + shift)
            logits_k = shared_lm_head(projected_h)
            targets_k = labels[:, shift:]

            # Compute cross-entropy loss against shifted target
            loss_k = F.cross_entropy(
                logits_k.view(-1, self.vocab_size),
                targets_k.contiguous().view(-1),
                ignore_index=-100,
            )

            total_aux_loss = total_aux_loss + loss_k
            metrics[f"mtp_loss_future_{k_idx+1}"] = loss_k.item()

            # Pass projected_h as input to subsequent future prediction block
            curr_h = F.pad(projected_h, (0, 0, 0, shift), mode="constant", value=0)

        weighted_loss = self.loss_weight * (total_aux_loss / max(1, self.num_future_tokens))
        metrics["mtp_total_loss"] = weighted_loss.item()
        return weighted_loss, metrics


# ==============================================================================
# 2. Entropy-Aware Reasoning Gate (Anti-Rumination Early Exit)
# ==============================================================================
class EntropyReasoningGate:
    """
    Calculates Shannon Entropy during the generation of the <think> block:
        H(p) = - sum_v (p_v * log2(p_v))
    
    If initial reasoning entropy is below `entropy_threshold` (high certainty),
    it immediately signals an early exit to `</think>` so straightforward tasks
    do not ruminate or waste compute.
    """

    def __init__(self, entropy_threshold: float = 0.45, window_size: int = 15):
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.entropy_history: List[float] = []

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Calculate Shannon entropy in bits for the top logits distribution."""
        probs = F.softmax(logits, dim=-1)
        # Avoid log(0) with clamping
        log_probs = torch.log2(probs.clamp(min=1e-9))
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
        return entropy

    def should_early_exit(self, logits: torch.Tensor, current_token_id: int, think_token_id: int) -> bool:
        """
        Tracks entropy during <think> generation.
        Returns True if model is overthinking a low-complexity task.
        """
        entropy = self.compute_entropy(logits)
        self.entropy_history.append(entropy)

        # Only evaluate after a minimum window of initial tokens
        if len(self.entropy_history) < self.window_size:
            return False

        # Calculate moving average entropy of recent tokens
        avg_entropy = sum(self.entropy_history[-self.window_size:]) / self.window_size

        if avg_entropy < self.entropy_threshold:
            logger.info(
                f"[Entropy Gate] High Confidence detected (H={avg_entropy:.3f} < {self.entropy_threshold}). "
                f"Triggering Early-Exit from <think> to prevent rumination."
            )
            return True

        return False

    def reset(self):
        self.entropy_history.clear()


# ==============================================================================
# 3. Adaptive Inference Wrapper
# ==============================================================================
class AdaptiveInferenceWrapper:
    """
    Wraps model generation with real-time entropy gating and surgical prompt routing.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        entropy_threshold: float = 0.45,
        think_end_str: str = "</think>\n\n",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gate = EntropyReasoningGate(entropy_threshold=entropy_threshold)
        self.think_end_str = think_end_str

    @torch.no_grad()
    def generate_with_adaptive_thinking(
        self,
        prompt: str,
        max_think_tokens: int = 2048,
        max_response_tokens: int = 4096,
        temperature: float = 0.6,
    ) -> str:
        """
        Executes generation, truncating <think> immediately if entropy drops below threshold.
        """
        self.model.eval()
        self.gate.reset()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        generated_tokens = []
        in_thinking_mode = True

        for step in range(max_think_tokens + max_response_tokens):
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)

            # Check entropy gate while in thinking mode
            if in_thinking_mode and step > 10:
                if self.gate.should_early_exit(next_token_logits, 0, 0):
                    # Force early exit by injecting </think> string into generation
                    early_exit_ids = self.tokenizer.encode(self.think_end_str, add_special_tokens=False)
                    for eid in early_exit_ids:
                        generated_tokens.append(eid)
                    in_thinking_mode = False
                    input_ids = torch.cat([input_ids, torch.tensor([early_exit_ids], device=self.model.device)], dim=-1)
                    continue

            # Standard temperature sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Check if natural </think> was generated
            decoded_recent = self.tokenizer.decode(generated_tokens[-4:])
            if "</think>" in decoded_recent:
                in_thinking_mode = False

            # Stop token check
            if token_id == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=False)


if __name__ == "__main__":
    print("--- Testing Adaptive Compute Components Locally ---")
    
    # 1. Test Entropy Calculation
    gate = EntropyReasoningGate(entropy_threshold=0.45)
    
    # Confident distribution (low entropy)
    sharp_logits = torch.tensor([10.0, 0.1, -2.0, 0.0])
    # Unsure distribution (high entropy)
    flat_logits = torch.tensor([1.0, 1.0, 1.0, 1.0])

    sharp_h = gate.compute_entropy(sharp_logits)
    flat_h = gate.compute_entropy(flat_logits)

    print(f"Sharp Logits Entropy (High Certainty): {sharp_h:.4f} bits")
    print(f"Flat Logits Entropy (Unsure / High Rumination): {flat_h:.4f} bits")

    # 2. Test MTP Module Shapes
    B, T, D = 2, 64, 256
    vocab = 1000
    mtp = MultiTokenPredictionModule(hidden_size=D, vocab_size=vocab, num_future_tokens=2)
    fake_h = torch.randn(B, T, D)
    fake_emb = nn.Embedding(vocab, D)
    fake_head = nn.Linear(D, vocab, bias=False)
    fake_ids = torch.randint(0, vocab, (B, T))
    fake_labels = fake_ids.clone()

    loss, metrics = mtp.compute_mtp_loss(fake_h, fake_emb, fake_head, fake_ids, fake_labels)
    print("MTP Loss Output:", loss.item())
    print("MTP Metrics:", metrics)
