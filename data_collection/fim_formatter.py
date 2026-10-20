"""
Fill-In-The-Middle (FIM) Data Transformation
Implements the PSM (Prefix-Suffix-Middle) and SPM (Suffix-Prefix-Middle)
probabilistic slicing used by state-of-the-art code LLMs (StarCoder, CodeLlama).

Standard Autoregressive Training:
  [Prefix] -> [Middle] -> [Suffix]

FIM Mode (PSM):
  <PRE> [Prefix] <SUF> [Suffix] <MID> [Middle] <EOT>

This trains the attention mechanism to synthesize missing code given both
surrounding context (before and after the cursor).
"""

import random
from typing import Tuple, Dict, Any, Optional

# Special Token String Identifiers
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_EOT = "<|endoftext|>"


def apply_fill_in_the_middle(
    text: str,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[str, bool]:
    """
    Transforms code into FIM representation with a specified probability.
    
    Args:
        text: Raw source code string
        fim_rate: Probability of applying FIM transformation (standard is 0.5)
        spm_rate: Probability of SPM vs PSM ordering given FIM is applied
        seed: Optional RNG seed for deterministic testing
        
    Returns:
        transformed_text: Formatted string (either FIM-encoded or unchanged)
        is_fim: Boolean flag indicating if FIM was applied
    """
    if seed is not None:
        random.seed(seed)

    # With probability (1 - fim_rate), leave text in standard left-to-right order
    if random.random() > fim_rate:
        return text, False

    # Guard against empty or tiny files
    if len(text) < 30:
        return text, False

    # Split points: choose two cut points uniformly at random
    cut_1 = random.randint(0, len(text))
    cut_2 = random.randint(0, len(text))
    
    start_cut, end_cut = sorted([cut_1, cut_2])
    
    prefix = text[:start_cut]
    middle = text[start_cut:end_cut]
    suffix = text[end_cut:]

    # Choose between Prefix-Suffix-Middle (PSM) and Suffix-Prefix-Middle (SPM)
    if random.random() < spm_rate:
        # SPM format
        transformed = f"{FIM_PREFIX}{suffix}{FIM_SUFFIX}{prefix}{FIM_MIDDLE}{middle}{FIM_EOT}"
    else:
        # PSM format
        transformed = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}{FIM_EOT}"

    return transformed, True


def process_record_fim(record: Dict[str, Any], fim_rate: float = 0.5) -> Dict[str, Any]:
    """Applies FIM to a record dictionary and records telemetry metadata."""
    content = record.get("content", "")
    transformed, is_fim = apply_fill_in_the_middle(content, fim_rate=fim_rate)
    
    updated_record = dict(record)
    updated_record["content"] = transformed
    updated_record["is_fim"] = is_fim
    return updated_record


if __name__ == "__main__":
    # Demonstration test case
    sample_code = """def add_numbers(a, b):
    # Calculate sum of two integers
    result = a + b
    return result"""

    print("--- Original Code ---")
    print(sample_code)

    print("\n--- Transformed with FIM ---")
    transformed, was_fim = apply_fill_in_the_middle(sample_code, fim_rate=1.0, seed=42)
    print(transformed)
