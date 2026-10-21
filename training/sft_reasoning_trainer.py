"""
Reasoning & Tool-Aware Supervised Fine-Tuning (SFT) Pipeline
Trains Cadbury-Chocolate 70B on synthetic CoT and multi-turn tool trajectories.

Key Methodologies:
1. Completion-Only Masking: Masks prompts, system instructions, and <tool_output> blocks
   with label -100 so CrossEntropy loss is computed ONLY on thoughts, tool calls, and assistant code.
2. NEFTune (Noisy Embeddings Fine-Tuning): Injects scaled uniform noise into embeddings during
   training (alpha=5.0) to regularize against memorization of synthetic formatting templates.
3. Special Token Embeddings: Expands vocabulary with <think>, <tool_call>, <tool_output> tags.
4. CadburyMoEBlock FSDP Auto-Wrapping: Shards full 70B MoE layers across cluster GPUs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.moe_mla_architecture import Cadbury70BMoEModel, CadburyMoEBlock, MLAConfig, MoEConfig
from utils.logger import setup_logger

logger = setup_logger("sft_reasoning_trainer")

# Special control tokens for CoT reasoning and agentic tools
REASONING_SPECIAL_TOKENS = [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_output>",
    "</tool_output>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


# ==============================================================================
# 1. NEFTune: Noisy Embeddings Fine-Tuning Hook
# ==============================================================================
class NEFTuneHook:
    """
    Applies uniform noise to embedding representations during forward passes.
    Formula: noise = uniform(-1, 1) * (noise_alpha / sqrt(L * d))
    Proven to boost instruction-following and reasoning benchmarks by 3-5%.
    """

    def __init__(self, noise_alpha: float = 5.0):
        self.noise_alpha = noise_alpha

    def __call__(self, module: nn.Module, input_args, output: torch.Tensor) -> torch.Tensor:
        if not module.training:
            return output

        dims = torch.tensor(output.size(1) * output.size(2), device=output.device, dtype=torch.float32)
        mag_norm = self.noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(output).uniform_(-1, 1) * mag_norm
        return output + noise


# ==============================================================================
# 2. Completion-Only Data Collator
# ==============================================================================
class CompletionOnlyReasoningCollator:
    """
    Masks user prompts, system prompts, and environment tool outputs by replacing
    their token IDs in `labels` with -100 (PyTorch cross_entropy ignore_index).
    
    Gradient updates only flow from:
      - <think>...</think>
      - <tool_call>...</tool_call>
      - Assistant markdown/code answers
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: str = "<|start_header_id|>assistant<|end_header_id|>",
        tool_output_start: str = "<tool_output>",
        tool_output_end: str = "</tool_output>",
        max_length: int = 32768,
    ):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.tool_out_start_ids = tokenizer.encode(tool_output_start, add_special_tokens=False)
        self.tool_out_end_ids = tokenizer.encode(tool_output_end, add_special_tokens=False)
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for feature in features:
            input_ids = feature["input_ids"]
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]

            labels = list(input_ids)
            masked_labels = [-100] * len(labels)

            seq_len = len(input_ids)
            i = 0
            in_assistant = False
            in_tool_output = False

            while i < seq_len:
                if self.response_template_ids and input_ids[i : i + len(self.response_template_ids)] == self.response_template_ids:
                    in_assistant = True
                    i += len(self.response_template_ids)
                    continue

                if self.tool_out_start_ids and input_ids[i : i + len(self.tool_out_start_ids)] == self.tool_out_start_ids:
                    in_tool_output = True
                    i += len(self.tool_out_start_ids)
                    continue

                if self.tool_out_end_ids and input_ids[i : i + len(self.tool_out_end_ids)] == self.tool_out_end_ids:
                    in_tool_output = False
                    i += len(self.tool_out_end_ids)
                    continue

                if in_assistant and not in_tool_output:
                    masked_labels[i] = input_ids[i]
                elif not self.response_template_ids:
                    masked_labels[i] = input_ids[i]

                i += 1

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(masked_labels, dtype=torch.long))
            batch_attention_mask.append(torch.ones(len(input_ids), dtype=torch.long))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask_padded,
        }


# ==============================================================================
# 3. JSONL SFT Dataset
# ==============================================================================
class SFTReasoningDataset(Dataset):
    """Loads tokenized multi-turn reasoning conversations from JSONL."""

    def __init__(self, data_file: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 32768):
        self.samples: List[List[int]] = []
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if "input_ids" in record:
                            self.samples.append(record["input_ids"][:max_length])
                        elif "text" in record:
                            tokens = tokenizer.encode(record["text"], max_length=max_length, truncation=True)
                            self.samples.append(tokens)
                    except Exception:
                        continue
        else:
            logger.warning(f"Data file '{data_file}' not found. Initializing with sample reasoning conversation.")
            sample_text = (
                "<|start_header_id|>system<|end_header_id|>\n"
                "You are an expert coder.<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                "Implement quickselect in Python.<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
                "<think>We need O(N) average time complexity using Lomuto partitioning.</think>\n"
                "```python\ndef quickselect(arr, k): pass\n```<|eot_id|>"
            )
            self.samples.append(tokenizer.encode(sample_text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"input_ids": self.samples[idx]}


# ==============================================================================
# 4. SFT Training Execution Setup
# ==============================================================================
def setup_sft_reasoning_trainer(
    model_name_or_path: str = "checkpoints/cadbury-70b",
    dataset_path: str = "data/processed/sft_reasoning/sample_reasoning_data.jsonl",
    output_dir: str = "checkpoints/cadbury-70b-sft-reasoning",
    neftune_alpha: float = 5.0,
    max_seq_len: int = 32768,
):
    """Initializes the Reasoning SFT Trainer with NEFTune and Completion-Only masking."""
    logger.info(f"Setting up SFT for Cadbury-70B model from: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path if os.path.exists(model_name_or_path) else "tokenization/cadbury_tokenizer",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>"

    # Register special tokens
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": REASONING_SPECIAL_TOKENS})
    logger.info(f"Registered {num_added} special reasoning/tool tokens.")

    # Instantiate Cadbury 70B MoE model
    mla_cfg = MLAConfig(hidden_size=8192, max_position_embeddings=max_seq_len)
    moe_cfg = MoEConfig(hidden_size=8192)
    model = Cadbury70BMoEModel(vocab_size=len(tokenizer), mla_cfg=mla_cfg, moe_cfg=moe_cfg)

    # Attach NEFTune hook to token embeddings
    if neftune_alpha > 0:
        logger.info(f"Activating NEFTune embedding noise regularization (alpha={neftune_alpha})...")
        model.get_input_embeddings().register_forward_hook(NEFTuneHook(noise_alpha=neftune_alpha))

    collator = CompletionOnlyReasoningCollator(
        tokenizer=tokenizer,
        max_length=max_seq_len,
    )

    dataset = SFTReasoningDataset(dataset_path, tokenizer, max_length=max_seq_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="CadburyMoEBlock",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Reasoning & Tool SFT Trainer configured successfully.")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SFT Reasoning Trainer")
    parser.add_argument("--model-path", type=str, default="checkpoints/cadbury-70b")
    parser.add_argument("--data-file", type=str, default="data/processed/sft_reasoning/sample_reasoning_data.jsonl")
    parser.add_argument("--output-dir", type=str, default="checkpoints/cadbury-70b-sft-reasoning")
    parser.add_argument("--neftune-alpha", type=float, default=5.0)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    args = parser.parse_args()

    trainer = setup_sft_reasoning_trainer(
        model_name_or_path=args.model_path,
        dataset_path=args.data_file,
        output_dir=args.output_dir,
        neftune_alpha=args.neftune_alpha,
        max_seq_len=args.max_seq_len,
    )
    logger.info("SFT Reasoning Trainer successfully initialized.")
