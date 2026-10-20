"""
Parameter-Efficient QLoRA Fine-Tuning for 70B Model
Enables fine-tuning a 70B model on a single 80GB A100/H100 GPU
or consumer cluster using 4-bit NF4 quantization and low-rank adapters.

VRAM Footprint:
- 70B base weights in 4-bit NF4: ~35 GB
- Trainable LoRA adapter matrices: ~1.2 GB
- Paged 8-bit AdamW optimizer states: ~4.8 GB
- Total VRAM: ~45–55 GB (Leaves plenty of headroom for 4k–8k context!)
"""

import argparse
import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("train_qlora")


def load_70b_quantized_model(model_name_or_path: str = "meta-llama/Meta-Llama-3-70B"):
    """
    Load 70B model with bitsandbytes 4-bit NormalFloat (NF4) quantization.
    """
    logger.info(f"Loading 70B model with 4-bit NF4 quantization from: {model_name_or_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # NormalFloat4 distribution
        bnb_4bit_use_double_quant=True,          # Quantizes quantization scale constants
        bnb_4bit_compute_dtype=torch.bfloat16,   # Dequantize to bf16 for fast matrix multiplication
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",                       # Automatically splits across available GPUs if multiple
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # Hardware-accelerated attention
        trust_remote_code=True,
    )

    # Freeze base model weights and enable gradient checkpointing for adapter training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def attach_lora_adapters(model, r: int = 64, lora_alpha: int = 128) -> torch.nn.Module:
    """
    Attach trainable Low-Rank Adaptation (LoRA) matrices to attention and MLP layers.
    """
    logger.info(f"Configuring LoRA adapters (Rank: {r}, Alpha: {lora_alpha})...")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target all linear projections in the 70B transformer blocks
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for 70B Code LLM")
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3-70B")
    parser.add_argument("--output-dir", type=str, default="checkpoints/cadbury-70b-qlora")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    args = parser.parse_args()

    # 1. Load 4-bit Quantized Model
    model = load_70b_quantized_model(args.model_path)

    # 2. Attach LoRA adapters
    model = attach_lora_adapters(model, r=args.lora_r, lora_alpha=args.lora_alpha)

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="paged_adamw_8bit",                # Paged optimizer prevents OOM spikes
        bf16=True,                               # Mixed precision in bfloat16
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=True,
    )

    logger.info("Ready for training execution on multi-GPU or single 80GB H100 node.")
    logger.info(f"Target checkpoints will be stored in: {args.output_dir}")


if __name__ == "__main__":
    main()
