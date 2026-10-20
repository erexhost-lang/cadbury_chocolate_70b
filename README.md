# 🍫 Cadbury-Chocolate 70B — Distributed Code-Generation LLM Pipeline

A complete, production-grade training architecture and preprocessing pipeline for a **70-Billion parameter code-generation language model**. Modeled after modern open-weights state-of-the-art architectures (Llama-3-70B, StarCoder2-33B/70B, and DeepSeek-Coder-33B/70B).

Everything in this repository is written in clean, modular, and thoroughly documented Python. It is designed to be studied, extended, and dispatched directly onto multi-node GPU clusters (SLURM / Kubernetes).

---

## 🏛️ Model Architecture & Mathematics (70B)

| Hyperparameter | Value | Architectural Purpose |
| :--- | :--- | :--- |
| **Active Parameters** | **~70.55 Billion** | High reasoning capacity across 80+ programming languages |
| **Layers (`n_layer`)** | **80** | Deep hierarchical abstraction of program logic & control flow |
| **Hidden Dimension (`d_model`)** | **8,192** | Wide residual stream for complex semantic representations |
| **Intermediate Size (`d_ff`)** | **28,672** | SwiGLU feed-forward network (~3.5× hidden dimension) |
| **Attention Heads (`n_head`)** | **64** | Multi-head attention (Query dimension = 128) |
| **KV Heads (`num_kv_heads`)** | **8** | **Grouped-Query Attention (GQA)** (8:1 ratio) |
| **Context Window** | **8,192 tokens** | Long-context repository-level reasoning (extensible to 32k) |
| **Vocabulary Size** | **128,256** | Byte-Level BPE preserving indentation & FIM tokens |
| **Position Embeddings** | **RoPE ($\theta = 500,000$)** | High base frequency for long-context extrapolation |
| **Attention Kernel** | **FlashAttention-2** | Hardware-fused $O(N)$ memory attention |

### Parameter Count Derivation
1. **Self-Attention per layer:**
   * $W_q: 8192 \times 8192 = 67,108,864$
   * $W_k: 8192 \times 1024 = 8,388,608$ (GQA 8 heads $\times$ 128 dim)
   * $W_v: 8192 \times 1024 = 8,388,608$
   * $W_o: 8192 \times 8192 = 67,108,864$
   * *Subtotal Attention:* $\approx 150.99\text{ Million params/layer}$

2. **SwiGLU MLP per layer:**
   * $W_{gate}, W_{up}, W_{down}: 3 \times (8192 \times 28672) = 704,643,072$
   * *Subtotal MLP:* $\approx 704.64\text{ Million params/layer}$

3. **Total Transformer Blocks:**
   * $80 \times (150.99\text{M} + 704.64\text{M}) = \mathbf{68.45\text{ Billion parameters}}$

4. **Embeddings & Final Head:**
   * $128,256 \text{ vocab} \times 8,192 \text{ hidden} \times 2 = \mathbf{2.10\text{ Billion parameters}}$
   * **Total Model Parameters: $\approx \mathbf{70.55\text{ Billion}}$**

---

## ⚡ Cluster Hardware & VRAM Memory Math

Training a 70B model requires understanding static and dynamic GPU memory:

* **Model Weights (in bfloat16):** $70.55\text{B} \times 2\text{ bytes} \approx \mathbf{141.1\text{ GB}}$
* **Gradients (in bfloat16):** $70.55\text{B} \times 2\text{ bytes} \approx \mathbf{141.1\text{ GB}}$
* **AdamW Optimizer States (fp32 master weights, momentum, variance):** $70.55\text{B} \times 8\text{ bytes} \approx \mathbf{564.4\text{ GB}}$
* **Total Static Pretraining Footprint:** $\approx \mathbf{846.6\text{ GB}}$

### Hardware Requirements Table

| Setup | Hardware | Method | Memory per GPU | Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Full Pretraining** | **32x H100 (80GB)** (4 Nodes) | PyTorch FSDP / ZeRO-3 | $\approx 26.5\text{ GB}$ static + activations | **Optimal** (3–4 weeks for 2T tokens) |
| **Full Pretraining** | **16x A100 (80GB)** (2 Nodes) | PyTorch FSDP + Act. Ckpt | $\approx 53.0\text{ GB}$ static + activations | **Supported** |
| **QLoRA Fine-Tuning** | **1x A100 / H100 (80GB)** | 4-bit NF4 Quantization + LoRA | $\approx 45.0\text{ GB}$ total VRAM | **Single-GPU Accessible!** |

---

## 📁 Repository Directory Structure

```text
cadbury-chocolate/
├── configs/
│   ├── model_70b_config.yaml     # Complete 70B architecture hyperparameters
│   ├── fsdp_config.yaml          # PyTorch FSDP FULL_SHARD cluster setup
│   ├── deepspeed_zero3.json      # DeepSpeed ZeRO-3 offload configuration
│   └── qlora_config.yaml         # 4-bit parameter-efficient fine-tuning config
├── data_collection/
│   ├── stream_hf_code.py         # Disk-free streaming ingestion from Hugging Face
│   ├── github_crawler.py         # Multi-language Git repository harvester
│   └── fim_formatter.py          # Fill-In-The-Middle (PSM/SPM) code transformations
├── data_processing/
│   ├── syntax_validator.py       # AST-based syntax checking (Python, JS, C, Rust)
│   ├── secret_scrubber.py        # PII, API tokens, and cryptographic key sanitizer
│   ├── minhash_dedup.py          # MinHash LSH near-deduplication (Jaccard similarity)
│   └── quality_filter.py         # Line-length, entropy, and autogen heuristics
├── tokenization/
│   ├── train_code_tokenizer.py   # Byte-Level BPE preserving code indentation & FIM
│   └── sequence_packer.py        # Zero-waste constant-length multipacking (8192 ctx)
├── training/
│   ├── pretrain_fsdp.py          # Multi-node FSDP pretrainer with FlashAttention-2
│   ├── train_qlora.py            # 4-bit QLoRA fine-tuning for 70B on 1x 80GB GPU
│   └── launch_slurm.sh           # HPC SLURM submission script (4 nodes x 8 GPUs)
├── utils/
│   ├── logger.py                 # Rank-aware distributed cluster logger
│   └── memory_tracker.py         # VRAM allocation and peak usage profiler
├── requirements.txt              # Production cluster dependencies
└── README.md
```

---

## 🚀 Execution Walkthrough (For Cluster Nodes)

### Stage 1: Data Ingestion (Streaming Mode)
Stream raw source code datasets (e.g. The Stack v2) without filling local hard drives:
```bash
python data_collection/stream_hf_code.py \
    --dataset "bigcode/the-stack-v2-train-smol" \
    --languages python rust go typescript cpp \
    --output "data/raw/raw_code_stream.jsonl" \
    --max-samples 500000
```

### Stage 2: Data Cleaning & Preprocessing
Run the cleaning pipeline to remove syntax errors, API keys, and near-duplicate files:
```bash
# 1. PII and secret redaction
python data_processing/secret_scrubber.py

# 2. Syntax validation and heuristic quality filtering
python data_processing/quality_filter.py

# 3. MinHash LSH deduplication
python data_processing/minhash_dedup.py
```

### Stage 3: Tokenization & Sequence Multipacking
Train the 128k Byte-Level BPE tokenizer and pack short files into fixed 8192-token windows:
```bash
# Train BPE tokenizer with FIM tags and indentation preservation
python tokenization/train_code_tokenizer.py \
    --input-file "data/processed/clean_corpus.txt" \
    --vocab-size 128256

# Multipack sequences into binary blocks (100% compute efficiency, zero padding waste)
python tokenization/sequence_packer.py
```

### Stage 4: Distributed 70B Pretraining
Launch multi-node training across GPU clusters:

#### Option A: Via SLURM (Standard High-Performance Cluster)
```bash
sbatch training/launch_slurm.sh
```

#### Option B: Via `torchrun` Directly (e.g., 2 Nodes $\times$ 8 GPUs = 16 GPUs)
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=cadbury_70b \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master-node-ip:29500 \
    training/pretrain_fsdp.py \
    --micro-batch-size 1 \
    --grad-accum-steps 16 \
    --seq-len 8192 \
    --max-steps 100000 \
    --output-dir checkpoints/cadbury-70b
```

### Stage 5: Fine-Tuning 70B on a Single 80GB GPU (QLoRA)
If you do not have a 32-GPU cluster, fine-tune the 70B model on instruction datasets using **4-bit NormalFloat quantization + LoRA**:
```bash
python training/train_qlora.py \
    --model-path "meta-llama/Meta-Llama-3-70B" \
    --batch-size 1 \
    --grad-accum 16 \
    --lora-r 64 \
    --lora-alpha 128 \
    --output-dir checkpoints/cadbury-70b-qlora
```

---

## 🧪 Benchmark & Evaluation Protocol

To benchmark code intelligence after training:
1. **HumanEval (Python Function Synthesis):** Measure pass@1 accuracy on 164 hand-crafted coding problems.
2. **MBPP (Mostly Basic Python Problems):** Measure pass@1 across 974 standard programming tasks.
3. **MultiPL-E (Multilingual Syntax):** Evaluate cross-lingual translation across JavaScript, TypeScript, Rust, Go, C++, and Java.

---

## 📜 License & Acknowledgments
* **Architecture:** Inspired by LLaMA (Meta AI), StarCoder (BigCode Project), and DeepSeek-Coder.
* **Open Source:** MIT License. Built for study, research, and high-performance cluster deployment.
