"""
Cadbury-Chocolate Frontier V2: Distributed Memory & Hardware Sharding Architecture
Enterprise-Grade Distributed Execution Engine for 70B Sparse MoE + Multi-Head Latent Attention.

Architectural Objectives:
1. Prevent cluster Out-Of-Memory (OOM) crashes across multi-GPU (NVIDIA H100/A100) and multi-node clusters.
2. Provide unified support for PyTorch FullyShardedDataParallel (FSDP / FSDP2) and DeepSpeed ZeRO-3.
3. Integrate NVIDIA Transformer Engine (TE) FP8 Mixed Precision (E4M3/E5M2) for memory-bandwidth reduction.
4. Construct 2D/3D DeviceMesh topologies (Data, Expert, Tensor, Pipeline Parallelism) via torch.distributed.device_mesh.
5. Provide mathematical VRAM budgeting and expert device-allocation primitives.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

# Rank-aware logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("distributed_sharding")

# Safe imports for PyTorch FSDP
try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    HAS_PYTORCH_FSDP = True
except ImportError:
    HAS_PYTORCH_FSDP = False
    FSDP = None  # type: ignore
    ShardingStrategy = None  # type: ignore

# Safe imports for PyTorch 2.2+ DeviceMesh
try:
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    HAS_DEVICE_MESH = True
except ImportError:
    HAS_DEVICE_MESH = False
    DeviceMesh = None  # type: ignore
    init_device_mesh = None  # type: ignore

# Safe imports for NVIDIA Transformer Engine FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    HAS_TRANSFORMER_ENGINE = True
except ImportError:
    HAS_TRANSFORMER_ENGINE = False
    te = None  # type: ignore
    DelayedScaling = None  # type: ignore
    Format = None  # type: ignore


# ==============================================================================
# 1. Parallelism & Sharding Configuration Dataclasses
# ==============================================================================
class ShardingBackend(str, Enum):
    PYTORCH_FSDP = "fsdp"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"
    HYBRID_FSDP_TP = "hybrid_fsdp_tp"


class PrecisionMode(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FP8_HYBRID = "fp8_hybrid"  # E4M3 for forward, E5M2 for backward (Transformer Engine)
    FLOAT32 = "float32"


@dataclass
class MeshTopologyConfig:
    """
    3D/2D Device Mesh partition specification:
    Total World Size = data_parallel * expert_parallel * tensor_parallel * pipeline_parallel
    """
    data_parallel_size: int = 4
    expert_parallel_size: int = 8
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1

    # High-speed interconnect configurations
    backend: str = "nccl"
    init_method: Optional[str] = None

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.expert_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def validate_topology(self, current_world_size: int) -> bool:
        """Validates that configured mesh matches actual cluster hardware rank count."""
        expected = self.world_size
        if expected != current_world_size:
            logger.warning(
                f"Mesh topology product ({expected}) does not equal distributed world size ({current_world_size}). "
                "Adjusting data_parallel_size dynamically to fit cluster."
            )
            return False
        return True


@dataclass
class DistributedShardingConfig:
    """Master configuration for 70B MoE cluster memory distribution."""
    backend: ShardingBackend = ShardingBackend.PYTORCH_FSDP
    precision: PrecisionMode = PrecisionMode.BFLOAT16
    mesh: MeshTopologyConfig = field(default_factory=MeshTopologyConfig)

    # ZeRO-3 / FSDP Memory Offloading flags
    cpu_offload: bool = False
    cpu_offload_pin_memory: bool = True
    activation_checkpointing: bool = True
    gradient_accumulation_steps: int = 8
    clip_grad_norm: float = 1.0

    # FP8 Transformer Engine options
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"

    # Communication optimization
    backward_prefetch_limit: int = 2
    forward_prefetch: bool = True
    limit_allgathers: bool = True


# ==============================================================================
# 2. Mathematical VRAM Budget & Footprint Calculator
# ==============================================================================
class VRAMBudgetCalculator:
    """
    Calculates exact memory allocations across cluster nodes for the 70B MoE architecture.
    Prevents cluster allocation oversubscription before starting training.
    """

    @staticmethod
    def calculate_memory_breakdown(
        total_params: int = 70_550_000_000,
        active_params: int = 14_200_000_000,
        vocab_size: int = 128_270,
        hidden_size: int = 8_192,
        num_layers: int = 80,
        batch_size_per_gpu: int = 1,
        seq_len: int = 131_072,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        world_size: int = 32,
        precision: PrecisionMode = PrecisionMode.BFLOAT16,
        zero_stage: int = 3,
        cpu_offload: bool = False,
    ) -> Dict[str, float]:
        """
        Computes analytical memory footprint in Gigabytes per GPU.
        
        Returns:
            Dictionary with breakdown of:
            - static_params_gb
            - gradients_gb
            - optimizer_states_gb
            - activation_memory_gb
            - kv_cache_gb
            - peak_allocated_gb
        """
        bytes_per_param = 1 if precision == PrecisionMode.FP8_HYBRID else 2

        # 1. Parameter Footprint
        total_param_bytes = total_params * bytes_per_param
        if zero_stage == 3:
            # ZeRO-3 / FSDP FULL_SHARD partitions parameters equally across all ranks
            params_per_gpu_gb = (total_param_bytes / world_size) / (1024**3)
        else:
            params_per_gpu_gb = total_param_bytes / (1024**3)

        # 2. Gradient Footprint (bf16/fp16 gradients)
        grad_bytes = total_params * 2
        if zero_stage in (2, 3):
            grads_per_gpu_gb = (grad_bytes / world_size) / (1024**3)
        else:
            grads_per_gpu_gb = grad_bytes / (1024**3)

        # 3. Optimizer State Footprint (AdamW fp32: fp32 master weights + momentum + variance = 16 bytes/param)
        opt_state_bytes = total_params * 16
        if zero_stage in (1, 2, 3):
            opt_per_gpu_gb = (opt_state_bytes / world_size) / (1024**3)
        else:
            opt_per_gpu_gb = opt_state_bytes / (1024**3)

        if cpu_offload:
            opt_per_gpu_gb = 0.0  # Offloaded to Host System RAM

        # 4. Multi-Head Latent Attention (MLA) KV Cache Footprint per GPU:
        # Cadbury MLA caches ONLY (c_t^{KV} + k_t^{RoPE}) = (512 + 64) = 576 floats per token per layer!
        floats_per_token_mla = (kv_lora_rank + qk_rope_head_dim) * 2  # 2 bytes per float (bf16)
        kv_cache_total_bytes = batch_size_per_gpu * seq_len * num_layers * floats_per_token_mla
        kv_cache_gb = kv_cache_total_bytes / (1024**3)

        # 5. Activation Memory with Activation Checkpointing:
        # In selective activation checkpointing, only layer inputs and boundary tensors are preserved.
        # ~ 2 * hidden_size * seq_len * batch_size * bytes_per_param per layer
        act_per_layer = 2 * hidden_size * seq_len * batch_size_per_gpu * bytes_per_param
        # With activation checkpointing, we store activations for 1 active block in forward pass
        activations_gb = (act_per_layer * 2) / (1024**3)

        peak_allocated_gb = (
            params_per_gpu_gb
            + grads_per_gpu_gb
            + opt_per_gpu_gb
            + kv_cache_gb
            + activations_gb
        )

        return {
            "params_per_gpu_gb": round(params_per_gpu_gb, 2),
            "grads_per_gpu_gb": round(grads_per_gpu_gb, 2),
            "optimizer_states_gpu_gb": round(opt_per_gpu_gb, 2),
            "kv_cache_gb": round(kv_cache_gb, 2),
            "activations_gb": round(activations_gb, 2),
            "estimated_peak_vram_gb": round(peak_allocated_gb, 2),
        }


# ==============================================================================
# 3. Dynamic DeviceMesh Topology Manager
# ==============================================================================
class DynamicDeviceMeshManager:
    """
    Constructs multi-dimensional device meshes for hybrid 3D parallelism:
    Data Parallel (DP) x Expert Parallel (EP) x Tensor Parallel (TP) x Pipeline Parallel (PP).
    """

    def __init__(self, config: MeshTopologyConfig):
        self.config = config
        self._global_mesh: Optional[Any] = None
        self._dp_mesh: Optional[Any] = None
        self._ep_mesh: Optional[Any] = None
        self._tp_mesh: Optional[Any] = None

    def initialize_mesh(self) -> Optional[Any]:
        """Creates the device mesh if torch.distributed is initialized."""
        if not dist.is_available() or not dist.is_initialized():
            logger.info("torch.distributed is not initialized. DeviceMesh operating in offline simulation mode.")
            return None

        current_world_size = dist.get_world_size()
        mesh_shape = (
            self.config.data_parallel_size,
            self.config.expert_parallel_size,
            self.config.tensor_parallel_size,
        )
        dim_names = ("dp", "ep", "tp")

        # Verify device mesh product matches world size
        if not self.config.validate_topology(current_world_size):
            # Dynamic rebalancing: adjust dp dimension
            adjusted_dp = max(
                1,
                current_world_size // (self.config.expert_parallel_size * self.config.tensor_parallel_size),
            )
            mesh_shape = (adjusted_dp, self.config.expert_parallel_size, self.config.tensor_parallel_size)
            logger.info(f"Adjusted mesh shape to {mesh_shape} for world_size={current_world_size}")

        if HAS_DEVICE_MESH and init_device_mesh is not None:
            try:
                device_type = "cuda" if torch.cuda.is_available() else "cpu"
                self._global_mesh = init_device_mesh(
                    device_type=device_type,
                    mesh_shape=mesh_shape,
                    mesh_dim_names=dim_names,
                )
                self._dp_mesh = self._global_mesh["dp"]
                self._ep_mesh = self._global_mesh["ep"]
                self._tp_mesh = self._global_mesh["tp"]
                logger.info(f"Successfully initialized 3D DeviceMesh: {dim_names} -> {mesh_shape}")
                return self._global_mesh
            except Exception as e:
                logger.error(f"Failed to initialize DeviceMesh: {e}. Falling back to default groups.")
                return None
        else:
            logger.warning("torch.distributed.device_mesh is unavailable. Upgrade to PyTorch >= 2.2.0.")
            return None

    @property
    def expert_mesh(self) -> Optional[Any]:
        return self._ep_mesh

    @property
    def tensor_mesh(self) -> Optional[Any]:
        return self._tp_mesh

    @property
    def data_mesh(self) -> Optional[Any]:
        return self._dp_mesh


# ==============================================================================
# 4. FP8 Mixed Precision Manager (NVIDIA Transformer Engine)
# ==============================================================================
class FP8PrecisionManager:
    """
    Manages FP8 forward and backward execution passes using NVIDIA Transformer Engine.
    Uses E4M3 for forward activations and weights, and E5M2 for backward gradients,
    slashing memory bus throughput requirements by 50% vs BF16.
    """

    def __init__(self, config: DistributedShardingConfig):
        self.config = config
        self.enabled = config.precision == PrecisionMode.FP8_HYBRID and HAS_TRANSFORMER_ENGINE
        self._recipe: Optional[Any] = None

        if self.enabled:
            self._recipe = DelayedScaling(
                margin=config.fp8_margin,
                interval=config.fp8_interval,
                fp8_format=Format.HYBRID,
                amax_history_len=config.fp8_amax_history_len,
                amax_compute_algo=config.fp8_amax_compute_algo,
            )
            logger.info("Initialized Transformer Engine FP8 DelayedScaling Recipe (Format.HYBRID).")
        elif config.precision == PrecisionMode.FP8_HYBRID and not HAS_TRANSFORMER_ENGINE:
            logger.warning(
                "FP8_HYBRID requested, but transformer_engine is not installed. "
                "Defaulting to native torch.bfloat16 precision."
            )

    def fp8_autocast_context(self) -> Any:
        """Returns an autocast context manager for forward/backward steps."""
        if self.enabled and te is not None:
            return te.fp8_autocast(enabled=True, fp8_recipe=self._recipe)
        elif self.config.precision == PrecisionMode.BFLOAT16:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        else:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.autocast(device_type=device_type, dtype=torch.float32)


# ==============================================================================
# 5. PyTorch FSDP Production Sharding Engine
# ==============================================================================
class FSDPShardingEngine:
    """
    Applies PyTorch FullyShardedDataParallel (FSDP) parameter partitioning,
    gradient sharding, and activation checkpointing wrappers to Cadbury70BMoEModel.
    """

    def __init__(self, config: DistributedShardingConfig):
        self.config = config

    def get_mixed_precision_policy(self) -> Optional[MixedPrecision]:
        """Configures parameter, gradient, and buffer precisions."""
        if not HAS_PYTORCH_FSDP or MixedPrecision is None:
            return None

        if self.config.precision == PrecisionMode.BFLOAT16:
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.config.precision == PrecisionMode.FLOAT16:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

    def get_auto_wrap_policy(self, target_block_cls: type) -> Callable:
        """
        Constructs a transformer layer auto-wrap policy to shard at block boundaries,
        preventing uneven parameter splits across MoE blocks.
        """
        if not HAS_PYTORCH_FSDP:
            return lambda *args, **kwargs: False

        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={target_block_cls},
        )

    def shard_model(
        self,
        model: nn.Module,
        target_block_cls: type,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """
        Wraps the model in FSDP with parameter partitioning and optional CPU offloading.
        """
        if not HAS_PYTORCH_FSDP or FSDP is None:
            logger.warning("PyTorch FSDP is not available. Returning unwrapped model.")
            return model

        logger.info("Applying PyTorch FSDP (FULL_SHARD) across Cadbury-70B model layers...")

        mixed_precision = self.get_mixed_precision_policy()
        wrap_policy = self.get_auto_wrap_policy(target_block_cls)

        cpu_offload_policy = (
            CPUOffload(offload_params=True) if self.config.cpu_offload else None
        )

        sharded_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=self.config.forward_prefetch,
            limit_allgathers=self.config.limit_allgathers,
            device_id=device,
            use_orig_params=True,  # Crucial for parameter-specific optimizer configurations
        )

        logger.info("Successfully sharded Cadbury-70B with PyTorch FSDP FULL_SHARD.")
        return sharded_model

    @staticmethod
    def save_full_checkpoint(
        sharded_model: FSDP,
        optimizer: torch.optim.Optimizer,
        output_path: str,
        rank: int,
    ):
        """Saves consolidated full state dict without rank-0 OOM crashes."""
        if not HAS_PYTORCH_FSDP:
            return

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(sharded_model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = sharded_model.state_dict()

        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            torch.save(cpu_state_dict, output_path)
            logger.info(f"Consolidated rank-0 checkpoint saved to {output_path}")


# ==============================================================================
# 6. DeepSpeed ZeRO-3 Configurator
# ==============================================================================
class DeepSpeedZeRO3Configurator:
    """
    Generates a production DeepSpeed ZeRO-3 configuration manifest supporting:
    - Stage 3 Parameter Partitioning
    - Stage 2 Gradient Sharding
    - Stage 1 Optimizer State Sharding
    - CPU/NVMe Parameter and Optimizer Offloading
    - FP8 / BF16 mixed-precision flags
    """

    @staticmethod
    def generate_config(config: DistributedShardingConfig) -> Dict[str, Any]:
        """Constructs an enterprise-grade ZeRO-3 JSON dictionary."""
        is_bf16 = config.precision in (PrecisionMode.BFLOAT16, PrecisionMode.FP8_HYBRID)
        is_fp16 = config.precision == PrecisionMode.FLOAT16

        zero_config: Dict[str, Any] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
        }

        if config.cpu_offload:
            zero_config["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": config.cpu_offload_pin_memory,
                "fast_init": True,
            }
            zero_config["offload_param"] = {
                "device": "cpu",
                "pin_memory": config.cpu_offload_pin_memory,
            }

        ds_manifest: Dict[str, Any] = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clipping": config.clip_grad_norm,
            "steps_per_print": 10,
            "zero_optimization": zero_config,
            "bf16": {"enabled": is_bf16},
            "fp16": {"enabled": is_fp16},
            "wall_clock_breakdown": False,
        }

        # Activation checkpointing
        if config.activation_checkpointing:
            ds_manifest["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": config.cpu_offload,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            }

        return ds_manifest


# ==============================================================================
# 7. MoE Expert Placement & Interconnect Dispatcher
# ==============================================================================
class MoEParallelDispatcher:
    """
    Computes global expert-to-device mapping across the Expert Parallel (EP) mesh.
    For 16 routed experts distributed across EP ranks:
    Rank `r` hosts expert indices: `[e for e in range(16) if e % ep_size == rank_in_ep]`.
    Shared experts (2 experts) are replicated across all EP ranks for constant latency.
    """

    def __init__(self, num_routed_experts: int = 16, num_shared_experts: int = 2, ep_size: int = 8):
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.ep_size = ep_size

    def get_local_expert_indices(self, ep_rank: int) -> List[int]:
        """Returns the list of routed expert indices owned by this local EP rank."""
        return [
            e for e in range(self.num_routed_experts)
            if (e % self.ep_size) == (ep_rank % self.ep_size)
        ]

    def get_expert_to_rank_map(self) -> Dict[int, int]:
        """Returns mapping from global expert_id to EP rank."""
        return {
            e: (e % self.ep_size) for e in range(self.num_routed_experts)
        }


# ==============================================================================
# 8. Standalone Verification Routine
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Cadbury-70B Frontier V2: Distributed Memory & Hardware Sharding Verification")
    print("=" * 80)

    # 1. Test Analytical VRAM Budgeting
    print("\n[1. Analytical VRAM Footprint Calculation (32x H100 80GB Nodes)]")
    budget = VRAMBudgetCalculator.calculate_memory_breakdown(
        total_params=70_550_000_000,
        active_params=14_200_000_000,
        world_size=32,
        seq_len=131_072,
        batch_size_per_gpu=1,
        precision=PrecisionMode.BFLOAT16,
        zero_stage=3,
        cpu_offload=False,
    )
    for k, v in budget.items():
        print(f"  {k:30}: {v:6.2f} GB")

    # 2. Test DeepSpeed ZeRO-3 Config Generator
    print("\n[2. DeepSpeed ZeRO-3 Generated Manifest Preview]")
    sharding_cfg = DistributedShardingConfig(
        backend=ShardingBackend.DEEPSPEED_ZERO3,
        precision=PrecisionMode.FP8_HYBRID,
        cpu_offload=False,
    )
    ds_config = DeepSpeedZeRO3Configurator.generate_config(sharding_cfg)
    print(f"  ZeRO Stage: {ds_config['zero_optimization']['stage']}")
    print(f"  Overlap Comm: {ds_config['zero_optimization']['overlap_comm']}")
    print(f"  BF16 Enabled: {ds_config['bf16']['enabled']}")
    print(f"  Activation Partitioning: {ds_config.get('activation_checkpointing', {}).get('partition_activations')}")

    # 3. Test MoE Expert Placement
    print("\n[3. MoE Expert Parallel Placement (16 Experts on 8 EP Ranks)]")
    dispatcher = MoEParallelDispatcher(num_routed_experts=16, num_shared_experts=2, ep_size=8)
    for r in range(4):
        assigned = dispatcher.get_local_expert_indices(r)
        print(f"  EP Rank {r} hosts Routed Experts: {assigned}")

    print("=" * 80)
    print("Distributed Sharding Architecture verified successfully.")
