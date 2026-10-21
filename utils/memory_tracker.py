"""
GPU & Host Memory Profiler for 70B Distributed Training
Tracks CUDA VRAM allocation, peak memory spikes, and host RAM.
"""

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import torch
from typing import Dict, Any


class MemoryTracker:
    """Profiles GPU VRAM and System RAM during training iterations."""

    def __init__(self, device: int = 0):
        self.device = device
        self.is_cuda = torch.cuda.is_available()

    def reset_peak_memory(self):
        """Reset the peak memory tracking statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_summary(self) -> Dict[str, Any]:
        """Return memory statistics formatted in Gigabytes."""
        summary = {}

        # System RAM usage
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            ram_gb = process.memory_info().rss / (1024 ** 3)
        else:
            try:
                import resource
                # ru_maxrss is in kilobytes on Linux
                ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            except Exception:
                ram_gb = 0.0
        summary["host_ram_used_gb"] = round(ram_gb, 2)

        # CUDA VRAM usage
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 3)

            summary["gpu_allocated_gb"] = round(allocated, 2)
            summary["gpu_reserved_gb"] = round(reserved, 2)
            summary["gpu_peak_allocated_gb"] = round(max_allocated, 2)
            summary["gpu_peak_reserved_gb"] = round(max_reserved, 2)

            total_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            summary["gpu_total_vram_gb"] = round(total_vram, 2)
            summary["gpu_utilization_pct"] = round((reserved / total_vram) * 100, 1)

        return summary

    def print_summary(self, prefix: str = ""):
        """Log a concise one-line memory summary."""
        stats = self.get_summary()
        if self.is_cuda:
            msg = (
                f"{prefix} VRAM: {stats['gpu_allocated_gb']:.1f}GB alloc / "
                f"{stats['gpu_reserved_gb']:.1f}GB res (Peak: {stats['gpu_peak_allocated_gb']:.1f}GB / "
                f"{stats['gpu_total_vram_gb']:.1f}GB | {stats['gpu_utilization_pct']}%) | "
                f"RAM: {stats['host_ram_used_gb']:.1f}GB"
            )
        else:
            msg = f"{prefix} RAM: {stats['host_ram_used_gb']:.1f}GB (CPU Mode)"
        print(f"\033[94m[MEMORY]\033[0m {msg}", flush=True)
