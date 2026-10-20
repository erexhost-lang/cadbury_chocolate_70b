"""
Distributed Rank-Aware Logger for Cadbury-Chocolate 70B
Ensures that only rank 0 prints verbose logs in multi-node clusters,
while all ranks can optionally write to individual log files for debugging.
"""

import logging
import os
import sys
from typing import Optional


def is_rank_zero() -> bool:
    """Check if the current process is the primary rank in distributed training."""
    # Check PyTorch distributed environment variables
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


def setup_logger(
    name: str = "cadbury",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a standard, structured logger with timestamps and rank prefix."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Format includes process rank to easily trace multi-node issues
    formatter = logging.Formatter(
        fmt=f"[%(asctime)s] [Rank {rank}/{world_size}] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler: Rank 0 gets normal logs, other ranks only get warnings/errors
    console_handler = logging.StreamHandler(sys.stdout)
    if is_rank_zero():
        console_handler.setLevel(level)
    else:
        console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler: All ranks can log to file if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_path = f"{log_file}.rank{rank}" if world_size > 1 else log_file
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def rank_zero_info(msg: str):
    """Convenience helper to print information strictly on rank 0."""
    if is_rank_zero():
        print(f"\033[92m[INFO]\033[0m {msg}", flush=True)
