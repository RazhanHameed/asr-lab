"""Deterministic seeding utilities for reproducible training.

This module ensures all random number generators are seeded consistently
across Python, NumPy, and PyTorch (both CPU and CUDA).
"""
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RandomState:
    """Container for all random number generator states.

    Captures the complete state of all RNGs to enable exact resumption
    of training from any checkpoint.
    """
    python_state: tuple[Any, ...]
    numpy_state: dict[str, Any]
    torch_cpu_state: torch.Tensor
    torch_cuda_states: list[torch.Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {
            "python_state": self.python_state,
            "numpy_state": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.numpy_state.items()
            },
            "torch_cpu_state": self.torch_cpu_state.tolist(),
        }
        if self.torch_cuda_states is not None:
            result["torch_cuda_states"] = [s.tolist() for s in self.torch_cuda_states]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RandomState":
        """Restore from serialized dictionary."""
        numpy_state = {
            k: np.array(v) if isinstance(v, list) and k in ("state",) else v
            for k, v in data["numpy_state"].items()
        }
        cuda_states = None
        if "torch_cuda_states" in data and data["torch_cuda_states"]:
            cuda_states = [torch.tensor(s, dtype=torch.uint8) for s in data["torch_cuda_states"]]
        return cls(
            python_state=tuple(data["python_state"]),
            numpy_state=numpy_state,
            torch_cpu_state=torch.tensor(data["torch_cpu_state"], dtype=torch.uint8),
            torch_cuda_states=cuda_states,
        )


def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
    """Set random seed for all random number generators.

    This ensures reproducible results by seeding:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU generator
    - PyTorch CUDA generators (all devices)
    - Environment variables for hash seeding

    Args:
        seed: The seed value to use (must be non-negative integer).
        deterministic_cudnn: If True, also sets cuDNN to deterministic mode.
            This may impact performance but ensures reproducibility.

    Example:
        >>> from asr_lab.reproducibility import set_seed
        >>> set_seed(42)
        >>> # All subsequent random operations are now reproducible
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Set Python hash seed (must be done before any hashing)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if deterministic_cudnn:
            # cuDNN settings for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_random_state() -> RandomState:
    """Capture the current state of all random number generators.

    Returns a RandomState object that can be used to restore the exact
    state later, enabling perfect resumption of training.

    Returns:
        RandomState containing all RNG states.

    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> set_random_state(state)  # Restore to saved state
    """
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [
            torch.cuda.get_rng_state(device=i)
            for i in range(torch.cuda.device_count())
        ]

    return RandomState(
        python_state=random.getstate(),
        numpy_state=dict(np.random.get_state(legacy=False)),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=cuda_states,
    )


def set_random_state(state: RandomState) -> None:
    """Restore all random number generators to a saved state.

    Args:
        state: RandomState object from get_random_state().

    Example:
        >>> state = get_random_state()
        >>> x1 = torch.rand(10)
        >>> set_random_state(state)
        >>> x2 = torch.rand(10)
        >>> assert torch.equal(x1, x2)  # Same random values
    """
    # Python random
    random.setstate(state.python_state)

    # NumPy
    np.random.set_state(state.numpy_state)

    # PyTorch CPU
    torch.set_rng_state(state.torch_cpu_state)

    # PyTorch CUDA
    if state.torch_cuda_states is not None and torch.cuda.is_available():
        for i, cuda_state in enumerate(state.torch_cuda_states):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(cuda_state, device=i)


def worker_init_fn(worker_id: int, base_seed: int | None = None) -> None:
    """Initialize DataLoader worker with deterministic seed.

    Use this as the worker_init_fn for torch.utils.data.DataLoader
    to ensure reproducible data loading across workers.

    Args:
        worker_id: Worker ID (provided by DataLoader).
        base_seed: Base seed to combine with worker_id. If None,
            uses the initial seed from torch.initial_seed().

    Example:
        >>> from functools import partial
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42),
        ... )
    """
    if base_seed is None:
        base_seed = torch.initial_seed() % (2**32)

    worker_seed = base_seed + worker_id
    set_seed(worker_seed, deterministic_cudnn=False)
