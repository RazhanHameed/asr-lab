"""Reproducibility utilities for ASR Lab.

This module provides tools for reproducible training across different hardware:
- Deterministic seeding for all random number generators
- Deterministic operation settings for PyTorch
- Hash utilities for verifying reproducibility
- Environment capture for experiment tracking

Inspired by Microsoft RepDL concepts, implemented natively.
"""
from asr_lab.reproducibility.seeding import (
    set_seed,
    get_random_state,
    set_random_state,
    RandomState,
)
from asr_lab.reproducibility.deterministic import (
    enable_deterministic_mode,
    disable_deterministic_mode,
    DeterministicContext,
    get_deterministic_status,
)
from asr_lab.reproducibility.hash import (
    get_tensor_hash,
    get_model_hash,
    get_state_dict_hash,
    verify_hash,
    HashMismatchError,
)
from asr_lab.reproducibility.environment import (
    capture_environment,
    save_environment,
    load_environment,
    compare_environments,
    EnvironmentInfo,
)

__all__ = [
    # Seeding
    "set_seed",
    "get_random_state",
    "set_random_state",
    "RandomState",
    # Deterministic mode
    "enable_deterministic_mode",
    "disable_deterministic_mode",
    "DeterministicContext",
    "get_deterministic_status",
    # Hashing
    "get_tensor_hash",
    "get_model_hash",
    "get_state_dict_hash",
    "verify_hash",
    "HashMismatchError",
    # Environment
    "capture_environment",
    "save_environment",
    "load_environment",
    "compare_environments",
    "EnvironmentInfo",
]
