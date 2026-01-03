"""Deterministic operation settings for PyTorch.

This module provides utilities to enable/disable deterministic operations
in PyTorch, ensuring reproducible results across different runs and hardware.
"""
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class DeterministicStatus:
    """Current status of deterministic settings."""
    torch_deterministic: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace: str | None
    torch_use_deterministic_algorithms: bool


def get_deterministic_status() -> DeterministicStatus:
    """Get current deterministic operation settings.

    Returns:
        DeterministicStatus with all current settings.
    """
    return DeterministicStatus(
        torch_deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic if torch.cuda.is_available() else False,
        cudnn_benchmark=torch.backends.cudnn.benchmark if torch.cuda.is_available() else False,
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch_use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def enable_deterministic_mode(warn_only: bool = False) -> None:
    """Enable fully deterministic operations in PyTorch.

    This configures PyTorch to use deterministic algorithms wherever possible,
    ensuring reproducible results across runs. Note that this may impact
    performance as some optimized non-deterministic algorithms will be disabled.

    Args:
        warn_only: If True, operations without deterministic implementations
            will only warn instead of raising an error.

    Deterministic settings applied:
        - torch.use_deterministic_algorithms(True)
        - CUBLAS_WORKSPACE_CONFIG=:4096:8 (for cuBLAS reproducibility)
        - cuDNN deterministic mode enabled
        - cuDNN benchmark disabled

    Example:
        >>> from asr_lab.reproducibility import enable_deterministic_mode, set_seed
        >>> enable_deterministic_mode()
        >>> set_seed(42)
        >>> # All operations are now deterministic

    Warning:
        Some operations may not have deterministic implementations and will
        raise an error (or warn if warn_only=True). Known problematic operations:
        - torch.nn.functional.interpolate with certain modes
        - torch.scatter_add
        - torch.index_add
    """
    # Set CUBLAS workspace config for deterministic cuBLAS operations
    # :4096:8 means 4096 bytes workspace with 8-way persistence
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enable deterministic algorithms globally
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    # cuDNN settings
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Disable TF32 for reproducibility (affects A100+)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def disable_deterministic_mode() -> None:
    """Disable deterministic mode, restoring default PyTorch behavior.

    This re-enables non-deterministic but faster algorithms, which is
    useful for inference or when reproducibility is not required.

    Example:
        >>> from asr_lab.reproducibility import disable_deterministic_mode
        >>> disable_deterministic_mode()
        >>> # Operations may now be non-deterministic but faster
    """
    # Disable deterministic algorithms
    torch.use_deterministic_algorithms(False)

    # Restore cuDNN defaults
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Re-enable TF32 for performance
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


@contextmanager
def DeterministicContext(warn_only: bool = False) -> Iterator[None]:
    """Context manager for deterministic operations.

    Enables deterministic mode within the context and restores previous
    settings on exit. Useful for ensuring specific code blocks are
    reproducible without affecting global settings.

    Args:
        warn_only: If True, warn instead of error for non-deterministic ops.

    Example:
        >>> from asr_lab.reproducibility import DeterministicContext
        >>> with DeterministicContext():
        ...     # All operations here are deterministic
        ...     output = model(input)
        >>> # Back to default (potentially non-deterministic) mode

    Yields:
        None
    """
    # Save current state
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if torch.cuda.is_available():
        previous_cudnn_deterministic = torch.backends.cudnn.deterministic
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        previous_tf32_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
        previous_tf32_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)

    try:
        enable_deterministic_mode(warn_only=warn_only)
        yield
    finally:
        # Restore previous state
        torch.use_deterministic_algorithms(previous_deterministic)

        if previous_cublas is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas
        elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = previous_cudnn_deterministic
            torch.backends.cudnn.benchmark = previous_cudnn_benchmark
            if previous_tf32_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
            if previous_tf32_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn


def check_reproducibility_requirements() -> list[str]:
    """Check system for reproducibility requirements.

    Returns a list of warnings about potential reproducibility issues.

    Returns:
        List of warning messages about potential issues.

    Example:
        >>> warnings = check_reproducibility_requirements()
        >>> for w in warnings:
        ...     print(f"Warning: {w}")
    """
    issues = []

    # Check PyTorch version (2.0+ has better deterministic support)
    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    if torch_version < (2, 0):
        issues.append(
            f"PyTorch version {torch.__version__} may have limited deterministic support. "
            "Consider upgrading to PyTorch 2.0+ for better reproducibility."
        )

    # Check CUDA availability
    if not torch.cuda.is_available():
        issues.append(
            "CUDA not available. Training will use CPU, which may have "
            "different numerical behavior than GPU training."
        )
    else:
        # Check for TF32 capable GPU
        if torch.cuda.get_device_capability()[0] >= 8:
            if hasattr(torch.backends.cuda, "matmul") and torch.backends.cuda.matmul.allow_tf32:
                issues.append(
                    "TF32 is enabled on Ampere+ GPU. This may cause non-deterministic results. "
                    "Use enable_deterministic_mode() to disable TF32."
                )

    # Check environment variables
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        issues.append(
            "CUBLAS_WORKSPACE_CONFIG not set. cuBLAS operations may be non-deterministic. "
            "Use enable_deterministic_mode() to set this automatically."
        )

    return issues
