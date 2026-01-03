"""Environment capture utilities for experiment tracking.

This module provides tools to capture and compare the complete training
environment, enabling verification that experiments can be reproduced
on different machines.
"""
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class EnvironmentInfo:
    """Complete snapshot of the training environment.

    This captures all relevant system information needed to reproduce
    an experiment, including hardware, software versions, and configuration.
    """
    # Timestamp
    captured_at: str = ""

    # System info
    python_version: str = ""
    platform_system: str = ""
    platform_release: str = ""
    platform_machine: str = ""

    # PyTorch info
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""
    cuda_device_count: int = 0
    cuda_devices: list[dict[str, Any]] = field(default_factory=list)

    # Numeric library info
    numpy_version: str = ""

    # Environment variables (relevant ones)
    env_vars: dict[str, str] = field(default_factory=dict)

    # Git info (if in a repo)
    git_commit: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Package versions
    package_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentInfo":
        """Create from dictionary."""
        return cls(**data)


def _get_git_info() -> tuple[str, str, bool]:
    """Get git commit, branch, and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        dirty = result.returncode != 0

        return commit, branch, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", "", False


def _get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    packages = [
        "torch", "torchaudio", "numpy", "einops", "sentencepiece",
        "transformers", "datasets", "mamba_ssm", "flash_attn",
        "triton", "accelerate", "tensorboard", "wandb",
    ]

    versions = {}
    for pkg in packages:
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass  # Package not installed

    return versions


def capture_environment() -> EnvironmentInfo:
    """Capture the complete current environment.

    Returns:
        EnvironmentInfo with all system and configuration details.

    Example:
        >>> env = capture_environment()
        >>> print(f"PyTorch: {env.torch_version}")
        >>> print(f"CUDA: {env.cuda_version}")
    """
    # CUDA device info
    cuda_devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cuda_devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    # Relevant environment variables
    relevant_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_CUDA_ARCH_LIST",
    ]
    env_vars = {k: os.environ[k] for k in relevant_vars if k in os.environ}

    # Git info
    git_commit, git_branch, git_dirty = _get_git_info()

    return EnvironmentInfo(
        captured_at=datetime.now().isoformat(),
        python_version=sys.version,
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda or "",
        cudnn_version=str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "",
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        cuda_devices=cuda_devices,
        numpy_version=np.__version__,
        env_vars=env_vars,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        package_versions=_get_package_versions(),
    )


def save_environment(env: EnvironmentInfo, path: str | Path) -> None:
    """Save environment info to a JSON file.

    Args:
        env: Environment info to save.
        path: Path to save JSON file.

    Example:
        >>> env = capture_environment()
        >>> save_environment(env, "experiment_env.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(env.to_dict(), f, indent=2)


def load_environment(path: str | Path) -> EnvironmentInfo:
    """Load environment info from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded EnvironmentInfo.

    Example:
        >>> env = load_environment("experiment_env.json")
        >>> print(f"Experiment ran on: {env.platform_system}")
    """
    with open(path) as f:
        data = json.load(f)
    return EnvironmentInfo.from_dict(data)


@dataclass
class EnvironmentDiff:
    """Differences between two environments."""
    differences: dict[str, tuple[Any, Any]]
    warnings: list[str]
    is_compatible: bool


def compare_environments(
    env1: EnvironmentInfo,
    env2: EnvironmentInfo,
    strict: bool = False,
) -> EnvironmentDiff:
    """Compare two environments and identify differences.

    Args:
        env1: First environment (e.g., original training).
        env2: Second environment (e.g., current system).
        strict: If True, any difference is considered incompatible.

    Returns:
        EnvironmentDiff with differences and compatibility assessment.

    Example:
        >>> original = load_environment("train_env.json")
        >>> current = capture_environment()
        >>> diff = compare_environments(original, current)
        >>> if not diff.is_compatible:
        ...     for w in diff.warnings:
        ...         print(f"Warning: {w}")
    """
    differences: dict[str, tuple[Any, Any]] = {}
    warnings: list[str] = []
    is_compatible = True

    # Compare key fields
    critical_fields = [
        ("torch_version", "PyTorch version"),
        ("cuda_version", "CUDA version"),
        ("cudnn_version", "cuDNN version"),
    ]

    important_fields = [
        ("python_version", "Python version"),
        ("platform_system", "Operating system"),
        ("numpy_version", "NumPy version"),
    ]

    for field_name, display_name in critical_fields:
        val1 = getattr(env1, field_name)
        val2 = getattr(env2, field_name)
        if val1 != val2:
            differences[field_name] = (val1, val2)
            warnings.append(
                f"{display_name} differs: {val1} vs {val2}. "
                "This may cause numerical differences."
            )
            is_compatible = False

    for field_name, display_name in important_fields:
        val1 = getattr(env1, field_name)
        val2 = getattr(env2, field_name)
        if val1 != val2:
            differences[field_name] = (val1, val2)
            warnings.append(f"{display_name} differs: {val1} vs {val2}")
            if strict:
                is_compatible = False

    # Compare GPU configuration
    if env1.cuda_device_count != env2.cuda_device_count:
        differences["cuda_device_count"] = (env1.cuda_device_count, env2.cuda_device_count)
        warnings.append(
            f"GPU count differs: {env1.cuda_device_count} vs {env2.cuda_device_count}"
        )

    # Compare key package versions
    for pkg in set(env1.package_versions.keys()) | set(env2.package_versions.keys()):
        v1 = env1.package_versions.get(pkg, "not installed")
        v2 = env2.package_versions.get(pkg, "not installed")
        if v1 != v2:
            differences[f"package:{pkg}"] = (v1, v2)
            if pkg in ("torch", "mamba_ssm", "flash_attn"):
                warnings.append(f"Package {pkg} version differs: {v1} vs {v2}")
                if strict:
                    is_compatible = False

    return EnvironmentDiff(
        differences=differences,
        warnings=warnings,
        is_compatible=is_compatible,
    )
