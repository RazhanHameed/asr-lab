"""Hash utilities for verifying reproducibility.

This module provides functions to compute cryptographic hashes of tensors
and models, enabling verification that training produces identical results
across different runs or hardware.
"""
import hashlib
from typing import Any

import torch


class HashMismatchError(Exception):
    """Raised when hash verification fails."""

    def __init__(self, expected: str, actual: str, context: str = ""):
        self.expected = expected
        self.actual = actual
        self.context = context
        message = f"Hash mismatch: expected {expected}, got {actual}"
        if context:
            message += f" ({context})"
        super().__init__(message)


def get_tensor_hash(tensor: torch.Tensor, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a tensor.

    The tensor is converted to CPU and numpy format before hashing
    to ensure consistent results regardless of device.

    Args:
        tensor: The tensor to hash.
        algorithm: Hash algorithm to use (default: sha256).
            Supported: md5, sha1, sha256, sha512.

    Returns:
        Hexadecimal hash string.

    Example:
        >>> x = torch.randn(100, 100)
        >>> hash1 = get_tensor_hash(x)
        >>> hash2 = get_tensor_hash(x.clone())
        >>> assert hash1 == hash2  # Same data, same hash
    """
    hasher = hashlib.new(algorithm)
    # Convert to contiguous CPU tensor with consistent dtype for hashing
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    hasher.update(data)
    return hasher.hexdigest()


def get_model_hash(
    model: torch.nn.Module,
    algorithm: str = "sha256",
    include_buffers: bool = True,
) -> str:
    """Compute cryptographic hash of all model parameters.

    Hashes are computed by iterating through named parameters in a
    consistent order, making the hash reproducible across runs.

    Args:
        model: The model to hash.
        algorithm: Hash algorithm to use (default: sha256).
        include_buffers: Whether to include non-parameter buffers
            (e.g., running mean/var in BatchNorm).

    Returns:
        Hexadecimal hash string representing the entire model state.

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> hash_before = get_model_hash(model)
        >>> model(torch.randn(1, 10))  # Forward doesn't change params
        >>> hash_after = get_model_hash(model)
        >>> assert hash_before == hash_after
    """
    hasher = hashlib.new(algorithm)

    # Hash parameters in sorted order for consistency
    for name in sorted(n for n, _ in model.named_parameters()):
        param = dict(model.named_parameters())[name]
        data = param.detach().cpu().contiguous().numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(data)

    if include_buffers:
        for name in sorted(n for n, _ in model.named_buffers()):
            buffer = dict(model.named_buffers())[name]
            data = buffer.detach().cpu().contiguous().numpy().tobytes()
            hasher.update(name.encode("utf-8"))
            hasher.update(data)

    return hasher.hexdigest()


def get_state_dict_hash(
    state_dict: dict[str, Any],
    algorithm: str = "sha256",
) -> str:
    """Compute cryptographic hash of a state dictionary.

    Useful for hashing checkpoints, optimizer states, or any other
    state dictionaries containing tensors.

    Args:
        state_dict: Dictionary containing tensors and other values.
        algorithm: Hash algorithm to use (default: sha256).

    Returns:
        Hexadecimal hash string.

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> hash1 = get_state_dict_hash(model.state_dict())
        >>> # Save and reload
        >>> torch.save(model.state_dict(), "model.pt")
        >>> loaded = torch.load("model.pt")
        >>> hash2 = get_state_dict_hash(loaded)
        >>> assert hash1 == hash2
    """
    hasher = hashlib.new(algorithm)

    def hash_value(key: str, value: Any) -> None:
        hasher.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().contiguous().numpy().tobytes()
            hasher.update(data)
        elif isinstance(value, dict):
            for k in sorted(value.keys()):
                hash_value(f"{key}.{k}", value[k])
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                hash_value(f"{key}[{i}]", v)
        else:
            # For other types, use string representation
            hasher.update(str(value).encode("utf-8"))

    for key in sorted(state_dict.keys()):
        hash_value(key, state_dict[key])

    return hasher.hexdigest()


def verify_hash(
    obj: torch.Tensor | torch.nn.Module | dict[str, Any],
    expected_hash: str,
    algorithm: str = "sha256",
    context: str = "",
) -> bool:
    """Verify that an object's hash matches expected value.

    Args:
        obj: Tensor, model, or state dict to verify.
        expected_hash: Expected hash value.
        algorithm: Hash algorithm used for expected_hash.
        context: Optional context string for error messages.

    Returns:
        True if hashes match.

    Raises:
        HashMismatchError: If hashes don't match.

    Example:
        >>> x = torch.randn(10)
        >>> expected = get_tensor_hash(x)
        >>> verify_hash(x, expected)  # Returns True
        >>> x[0] = 999  # Modify tensor
        >>> verify_hash(x, expected)  # Raises HashMismatchError
    """
    if isinstance(obj, torch.Tensor):
        actual_hash = get_tensor_hash(obj, algorithm)
    elif isinstance(obj, torch.nn.Module):
        actual_hash = get_model_hash(obj, algorithm)
    elif isinstance(obj, dict):
        actual_hash = get_state_dict_hash(obj, algorithm)
    else:
        raise TypeError(f"Cannot hash object of type {type(obj)}")

    if actual_hash != expected_hash:
        raise HashMismatchError(expected_hash, actual_hash, context)

    return True


def print_hash(obj: torch.Tensor | torch.nn.Module | dict[str, Any]) -> None:
    """Print the hash of an object to stdout.

    Convenience function for debugging and verification.

    Args:
        obj: Tensor, model, or state dict to hash.

    Example:
        >>> model = torch.nn.Linear(10, 10)
        >>> print_hash(model)
        Model hash: a1b2c3d4...
    """
    if isinstance(obj, torch.Tensor):
        hash_val = get_tensor_hash(obj)
        print(f"Tensor hash: {hash_val}")
    elif isinstance(obj, torch.nn.Module):
        hash_val = get_model_hash(obj)
        print(f"Model hash: {hash_val}")
    elif isinstance(obj, dict):
        hash_val = get_state_dict_hash(obj)
        print(f"State dict hash: {hash_val}")
    else:
        raise TypeError(f"Cannot hash object of type {type(obj)}")
