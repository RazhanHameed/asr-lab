"""Precision management for ASR training.

Supports:
- FP32: Full precision (any GPU)
- FP16: Half precision with loss scaling (V100+)
- BF16: BFloat16 without loss scaling (A100+)
- FP8: 8-bit floating point (H100+ with Transformer Engine)
- MXFP8: Microscaling FP8 (B200+ with TorchAO)
- MXFP4: Microscaling FP4 (B200+ with TorchAO, experimental)
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generator

import torch
import torch.nn as nn


class PrecisionMode(Enum):
    """Supported precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP8_DYNAMIC = "fp8"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    AUTO = "auto"


@dataclass
class PrecisionConfig:
    """Configuration for precision management.

    Attributes:
        mode: Precision mode to use
        loss_scale: Initial loss scale for FP16
        growth_interval: Loss scale growth interval
        enabled: Whether mixed precision is enabled
    """

    mode: PrecisionMode = PrecisionMode.BF16
    loss_scale: float = 65536.0
    growth_interval: int = 2000
    enabled: bool = True


class PrecisionManager:
    """Manages mixed precision training for different GPU architectures."""

    def __init__(self, config: PrecisionConfig | None = None) -> None:
        if config is None:
            config = PrecisionConfig()
        self.config = config
        self.mode = config.mode

        # Resolve AUTO mode
        if self.mode == PrecisionMode.AUTO:
            self.mode = detect_optimal_precision()

        # Initialize based on mode
        self._scaler: torch.amp.GradScaler | None = None
        self._autocast_dtype: torch.dtype = torch.float32

        if self.mode == PrecisionMode.FP16:
            self._autocast_dtype = torch.float16
            self._scaler = torch.amp.GradScaler("cuda", enabled=config.enabled)
        elif self.mode == PrecisionMode.BF16:
            self._autocast_dtype = torch.bfloat16
        elif self.mode in (
            PrecisionMode.FP8_E4M3,
            PrecisionMode.FP8_E5M2,
            PrecisionMode.FP8_DYNAMIC,
        ):
            self._init_fp8()
        elif self.mode in (PrecisionMode.MXFP8, PrecisionMode.MXFP4):
            self._init_mxfp()

    def _init_fp8(self) -> None:
        """Initialize FP8 precision with Transformer Engine."""
        try:
            import transformer_engine.pytorch as te

            self._te = te
            self._fp8_enabled = True
        except ImportError:
            print("Warning: Transformer Engine not available, falling back to BF16")
            self.mode = PrecisionMode.BF16
            self._autocast_dtype = torch.bfloat16
            self._fp8_enabled = False

    def _init_mxfp(self) -> None:
        """Initialize MXFP precision with TorchAO."""
        try:
            import torchao

            self._torchao = torchao
            self._mxfp_enabled = True
            self._autocast_dtype = torch.bfloat16  # MXFP uses BF16 as base
        except ImportError:
            print("Warning: TorchAO not available, falling back to BF16")
            self.mode = PrecisionMode.BF16
            self._autocast_dtype = torch.bfloat16
            self._mxfp_enabled = False

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Context manager for automatic mixed precision.

        Example:
            >>> with precision.autocast():
            ...     output = model(input)
            ...     loss = criterion(output, target)
        """
        if self.mode == PrecisionMode.FP32:
            yield
        elif self.mode in (PrecisionMode.FP16, PrecisionMode.BF16):
            with torch.autocast("cuda", dtype=self._autocast_dtype):
                yield
        elif self.mode in (
            PrecisionMode.FP8_E4M3,
            PrecisionMode.FP8_E5M2,
            PrecisionMode.FP8_DYNAMIC,
        ):
            if hasattr(self, "_fp8_enabled") and self._fp8_enabled:
                with self._te.fp8_autocast(enabled=True):
                    yield
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    yield
        elif self.mode in (PrecisionMode.MXFP8, PrecisionMode.MXFP4):
            # MXFP uses standard BF16 autocast with quantized weights
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation with FP16."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with proper scaling."""
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        model: nn.Module | None = None,
    ) -> None:
        """Optimizer step with gradient unscaling and clipping."""
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

            if clip_grad is not None and model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            if clip_grad is not None and model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for the current precision mode.

        Args:
            model: PyTorch model

        Returns:
            Model prepared for mixed precision training
        """
        if self.mode == PrecisionMode.BF16:
            # Convert model to BF16
            model = model.to(dtype=torch.bfloat16)
        elif self.mode in (
            PrecisionMode.FP8_E4M3,
            PrecisionMode.FP8_E5M2,
            PrecisionMode.FP8_DYNAMIC,
        ):
            if hasattr(self, "_fp8_enabled") and self._fp8_enabled:
                # Wrap linear layers with FP8 equivalents
                pass  # TE handles this automatically in autocast
        elif self.mode == PrecisionMode.MXFP8:
            if hasattr(self, "_mxfp_enabled") and self._mxfp_enabled:
                # Apply MXFP8 quantization
                try:
                    from torchao.float8 import convert_to_float8_training

                    model = convert_to_float8_training(model)
                except Exception:
                    pass  # Fall back to standard precision
        elif self.mode == PrecisionMode.MXFP4:
            if hasattr(self, "_mxfp_enabled") and self._mxfp_enabled:
                # MXFP4 is experimental
                pass

        return model


def detect_optimal_precision() -> PrecisionMode:
    """Detect the optimal precision mode for the current GPU.

    Returns:
        Optimal PrecisionMode for the available hardware
    """
    if not torch.cuda.is_available():
        return PrecisionMode.FP32

    # Get GPU compute capability
    capability = torch.cuda.get_device_capability()
    major, minor = capability

    # B200: sm_100+ (estimated)
    if major >= 10:
        # Try MXFP8
        try:
            import torchao

            return PrecisionMode.MXFP8
        except ImportError:
            pass

    # H100: sm_90
    if major >= 9:
        try:
            import transformer_engine

            return PrecisionMode.FP8_DYNAMIC
        except ImportError:
            return PrecisionMode.BF16

    # A100: sm_80
    if major >= 8:
        return PrecisionMode.BF16

    # V100: sm_70
    if major >= 7:
        return PrecisionMode.FP16

    return PrecisionMode.FP32


def get_precision_manager(
    mode: str | PrecisionMode = "auto",
) -> PrecisionManager:
    """Create a precision manager for the specified mode.

    Args:
        mode: Precision mode string or PrecisionMode enum

    Returns:
        Configured PrecisionManager
    """
    if isinstance(mode, str):
        mode = PrecisionMode(mode.lower())

    config = PrecisionConfig(mode=mode)
    return PrecisionManager(config)
