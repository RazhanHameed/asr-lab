"""Configuration for SSM-based ASR models."""

from dataclasses import dataclass, field
from typing import Any

from asr_lab.models.base import DecoderType, ModelConfig, ModelType


@dataclass
class SSMConfig(ModelConfig):
    """Configuration for SSM (Mamba2) ASR model.

    Attributes:
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        headdim: Head dimension for multi-head operations
        attention_every_n_layers: Insert attention layer every N Mamba layers
        bidirectional: Use bidirectional SSM (offline mode)
        chunk_size: Chunk size for efficient computation
    """

    # SSM-specific parameters
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    attention_every_n_layers: int = 4
    bidirectional: bool = True
    chunk_size: int = 256

    # Frontend parameters
    conv_channels: list[int] = field(default_factory=lambda: [256, 256])
    conv_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    conv_strides: list[int] = field(default_factory=lambda: [2, 2])

    def __post_init__(self) -> None:
        self.model_type = ModelType.SSM
        # Store extra params in extra dict for serialization
        self.extra = {
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "attention_every_n_layers": self.attention_every_n_layers,
            "bidirectional": self.bidirectional,
            "chunk_size": self.chunk_size,
            "conv_channels": self.conv_channels,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_strides": self.conv_strides,
        }

    @classmethod
    def small(cls) -> "SSMConfig":
        """Small model (~5M params) for any GPU."""
        return cls(
            d_model=192,
            n_layers=12,
            d_state=32,
            headdim=32,
            use_flash_attention=False,
        )

    @classmethod
    def base(cls) -> "SSMConfig":
        """Base model (~19M params) for V100+."""
        return cls(
            d_model=256,
            n_layers=18,
            d_state=64,
            headdim=64,
        )

    @classmethod
    def large(cls) -> "SSMConfig":
        """Large model (~50M params) for A100+."""
        return cls(
            d_model=384,
            n_layers=24,
            d_state=64,
            headdim=64,
        )

    @classmethod
    def xlarge(cls) -> "SSMConfig":
        """XLarge model (~100M params) for A100 80GB+."""
        return cls(
            d_model=512,
            n_layers=32,
            d_state=128,
            headdim=128,
            chunk_size=128,
        )

    @classmethod
    def a100_optimized(cls) -> "SSMConfig":
        """Optimized for A100 GPUs."""
        return cls(
            d_model=384,
            n_layers=24,
            d_state=128,
            headdim=128,
            expand=2,
            use_bf16=True,
            chunk_size=256,
        )

    @classmethod
    def h100_optimized(cls) -> "SSMConfig":
        """Optimized for H100 GPUs with FP8."""
        return cls(
            d_model=512,
            n_layers=28,
            d_state=128,
            headdim=128,
            expand=2,
            use_bf16=True,
            chunk_size=256,
        )

    @classmethod
    def b200_optimized(cls) -> "SSMConfig":
        """Optimized for B200 GPUs with MXFP8."""
        return cls(
            d_model=512,
            n_layers=36,
            d_state=128,
            headdim=128,
            expand=2,
            use_bf16=True,
            chunk_size=256,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SSMConfig":
        """Create config from dictionary."""
        # Remove model_type and decoder_type as they're set in __post_init__
        data = data.copy()
        data.pop("model_type", None)
        decoder_type = DecoderType(data.pop("decoder_type", "ctc"))
        return cls(decoder_type=decoder_type, **data)
