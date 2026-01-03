"""Configuration for Fast Conformer ASR models."""

from dataclasses import dataclass
from typing import Any

from asr_lab.models.base import DecoderType, ModelConfig, ModelType


@dataclass
class ConformerConfig(ModelConfig):
    """Configuration for Fast Conformer ASR model.

    Attributes:
        n_heads: Number of attention heads
        ffn_dim: Feed-forward dimension (typically 4x d_model)
        conv_kernel_size: Kernel size for depthwise convolution
        subsampling_factor: Total audio downsampling factor
        subsampling_conv_channels: Channels in subsampling convolutions
        attention_context_size: Left/right context for limited attention (-1 for full)
        use_relative_attention: Use relative positional encoding
        macaron_style: Use macaron-style FFN (FFN-Attn-Conv-FFN)
    """

    # Attention parameters
    n_heads: int = 8
    attention_context_size: int = -1  # -1 for full attention

    # FFN parameters
    ffn_dim: int = 1024
    macaron_style: bool = True

    # Convolution parameters
    conv_kernel_size: int = 31

    # Subsampling
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256

    # Positional encoding
    use_relative_attention: bool = True

    def __post_init__(self) -> None:
        self.model_type = ModelType.CONFORMER
        self.extra = {
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "conv_kernel_size": self.conv_kernel_size,
            "subsampling_factor": self.subsampling_factor,
            "subsampling_conv_channels": self.subsampling_conv_channels,
            "attention_context_size": self.attention_context_size,
            "use_relative_attention": self.use_relative_attention,
            "macaron_style": self.macaron_style,
        }

    @classmethod
    def small(cls) -> "ConformerConfig":
        """Small model (~5M params)."""
        return cls(
            d_model=176,
            n_layers=16,
            n_heads=4,
            ffn_dim=704,
            conv_kernel_size=31,
        )

    @classmethod
    def base(cls) -> "ConformerConfig":
        """Base model (~19M params)."""
        return cls(
            d_model=256,
            n_layers=18,
            n_heads=8,
            ffn_dim=1024,
            conv_kernel_size=31,
        )

    @classmethod
    def large(cls) -> "ConformerConfig":
        """Large model (~50M params)."""
        return cls(
            d_model=512,
            n_layers=18,
            n_heads=8,
            ffn_dim=2048,
            conv_kernel_size=31,
        )

    @classmethod
    def xlarge(cls) -> "ConformerConfig":
        """XLarge model (~100M params)."""
        return cls(
            d_model=512,
            n_layers=24,
            n_heads=8,
            ffn_dim=2048,
            conv_kernel_size=31,
        )

    @classmethod
    def fast_conformer(cls) -> "ConformerConfig":
        """Fast Conformer with limited context attention (2.4x faster)."""
        return cls(
            d_model=512,
            n_layers=18,
            n_heads=8,
            ffn_dim=2048,
            conv_kernel_size=9,  # Reduced from 31
            attention_context_size=128,  # Limited context
            subsampling_factor=8,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConformerConfig":
        """Create config from dictionary."""
        data = data.copy()
        data.pop("model_type", None)
        decoder_type = DecoderType(data.pop("decoder_type", "ctc"))
        return cls(decoder_type=decoder_type, **data)
