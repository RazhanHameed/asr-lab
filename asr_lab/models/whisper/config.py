"""Configuration for Whisper-like ASR models."""

from dataclasses import dataclass
from typing import Any

from asr_lab.models.base import DecoderType, ModelConfig, ModelType


@dataclass
class WhisperConfig(ModelConfig):
    """Configuration for Whisper-like ASR model.

    Attributes:
        n_encoder_layers: Number of encoder transformer layers
        n_decoder_layers: Number of decoder transformer layers
        n_heads: Number of attention heads
        encoder_ffn_dim: Encoder feed-forward dimension
        decoder_ffn_dim: Decoder feed-forward dimension
        max_source_positions: Maximum encoder sequence length
        max_target_positions: Maximum decoder sequence length
        tie_embeddings: Whether to tie input/output embeddings
    """

    # Encoder parameters
    n_encoder_layers: int = 6
    encoder_ffn_dim: int = 1024

    # Decoder parameters
    n_decoder_layers: int = 6
    decoder_ffn_dim: int = 1024

    # Attention parameters
    n_heads: int = 8

    # Positional encoding
    max_source_positions: int = 1500
    max_target_positions: int = 448

    # Embedding
    tie_embeddings: bool = True

    # Frontend
    conv_channels: int = 256
    conv_kernel_size: int = 3

    def __post_init__(self) -> None:
        self.model_type = ModelType.WHISPER
        self.decoder_type = DecoderType.ATTENTION
        self.extra = {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "n_heads": self.n_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "max_source_positions": self.max_source_positions,
            "max_target_positions": self.max_target_positions,
            "tie_embeddings": self.tie_embeddings,
            "conv_channels": self.conv_channels,
            "conv_kernel_size": self.conv_kernel_size,
        }

    @classmethod
    def tiny(cls) -> "WhisperConfig":
        """Tiny model (~5M params)."""
        return cls(
            d_model=192,
            n_layers=4,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_heads=4,
            encoder_ffn_dim=768,
            decoder_ffn_dim=768,
        )

    @classmethod
    def base(cls) -> "WhisperConfig":
        """Base model (~19M params)."""
        return cls(
            d_model=256,
            n_layers=6,
            n_encoder_layers=6,
            n_decoder_layers=6,
            n_heads=8,
            encoder_ffn_dim=1024,
            decoder_ffn_dim=1024,
        )

    @classmethod
    def small(cls) -> "WhisperConfig":
        """Small model (~50M params)."""
        return cls(
            d_model=384,
            n_layers=12,
            n_encoder_layers=12,
            n_decoder_layers=12,
            n_heads=8,
            encoder_ffn_dim=1536,
            decoder_ffn_dim=1536,
        )

    @classmethod
    def medium(cls) -> "WhisperConfig":
        """Medium model (~100M params)."""
        return cls(
            d_model=512,
            n_layers=16,
            n_encoder_layers=16,
            n_decoder_layers=16,
            n_heads=8,
            encoder_ffn_dim=2048,
            decoder_ffn_dim=2048,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhisperConfig":
        """Create config from dictionary."""
        data = data.copy()
        data.pop("model_type", None)
        data.pop("decoder_type", None)
        return cls(**data)
