"""Complete SSM-based ASR model."""

import torch

from asr_lab.models.base import ASRModel, CTCDecoder, DecoderType, TransducerDecoder
from asr_lab.models.ssm.config import SSMConfig
from asr_lab.models.ssm.encoder import SSMEncoder


class SSMASRModel(ASRModel):
    """State Space Model ASR with Mamba2 encoder.

    This model combines:
    - Convolutional frontend for initial feature processing
    - Mamba2 SSM blocks for efficient sequence modeling
    - Optional Flash Attention layers for global context
    - CTC or Transducer decoder for output

    Example:
        >>> config = SSMConfig.base()
        >>> model = SSMASRModel(config)
        >>> features = torch.randn(2, 100, 80)  # (batch, time, n_mels)
        >>> output = model(features)
        >>> print(output["logits"].shape)  # (2, 25, vocab_size)
    """

    def __init__(self, config: SSMConfig) -> None:
        super().__init__(config)
        self.ssm_config = config

        # Encoder
        self._encoder = SSMEncoder(config)

        # Decoder
        if config.decoder_type == DecoderType.CTC:
            self._decoder = CTCDecoder(config)
        elif config.decoder_type == DecoderType.TRANSDUCER:
            self._decoder = TransducerDecoder(config)
        else:
            raise ValueError(f"Unsupported decoder type: {config.decoder_type}")

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through encoder and decoder.

        Args:
            features: Input mel features of shape (batch, time, n_mels)
            feature_lengths: Feature lengths of shape (batch,)
            targets: Target token sequences of shape (batch, max_target_len)
            target_lengths: Target lengths of shape (batch,)
            streaming: Whether to use streaming/causal mode

        Returns:
            Dictionary containing:
                - "logits": Output logits
                - "encoder_output": Encoder hidden states
                - "loss": CTC/RNNT loss (if targets provided)
        """
        # Encode
        encoder_output, encoder_lengths = self.encoder(
            features, feature_lengths, streaming=streaming
        )

        # Decode
        decoder_output = self.decoder(
            encoder_output,
            encoder_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

        return {
            "encoder_output": encoder_output,
            "encoder_lengths": encoder_lengths,
            **decoder_output,
        }

    @classmethod
    def from_config(cls, config: SSMConfig) -> "SSMASRModel":
        """Create model from configuration."""
        return cls(config)

    @classmethod
    def small(cls) -> "SSMASRModel":
        """Create small model (~5M params)."""
        return cls(SSMConfig.small())

    @classmethod
    def base(cls) -> "SSMASRModel":
        """Create base model (~19M params)."""
        return cls(SSMConfig.base())

    @classmethod
    def large(cls) -> "SSMASRModel":
        """Create large model (~50M params)."""
        return cls(SSMConfig.large())

    @classmethod
    def a100_optimized(cls) -> "SSMASRModel":
        """Create A100-optimized model."""
        return cls(SSMConfig.a100_optimized())

    @classmethod
    def h100_optimized(cls) -> "SSMASRModel":
        """Create H100-optimized model."""
        return cls(SSMConfig.h100_optimized())

    @classmethod
    def b200_optimized(cls) -> "SSMASRModel":
        """Create B200-optimized model."""
        return cls(SSMConfig.b200_optimized())
