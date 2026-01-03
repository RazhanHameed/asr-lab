"""Complete Fast Conformer ASR model."""

import torch

from asr_lab.models.base import ASRModel, CTCDecoder, DecoderType, TransducerDecoder
from asr_lab.models.conformer.config import ConformerConfig
from asr_lab.models.conformer.encoder import ConformerEncoder


class ConformerASRModel(ASRModel):
    """Fast Conformer ASR model with CTC or Transducer decoder.

    This model implements an efficient Conformer architecture:
    - Depthwise separable convolutional subsampling (8x)
    - Limited context self-attention (2.4x faster)
    - Macaron-style feed-forward modules
    - CTC or RNN-T decoder

    Example:
        >>> config = ConformerConfig.base()
        >>> model = ConformerASRModel(config)
        >>> features = torch.randn(2, 100, 80)  # (batch, time, n_mels)
        >>> output = model(features)
        >>> print(output["logits"].shape)  # (2, 12, vocab_size) with 8x downsample
    """

    def __init__(self, config: ConformerConfig) -> None:
        super().__init__(config)
        self.conformer_config = config

        # Encoder
        self._encoder = ConformerEncoder(config)

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
    def from_config(cls, config: ConformerConfig) -> "ConformerASRModel":
        """Create model from configuration."""
        return cls(config)

    @classmethod
    def small(cls) -> "ConformerASRModel":
        """Create small model (~5M params)."""
        return cls(ConformerConfig.small())

    @classmethod
    def base(cls) -> "ConformerASRModel":
        """Create base model (~19M params)."""
        return cls(ConformerConfig.base())

    @classmethod
    def large(cls) -> "ConformerASRModel":
        """Create large model (~50M params)."""
        return cls(ConformerConfig.large())

    @classmethod
    def fast_conformer(cls) -> "ConformerASRModel":
        """Create Fast Conformer with limited context attention."""
        return cls(ConformerConfig.fast_conformer())
