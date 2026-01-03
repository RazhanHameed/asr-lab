"""Complete Whisper-like ASR model."""

import torch

from asr_lab.models.base import ASRModel
from asr_lab.models.whisper.config import WhisperConfig
from asr_lab.models.whisper.decoder import WhisperDecoder
from asr_lab.models.whisper.encoder import WhisperEncoder


class WhisperASRModel(ASRModel):
    """Whisper-like encoder-decoder ASR model.

    This model implements a simplified Whisper architecture:
    - Convolutional audio encoder with Transformer layers
    - Autoregressive Transformer decoder with cross-attention
    - Sequence-to-sequence training with teacher forcing

    Example:
        >>> config = WhisperConfig.base()
        >>> model = WhisperASRModel(config)
        >>> features = torch.randn(2, 100, 80)  # (batch, time, n_mels)
        >>> targets = torch.randint(0, 256, (2, 20))  # (batch, target_len)
        >>> output = model(features, targets=targets, target_lengths=torch.tensor([20, 20]))
        >>> print(output["loss"])
    """

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__(config)
        self.whisper_config = config

        # Encoder
        self._encoder = WhisperEncoder(config)

        # Decoder
        self._decoder = WhisperDecoder(config)

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
            streaming: Not used (Whisper doesn't support streaming)

        Returns:
            Dictionary containing:
                - "logits": Output logits (batch, target_len, vocab_size)
                - "encoder_output": Encoder hidden states
                - "loss": Cross-entropy loss (if targets provided)
        """
        # Encode
        encoder_output, encoder_lengths = self.encoder(features, feature_lengths)

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

    def transcribe(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        streaming: bool = False,
        max_length: int = 200,
    ) -> list[list[int]]:
        """Transcribe audio features to token sequences.

        Args:
            features: Input audio features
            feature_lengths: Feature sequence lengths
            streaming: Not used
            max_length: Maximum output sequence length

        Returns:
            List of decoded token sequences
        """
        with torch.no_grad():
            encoder_output, encoder_lengths = self.encoder(features, feature_lengths)
            return self.decoder.decode(
                encoder_output, encoder_lengths, max_length=max_length
            )

    @classmethod
    def from_config(cls, config: WhisperConfig) -> "WhisperASRModel":
        """Create model from configuration."""
        return cls(config)

    @classmethod
    def tiny(cls) -> "WhisperASRModel":
        """Create tiny model (~5M params)."""
        return cls(WhisperConfig.tiny())

    @classmethod
    def base(cls) -> "WhisperASRModel":
        """Create base model (~19M params)."""
        return cls(WhisperConfig.base())

    @classmethod
    def small(cls) -> "WhisperASRModel":
        """Create small model (~50M params)."""
        return cls(WhisperConfig.small())

    @classmethod
    def medium(cls) -> "WhisperASRModel":
        """Create medium model (~100M params)."""
        return cls(WhisperConfig.medium())
