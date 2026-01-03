"""Complete Omni-Router MoE ASR model."""

import torch

from asr_lab.models.base import ASRModel, CTCDecoder, DecoderType, ModelConfig, TransducerDecoder
from asr_lab.models.omni_router.config import OmniRouterConfig
from asr_lab.models.omni_router.encoder import OmniRouterEncoder


class OmniRouterASRModel(ASRModel):
    """Omni-Router MoE ASR model with shared routing.

    This model implements Apple's ASRU 2025 Omni-Router MoE architecture which
    uses shared routing across MoE layers for improved expert specialization,
    achieving 11.2% WER reduction compared to independent per-layer routing.

    Key features:
    - Frame stacking for efficient temporal downsampling
    - CAPE positional embeddings with training augmentation
    - Transformer encoder with MoE FFN layers
    - Shared router across layers within each expert group
    - Hierarchical expert configuration (e.g., "4x2-4x4-4x8")
    - CTC or Transducer decoder

    Example:
        >>> config = OmniRouterConfig.base()
        >>> model = OmniRouterASRModel(config)
        >>> features = torch.randn(2, 400, 80)  # (batch, time, n_mels)
        >>> output = model(features)
        >>> print(output["logits"].shape)  # (2, 100, vocab_size)
    """

    def __init__(self, config: OmniRouterConfig) -> None:
        super().__init__(config)
        self.omni_config = config

        self._encoder = OmniRouterEncoder(config)

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
            features: Input mel features of shape (batch, time, n_mels).
            feature_lengths: Feature lengths of shape (batch,).
            targets: Target token sequences of shape (batch, max_target_len).
            target_lengths: Target lengths of shape (batch,).
            streaming: Whether to use streaming/causal mode (uses config mask_mode).

        Returns:
            Dictionary containing:
                - "logits": Output logits of shape (batch, time', vocab_size)
                - "encoder_output": Encoder hidden states
                - "encoder_lengths": Output sequence lengths
                - "loss": CTC/RNNT loss (if targets provided)
        """
        encoder_output, encoder_lengths = self.encoder(
            features, feature_lengths, streaming=streaming
        )

        decoder_output = self.decoder(
            encoder_output,
            encoder_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

        result: dict[str, torch.Tensor] = {
            "encoder_output": encoder_output,
            **decoder_output,
        }
        if encoder_lengths is not None:
            result["encoder_lengths"] = encoder_lengths

        return result

    @classmethod
    def from_config(cls, config: ModelConfig) -> "OmniRouterASRModel":
        """Create model from configuration."""
        if not isinstance(config, OmniRouterConfig):
            raise TypeError(f"Expected OmniRouterConfig, got {type(config).__name__}")
        return cls(config)

    @classmethod
    def small(cls) -> "OmniRouterASRModel":
        """Create small dense model (~85M params)."""
        return cls(OmniRouterConfig.small())

    @classmethod
    def base(cls) -> "OmniRouterASRModel":
        """Create base MoE model (~250M params)."""
        return cls(OmniRouterConfig.base())

    @classmethod
    def large(cls) -> "OmniRouterASRModel":
        """Create large MoE model (~613M params)."""
        return cls(OmniRouterConfig.large())

    @classmethod
    def streaming(cls) -> "OmniRouterASRModel":
        """Create streaming-optimized MoE model."""
        return cls(OmniRouterConfig.streaming())
