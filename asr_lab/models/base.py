"""Base classes for ASR models.

This module defines the abstract interfaces for all ASR model components,
enabling modular composition of encoders and decoders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class ModelType(Enum):
    """Supported model architectures."""

    SSM = "ssm"
    WHISPER = "whisper"
    CONFORMER = "conformer"


class DecoderType(Enum):
    """Supported decoder types."""

    CTC = "ctc"
    TRANSDUCER = "transducer"  # RNN-T
    ATTENTION = "attention"  # Seq2Seq


@dataclass
class ModelConfig:
    """Base configuration for all ASR models.

    Attributes:
        model_type: The architecture type (SSM, Whisper, Conformer)
        decoder_type: The decoder type (CTC, Transducer, Attention)
        d_model: Model dimension
        n_layers: Number of encoder layers
        vocab_size: Vocabulary size for output
        sample_rate: Audio sample rate in Hz
        n_mels: Number of mel filterbank channels
        use_flash_attention: Whether to use Flash Attention 2
        use_bf16: Whether to use BF16 precision
        dropout: Dropout probability
    """

    model_type: ModelType = ModelType.SSM
    decoder_type: DecoderType = DecoderType.CTC
    d_model: int = 256
    n_layers: int = 18
    vocab_size: int = 256
    sample_rate: int = 16000
    n_mels: int = 80
    use_flash_attention: bool = True
    use_bf16: bool = True
    dropout: float = 0.1

    # Additional fields to be overridden by subclasses
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type.value,
            "decoder_type": self.decoder_type.value,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "use_flash_attention": self.use_flash_attention,
            "use_bf16": self.use_bf16,
            "dropout": self.dropout,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        model_type = ModelType(data.pop("model_type", "ssm"))
        decoder_type = DecoderType(data.pop("decoder_type", "ctc"))
        return cls(model_type=model_type, decoder_type=decoder_type, **data)


class Encoder(nn.Module, ABC):
    """Abstract base class for ASR encoders.

    An encoder transforms audio features into a sequence of hidden representations.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode audio features.

        Args:
            features: Input features of shape (batch, time, n_mels)
            lengths: Optional sequence lengths of shape (batch,)
            streaming: Whether to use causal/streaming mode

        Returns:
            encoded: Encoded features of shape (batch, time', d_model)
            lengths: Output sequence lengths (may differ due to downsampling)
        """
        ...

    @property
    @abstractmethod
    def downsample_factor(self) -> int:
        """Return the time dimension downsampling factor."""
        ...

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after encoding."""
        return input_lengths // self.downsample_factor


class Decoder(nn.Module, ABC):
    """Abstract base class for ASR decoders.

    A decoder transforms encoder outputs into output logits or token sequences.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

    @abstractmethod
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode encoder outputs.

        Args:
            encoder_output: Encoder output of shape (batch, time, d_model)
            encoder_lengths: Encoder output lengths of shape (batch,)
            targets: Optional target sequences for training
            target_lengths: Optional target lengths

        Returns:
            Dictionary containing at minimum:
                - "logits": Output logits of shape (batch, time, vocab_size)
            May also contain:
                - "loss": Computed loss if targets provided
                - "predictions": Decoded token sequences
        """
        ...

    @abstractmethod
    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Greedy decode encoder outputs to token sequences.

        Args:
            encoder_output: Encoder output of shape (batch, time, d_model)
            encoder_lengths: Encoder output lengths

        Returns:
            List of decoded token sequences (one per batch item)
        """
        ...


class CTCDecoder(Decoder):
    """CTC-based decoder with greedy decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.projection = nn.Linear(config.d_model, config.vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        self.blank_id = 0

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute CTC logits and optionally loss."""
        logits = self.projection(encoder_output)  # (B, T, V)
        result: dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None and target_lengths is not None:
            # CTC loss expects (T, B, V)
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

            if encoder_lengths is None:
                encoder_lengths = torch.full(
                    (logits.size(0),),
                    logits.size(1),
                    dtype=torch.long,
                    device=logits.device,
                )

            loss = self.ctc_loss(log_probs, targets, encoder_lengths, target_lengths)
            result["loss"] = loss

        return result

    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Greedy CTC decoding with blank removal and deduplication."""
        logits = self.projection(encoder_output)
        predictions = logits.argmax(dim=-1)  # (B, T)

        decoded: list[list[int]] = []
        for i, pred in enumerate(predictions):
            length = encoder_lengths[i] if encoder_lengths is not None else len(pred)
            tokens: list[int] = []
            prev_token = self.blank_id
            for t in range(length):
                token = pred[t].item()
                if token != self.blank_id and token != prev_token:
                    tokens.append(token)
                prev_token = token
            decoded.append(tokens)

        return decoded


class TransducerDecoder(Decoder):
    """RNN-Transducer (RNNT) decoder.

    Implements the prediction network and joint network for RNN-T.
    """

    def __init__(
        self,
        config: ModelConfig,
        hidden_size: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__(config)

        # Prediction network (label encoder)
        self.embedding = nn.Embedding(config.vocab_size, hidden_size)
        self.prediction_rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=config.dropout if n_layers > 1 else 0.0,
        )

        # Joint network
        self.encoder_proj = nn.Linear(config.d_model, hidden_size)
        self.predictor_proj = nn.Linear(hidden_size, hidden_size)
        self.joint = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size, config.vocab_size),
        )

        self.blank_id = 0
        self.hidden_size = hidden_size

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute joint network output."""
        batch_size, enc_time, _ = encoder_output.shape
        encoder_proj = self.encoder_proj(encoder_output)  # (B, T, H)

        if targets is not None:
            # Training: compute full joint output
            # Prepend blank token to targets
            targets_with_blank = torch.cat(
                [
                    torch.zeros(
                        batch_size, 1, dtype=targets.dtype, device=targets.device
                    ),
                    targets,
                ],
                dim=1,
            )
            pred_embed = self.embedding(targets_with_blank)  # (B, U+1, H)
            pred_out, _ = self.prediction_rnn(pred_embed)  # (B, U+1, H)
            predictor_proj = self.predictor_proj(pred_out)  # (B, U+1, H)

            # Joint: (B, T, 1, H) + (B, 1, U+1, H) -> (B, T, U+1, H)
            joint_input = encoder_proj.unsqueeze(2) + predictor_proj.unsqueeze(1)
            logits = self.joint(joint_input)  # (B, T, U+1, V)

            return {"logits": logits}

        # Inference: return encoder projection for greedy decode
        return {"encoder_proj": encoder_proj}

    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Greedy RNNT decoding."""
        batch_size = encoder_output.size(0)
        encoder_proj = self.encoder_proj(encoder_output)  # (B, T, H)
        device = encoder_output.device

        decoded: list[list[int]] = []

        for b in range(batch_size):
            length = (
                encoder_lengths[b].item()
                if encoder_lengths is not None
                else encoder_output.size(1)
            )
            tokens: list[int] = []

            # Initialize prediction network state
            hidden: tuple[torch.Tensor, torch.Tensor] | None = None
            pred_token = torch.zeros(1, 1, dtype=torch.long, device=device)

            for t in range(int(length)):
                enc_t = encoder_proj[b : b + 1, t : t + 1]  # (1, 1, H)

                # Prediction step
                pred_embed = self.embedding(pred_token)  # (1, 1, H)
                pred_out, hidden = self.prediction_rnn(pred_embed, hidden)
                predictor_proj = self.predictor_proj(pred_out)  # (1, 1, H)

                # Joint
                joint_input = enc_t + predictor_proj  # (1, 1, H)
                logits = self.joint(joint_input)  # (1, 1, V)
                next_token = logits.argmax(dim=-1).item()

                # Non-blank token emitted
                if next_token != self.blank_id:
                    tokens.append(next_token)
                    pred_token = torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    )

            decoded.append(tokens)

        return decoded


class ASRModel(nn.Module, ABC):
    """Abstract base class for complete ASR models.

    An ASR model combines an encoder and decoder for end-to-end speech recognition.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._encoder: Encoder | None = None
        self._decoder: Decoder | None = None

    @property
    def encoder(self) -> Encoder:
        """Get the encoder module."""
        if self._encoder is None:
            raise ValueError("Encoder not initialized")
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        """Get the decoder module."""
        if self._decoder is None:
            raise ValueError("Decoder not initialized")
        return self._decoder

    @abstractmethod
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
            features: Input audio features (batch, time, n_mels)
            feature_lengths: Feature sequence lengths
            targets: Optional target sequences for training
            target_lengths: Optional target lengths
            streaming: Whether to use streaming/causal mode

        Returns:
            Dictionary containing:
                - "logits": Output logits
                - "loss": Training loss (if targets provided)
                - "encoder_output": Intermediate encoder output
        """
        ...

    def transcribe(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> list[list[int]]:
        """Transcribe audio features to token sequences.

        Args:
            features: Input audio features
            feature_lengths: Feature sequence lengths
            streaming: Whether to use streaming mode

        Returns:
            List of decoded token sequences
        """
        with torch.no_grad():
            encoder_output, encoder_lengths = self.encoder(
                features, feature_lengths, streaming=streaming
            )
            return self.decoder.decode(encoder_output, encoder_lengths)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "ASRModel":
        """Create model from configuration."""
        ...

    @classmethod
    def from_pretrained(cls, path: str) -> "ASRModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = ModelConfig.from_dict(checkpoint["config"])
        model = cls.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "config": self.config.to_dict(),
                "model_state_dict": self.state_dict(),
            },
            path,
        )
