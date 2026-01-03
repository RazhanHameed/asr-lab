"""Audio feature extraction for ASR."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


@dataclass
class FeatureConfig:
    """Configuration for audio feature extraction.

    Attributes:
        sample_rate: Target sample rate in Hz
        n_mels: Number of mel filterbank channels
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        f_min: Minimum frequency for mel filterbank
        f_max: Maximum frequency for mel filterbank
        normalize: Whether to normalize features
        log_mel: Whether to use log mel spectrogram
    """

    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: float = 8000.0
    normalize: bool = True
    log_mel: bool = True
    n_mfcc: int = 40


class FeatureExtractor(nn.Module, ABC):
    """Abstract base class for audio feature extractors."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.

        Args:
            waveform: Audio waveform of shape (batch, samples) or (samples,)

        Returns:
            Features of shape (batch, time, n_features)
        """
        ...

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after feature extraction."""
        return input_lengths // self.config.hop_length + 1


class MelSpectrogramExtractor(FeatureExtractor):
    """Log mel spectrogram feature extractor.

    This is the standard feature extraction used by most modern ASR models
    including Whisper and Conformer.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        if config is None:
            config = FeatureConfig()
        super().__init__(config)

        self.mel_spec = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

        # Global normalization stats (computed from large dataset)
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log mel spectrogram features.

        Args:
            waveform: Audio waveform of shape (batch, samples) or (samples,)

        Returns:
            Log mel spectrogram of shape (batch, time, n_mels)
        """
        # Handle 1D input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram: (batch, n_mels, time)
        mel = self.mel_spec(waveform)

        # Log scale
        if self.config.log_mel:
            mel = torch.clamp(mel, min=1e-10).log10()
            mel = torch.maximum(mel, mel.max() - 8.0)
            mel = (mel + 4.0) / 4.0

        # Normalize
        if self.config.normalize:
            mel = (mel - self.mean) / (self.std + 1e-6)

        # Transpose to (batch, time, n_mels)
        return mel.transpose(1, 2)

    @classmethod
    def whisper_style(cls) -> "MelSpectrogramExtractor":
        """Create Whisper-style mel spectrogram extractor."""
        config = FeatureConfig(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
        )
        return cls(config)

    @classmethod
    def conformer_style(cls) -> "MelSpectrogramExtractor":
        """Create Conformer-style mel spectrogram extractor."""
        config = FeatureConfig(
            sample_rate=16000,
            n_mels=80,
            n_fft=512,
            hop_length=160,
            win_length=400,
            f_min=0.0,
            f_max=8000.0,
        )
        return cls(config)

    @classmethod
    def ssm_style(cls) -> "MelSpectrogramExtractor":
        """Create SSM-ASR style mel spectrogram extractor.

        Matches ABR asr-19m-v2 configuration:
        - 80 mel filterbank bins
        - 25ms window size (400 samples at 16kHz)
        - 10ms window stride (160 samples at 16kHz)
        - 0-8000 Hz frequency range
        """
        config = FeatureConfig(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,  # 25ms at 16kHz
            hop_length=160,  # 10ms at 16kHz
            win_length=400,  # 25ms at 16kHz
            f_min=0.0,
            f_max=8000.0,
            normalize=True,
            log_mel=True,
        )
        return cls(config)


class MFCCExtractor(FeatureExtractor):
    """MFCC feature extractor.

    Traditional MFCC features used by some ASR systems.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        if config is None:
            config = FeatureConfig()
        super().__init__(config)

        self.mfcc = T.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            melkwargs={
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "win_length": config.win_length,
                "n_mels": config.n_mels,
                "f_min": config.f_min,
                "f_max": config.f_max,
            },
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features.

        Args:
            waveform: Audio waveform of shape (batch, samples) or (samples,)

        Returns:
            MFCC features of shape (batch, time, n_mfcc)
        """
        # Handle 1D input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute MFCC: (batch, n_mfcc, time)
        mfcc = self.mfcc(waveform)

        # Normalize
        if self.config.normalize:
            mfcc = (mfcc - mfcc.mean(dim=-1, keepdim=True)) / (
                mfcc.std(dim=-1, keepdim=True) + 1e-6
            )

        # Transpose to (batch, time, n_mfcc)
        return mfcc.transpose(1, 2)


def load_audio(
    path: str,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Load and resample audio file.

    Args:
        path: Path to audio file
        target_sr: Target sample rate

    Returns:
        Waveform tensor of shape (samples,)
    """
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)
