"""Audio augmentation for ASR training."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpecAugmentConfig:
    """Configuration for SpecAugment.

    Attributes:
        freq_mask_param: Maximum frequency mask width
        time_mask_param: Maximum time mask width
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        time_mask_ratio: Maximum ratio of time steps to mask
    """

    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2
    time_mask_ratio: float = 0.05


class SpecAugment(nn.Module):
    """SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies frequency and time masking to spectrograms during training.

    Reference: https://arxiv.org/abs/1904.08779
    """

    def __init__(self, config: SpecAugmentConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = SpecAugmentConfig()
        self.config = config

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to features.

        Args:
            features: Input features of shape (batch, time, n_features)

        Returns:
            Augmented features of same shape
        """
        if not self.training:
            return features

        batch_size, time_steps, n_features = features.shape
        augmented = features.clone()

        for b in range(batch_size):
            # Frequency masking
            for _ in range(self.config.num_freq_masks):
                f = torch.randint(0, self.config.freq_mask_param + 1, (1,)).item()
                f0 = torch.randint(0, max(1, n_features - f), (1,)).item()
                augmented[b, :, f0 : f0 + f] = 0.0

            # Time masking
            max_time_mask = min(
                self.config.time_mask_param,
                int(time_steps * self.config.time_mask_ratio),
            )
            for _ in range(self.config.num_time_masks):
                t = torch.randint(0, max(1, max_time_mask + 1), (1,)).item()
                t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
                augmented[b, t0 : t0 + t, :] = 0.0

        return augmented


class SpeedPerturbation(nn.Module):
    """Speed perturbation for audio augmentation.

    Randomly changes audio speed while preserving pitch.
    """

    def __init__(
        self,
        speeds: list[float] | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.speeds = speeds or [0.9, 1.0, 1.1]
        self.p = p

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply speed perturbation.

        Args:
            waveform: Audio waveform of shape (samples,) or (batch, samples)
            sample_rate: Sample rate of the audio

        Returns:
            Speed-perturbed waveform
        """
        if not self.training or torch.rand(1).item() > self.p:
            return waveform

        speed = self.speeds[torch.randint(len(self.speeds), (1,)).item()]
        if speed == 1.0:
            return waveform

        # Simple resampling-based speed change
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        orig_len = waveform.size(-1)
        new_len = int(orig_len / speed)

        # Resample
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(1),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        return waveform.squeeze(0) if waveform.size(0) == 1 else waveform


class NoiseInjection(nn.Module):
    """Add random noise to audio."""

    def __init__(
        self,
        min_snr_db: float = 10.0,
        max_snr_db: float = 50.0,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add noise to waveform.

        Args:
            waveform: Audio waveform

        Returns:
            Noisy waveform
        """
        if not self.training or torch.rand(1).item() > self.p:
            return waveform

        # Random SNR
        snr_db = (
            torch.rand(1).item() * (self.max_snr_db - self.min_snr_db)
            + self.min_snr_db
        )

        # Compute signal power
        signal_power = waveform.pow(2).mean()

        # Generate noise with appropriate power
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * noise_power.sqrt()

        return waveform + noise
