"""CAPE (Continuous Augmented Positional Embeddings) for Omni-Router.

CAPE extends standard sinusoidal positional embeddings with training-time
augmentations including position shifting and scaling for improved robustness.
"""

import math
from typing import cast

import torch
import torch.nn as nn
from einops import rearrange, repeat

from asr_lab.models.omni_router.config import CAPEConfig


class CAPE1d(nn.Module):
    """Continuous Augmented Positional Embeddings for 1D sequences.

    CAPE applies sinusoidal positional embeddings with optional training-time
    augmentations including global/local shifts and scaling.

    Args:
        d_model: Model dimension for positional embeddings.
        config: CAPE configuration with augmentation parameters.
    """

    def __init__(self, d_model: int, config: CAPEConfig) -> None:
        super().__init__()

        if config.max_global_shift < 0:
            raise ValueError(
                f"max_global_shift must be >= 0, got {config.max_global_shift}"
            )
        if config.max_local_shift < 0:
            raise ValueError(
                f"max_local_shift must be >= 0, got {config.max_local_shift}"
            )
        if config.max_global_scaling < 1:
            raise ValueError(
                f"max_global_scaling must be >= 1, got {config.max_global_scaling}"
            )

        self.max_global_shift = config.max_global_shift
        self.max_local_shift = config.max_local_shift
        self.max_global_scaling = config.max_global_scaling
        self.normalize = config.normalize
        self.freq_scale = config.freq_scale
        self.positions_delta = config.positions_delta

        freq = self.freq_scale * torch.exp(
            -2.0 * torch.floor(torch.arange(d_model) / 2) * (math.log(1e4) / d_model)
        )
        self.register_buffer("freq", freq)

        sin2cos_phase_shift = torch.pi / 2.0
        cos_shifts = sin2cos_phase_shift * (torch.arange(d_model) % 2)
        self.register_buffer("cos_shifts", cos_shifts)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute positional embeddings for input sequence.

        Args:
            x: Input tensor of shape (batch, time, d_model).
            x_lengths: Optional sequence lengths of shape (batch,).

        Returns:
            Positional embeddings of shape (batch, time, d_model).
        """
        return self._compute_pos_emb(x, x_lengths, self.positions_delta)

    def get_pos_emb(
        self, batch_size: int, start_pos: int, n_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Get positional embeddings for streaming inference.

        Args:
            batch_size: Batch size.
            start_pos: Starting position index.
            n_tokens: Number of tokens.
            device: Target device.

        Returns:
            Positional embeddings of shape (batch, time, d_model).
        """
        if self.normalize:
            raise ValueError("Cannot use get_pos_emb with normalize=True")

        positions = repeat(
            start_pos + torch.arange(n_tokens, device=device),
            "t -> b t 1",
            b=batch_size,
        )

        freq = cast(torch.Tensor, self.freq)
        cos_shifts = cast(torch.Tensor, self.cos_shifts)

        product = positions * freq
        pos_emb = torch.sin(product + cos_shifts)
        pos_emb = torch.nan_to_num(pos_emb, nan=0)

        return pos_emb

    def _compute_pos_emb(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor | None = None,
        positions_delta: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal method to compute positional embeddings with augmentation."""
        batch_size, n_tokens, _ = x.shape

        positions = repeat(
            torch.arange(n_tokens, device=x.device, dtype=x.dtype),
            "t -> b t",
            b=batch_size,
        )

        if x_lengths is not None:
            padding_mask = positions >= x_lengths[:, None].to(x.dtype)
            positions = positions.masked_fill(padding_mask, float("nan"))

        if positions_delta is None:
            positions_delta = 1.0
        else:
            if torch.is_tensor(positions_delta) and positions_delta.dim() == 1:
                positions_delta = rearrange(positions_delta, "b -> b 1")
            positions = positions * positions_delta

        if self.normalize:
            positions = positions - torch.nanmean(positions, dim=1, keepdim=True)

        positions = self._augment_positions(positions, positions_delta)

        freq = cast(torch.Tensor, self.freq)
        cos_shifts = cast(torch.Tensor, self.cos_shifts)

        product = rearrange(positions, "b t -> b t 1") * freq
        pos_emb = torch.sin(product + cos_shifts)
        pos_emb = torch.nan_to_num(pos_emb, nan=0)

        return pos_emb

    def _augment_positions(
        self,
        positions: torch.Tensor,
        positions_delta: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply position augmentations during training."""
        if not self.training:
            return positions

        batch_size, n_tokens = positions.shape
        device = positions.device
        dtype = positions.dtype

        delta: float | torch.Tensor = 0.0
        if self.max_global_shift > 0:
            delta = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
                -self.max_global_shift, self.max_global_shift
            )

        delta_local: float | torch.Tensor = 0.0
        if self.max_local_shift > 0:
            delta_local = torch.empty(
                batch_size, n_tokens, device=device, dtype=dtype
            ).uniform_(-self.max_local_shift, self.max_local_shift)
            if positions_delta is not None:
                if torch.is_tensor(positions_delta) and positions_delta.dim() == 1:
                    positions_delta = rearrange(positions_delta, "b -> b 1")
                delta_local = delta_local * positions_delta

        log_lambdas: torch.Tensor
        if self.max_global_scaling > 1.0:
            log_lambdas = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
                -math.log(self.max_global_scaling),
                math.log(self.max_global_scaling),
            )
        else:
            log_lambdas = torch.zeros(1, device=device, dtype=dtype)

        positions = (positions + delta + delta_local) * torch.exp(log_lambdas)

        return positions
