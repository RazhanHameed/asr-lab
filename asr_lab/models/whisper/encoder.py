"""Whisper-style Transformer encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_lab.models.base import Encoder
from asr_lab.models.whisper.config import WhisperConfig

try:
    from flash_attn.modules.mha import MHA as FlashMHA

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in original Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional Flash Attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE and not is_cross_attention
        self.is_cross_attention = is_cross_attention

        if self.use_flash:
            self.attn = FlashMHA(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                use_flash_attn=True,
            )
        else:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.use_flash:
            return self.attn(x, causal=is_causal)[0]

        batch, seq_len, _ = x.shape

        # Cross-attention: K, V from encoder
        if self.is_cross_attention and encoder_output is not None:
            q = self.q_proj(x)
            k = self.k_proj(encoder_output)
            v = self.v_proj(encoder_output)
            kv_len = encoder_output.size(1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            kv_len = seq_len

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_flash)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class WhisperEncoder(Encoder):
    """Whisper-style convolutional + Transformer encoder."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__(config)
        self.whisper_config = config

        # Convolutional frontend (2 conv layers with stride 2 each = 4x downsample)
        self.conv1 = nn.Conv1d(
            config.n_mels,
            config.conv_channels,
            kernel_size=config.conv_kernel_size,
            stride=1,
            padding=config.conv_kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            config.conv_channels,
            config.d_model,
            kernel_size=config.conv_kernel_size,
            stride=2,
            padding=config.conv_kernel_size // 2,
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_source_positions
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model,
                config.n_heads,
                config.encoder_ffn_dim,
                config.dropout,
                config.use_flash_attention,
            )
            for _ in range(config.n_encoder_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self._downsample = 2  # Due to conv2 stride

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # features: (B, T, n_mels)
        x = features.transpose(1, 2)  # (B, n_mels, T)

        # Conv frontend
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = x.transpose(1, 2)  # (B, T', d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Update lengths
        if lengths is not None:
            lengths = self.get_output_lengths(lengths)

        # Create mask
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)

        return x, lengths

    @property
    def downsample_factor(self) -> int:
        return self._downsample
