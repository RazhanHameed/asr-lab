"""SSM Encoder using Mamba2 blocks with optional Flash Attention."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from asr_lab.models.base import Encoder
from asr_lab.models.ssm.config import SSMConfig

# Try to import optimized implementations
try:
    from mamba_ssm import Mamba2

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    from flash_attn.modules.mha import MHA as FlashMHA

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""

    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = d_model * expand
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class FallbackMambaBlock(nn.Module):
    """Pure PyTorch Mamba block for CPU or when mamba-ssm unavailable."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv1D
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A as log for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Conv1D
        x_conv = rearrange(x_proj, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)

        # SSM parameters
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Discretize A
        A = -torch.exp(self.A_log)

        # Simplified SSM scan (not fully optimized)
        y = self._ssm_scan(x_conv, dt, A, B, C)

        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified SSM scan."""
        batch, seq_len, d_inner = x.shape

        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h = A * h + B * x
            dA = torch.exp(dt[:, t] * A.unsqueeze(0))  # (B, D, N)
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)  # (B, D, N)
            h = h * dA.unsqueeze(1) + dB * x[:, t].unsqueeze(-1)

            # Output: y = C * h + D * x
            y = (h * C[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class OptimizedMambaBlock(nn.Module):
    """Wrapper around official Mamba2 implementation."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Use FallbackMambaBlock.")

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class FlashMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Flash Attention 2 support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE

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
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.use_flash:
            return self.attn(x, causal=is_causal)[0]

        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
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


class SSMBlock(nn.Module):
    """Single SSM block with normalization and residual connection."""

    def __init__(
        self,
        config: SSMConfig,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        # Mamba block
        self.norm1 = RMSNorm(config.d_model)
        if MAMBA_AVAILABLE and torch.cuda.is_available():
            self.mamba = OptimizedMambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                headdim=config.headdim,
                chunk_size=config.chunk_size,
            )
        else:
            self.mamba = FallbackMambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            self.norm2 = RMSNorm(config.d_model)
            self.attention = FlashMultiHeadAttention(
                d_model=config.d_model,
                n_heads=config.d_model // config.headdim,
                dropout=config.dropout,
                use_flash=config.use_flash_attention,
            )

        # FFN
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, expand=4, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Mamba
        x = x + self.mamba(self.norm1(x))

        # Attention
        if self.use_attention:
            x = x + self.attention(self.norm2(x), mask=mask, is_causal=is_causal)

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


class ConvFrontend(nn.Module):
    """Convolutional frontend for audio feature processing."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim

        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], output_dim) if channels[-1] != output_dim else nn.Identity()

        # Calculate total downsampling factor
        self._downsample = 1
        for s in strides:
            self._downsample *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # (B, D, T') -> (B, T', D)
        x = x.transpose(1, 2)
        return self.proj(x)

    @property
    def downsample_factor(self) -> int:
        return self._downsample


class SSMEncoder(Encoder):
    """SSM-based encoder using Mamba2 blocks."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__(config)
        self.ssm_config = config

        # Frontend
        self.frontend = ConvFrontend(
            input_dim=config.n_mels,
            output_dim=config.d_model,
            channels=config.conv_channels,
            kernel_sizes=config.conv_kernel_sizes,
            strides=config.conv_strides,
        )

        # SSM blocks
        self.blocks = nn.ModuleList()
        for i in range(config.n_layers):
            use_attention = (i + 1) % config.attention_every_n_layers == 0
            self.blocks.append(SSMBlock(config, use_attention=use_attention))

        self.final_norm = RMSNorm(config.d_model)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Frontend
        x = self.frontend(features)

        # Update lengths for downsampling
        if lengths is not None:
            lengths = self.get_output_lengths(lengths)

        # Create mask if needed
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Process through blocks
        is_causal = streaming or not self.ssm_config.bidirectional
        for block in self.blocks:
            x = block(x, mask=mask, is_causal=is_causal)

        x = self.final_norm(x)

        return x, lengths

    @property
    def downsample_factor(self) -> int:
        return self.frontend.downsample_factor
