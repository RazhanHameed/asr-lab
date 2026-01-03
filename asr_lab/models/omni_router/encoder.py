"""Omni-Router MoE Encoder components.

Implements the core components of Apple's Omni-Router MoE architecture:
- SwitchGate: Router that assigns tokens to experts with optional noise
- MultiHeadAttention: Self-attention with causal/non-causal masking
- TransformerEncoderBlock: Block with optional MoE FFN
- TransformerEncoder: Stacked blocks with shared router option
- OmniRouterEncoder: Full encoder with frame stacking and CAPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import rearrange, repeat

from asr_lab.models.base import Encoder
from asr_lab.models.omni_router.config import (
    LayerNormPosition,
    MaskMode,
    OmniRouterConfig,
)
from asr_lab.models.omni_router.posemb import CAPE1d


class SwitchGate(nn.Module):
    """Router that assigns tokens to experts using Switch Transformer style gating.

    Each token is routed to its top-1 expert based on learned router weights.
    Optional noise injection during training improves load balancing.

    Args:
        in_dim: Input dimension.
        num_experts: Number of experts to route to.
        noise_scale: Scale of noise to add during training (0 to disable).
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.noise_scale = noise_scale
        self.w_gate = nn.Linear(in_dim, num_experts)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to logits during training."""
        return x + torch.randn_like(x) * self.noise_scale  # noqa: S311

    def _softmax_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of routing distribution."""
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing probabilities for input tokens.

        Args:
            x: Input tensor of shape (batch * time, d_model) or (batch, time, d_model).

        Returns:
            gate_probs: Routing probabilities of shape (..., num_experts).
            gate_entropy: Scalar entropy of routing distribution.
        """
        logits = self.w_gate(x)
        if self.training and self.noise_scale > 0:
            logits = self._add_noise(logits)
        gate_probs = nnf.softmax(logits, dim=-1)
        gate_entropy = self._softmax_entropy(gate_probs)
        return gate_probs, gate_entropy


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with support for causal and non-causal masking.

    Uses scaled dot-product attention with efficient fused implementation.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in linear projections.
        mask_mode: CAUSAL for autoregressive, NON_CAUSAL for bidirectional.
        attn_masking_val: Value used for masked positions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        mask_mode: MaskMode = MaskMode.NON_CAUSAL,
        attn_masking_val: float = -10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mask_mode = mask_mode
        self.attn_masking_val = attn_masking_val

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.w_out = nn.Linear(d_model, d_model, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with uniform distribution."""
        scale_qo = (1.0 / self.d_model) ** 0.5
        nn.init.uniform_(self.w_q.weight, -scale_qo, scale_qo)
        nn.init.uniform_(self.w_kv.weight, -scale_qo, scale_qo)
        nn.init.uniform_(self.w_out.weight, -scale_qo, scale_qo)

    def _get_padding_mask(
        self, q: torch.Tensor, k: torch.Tensor, kv_length: torch.Tensor
    ) -> torch.Tensor:
        """Create padding mask for attention.

        Args:
            q: Query tensor of shape (batch, heads, tq, head_dim).
            k: Key tensor of shape (batch, heads, tk, head_dim).
            kv_length: Sequence lengths of shape (batch,).

        Returns:
            Attention mask of shape (batch, heads, tq, tk).
        """
        b, h, tq, _ = q.shape
        tk = k.shape[2]

        indices = repeat(
            torch.arange(tk, device=k.device),
            "tk -> b h tq tk",
            b=b,
            h=h,
            tq=tq,
        )
        padding_mask = indices < rearrange(kv_length, "b -> b 1 1 1")
        return (~padding_mask).float() * self.attn_masking_val

    def forward(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor | None = None,
        kv_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: Query tensor of shape (batch, time, d_model).
            x_src: Key/value source tensor. If None, uses x (self-attention).
            kv_length: Sequence lengths for padding mask.

        Returns:
            Output tensor of shape (batch, time, d_model).
        """
        if x_src is None:
            x_src = x

        is_causal = self.mask_mode == MaskMode.CAUSAL

        q = rearrange(
            self.w_q(x),
            "b t (h hc) -> b h t hc",
            h=self.n_heads,
            hc=self.head_dim,
        )
        k, v = rearrange(
            self.w_kv(x_src),
            "b t (kv h hc) -> kv b h t hc",
            kv=2,
            h=self.n_heads,
            hc=self.head_dim,
        )

        attn_mask = None
        if not is_causal and kv_length is not None:
            attn_mask = self._get_padding_mask(q, k, kv_length)

        scale_factor = q.size(-1) ** (-0.25)
        result = nnf.scaled_dot_product_attention(
            q * scale_factor,
            k * scale_factor,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=1.0,
        )

        result = rearrange(result, "b h t hc -> b t (h hc)")
        output: torch.Tensor = self.w_out(result)
        return output


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network.

    Routes each token to its top-1 expert, scales output by routing probability.

    Args:
        d_model: Model dimension.
        mlp_dim: Hidden dimension of each expert FFN.
        n_experts: Number of expert networks.
        router: Optional shared router. If None, creates a new one.
        noise_scale: Noise scale for router (used if creating new router).
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        n_experts: int,
        router: SwitchGate | None = None,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp_dim = mlp_dim
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, d_model),
            )
            for _ in range(n_experts)
        ])

        if router is not None:
            self.router = router
        else:
            self.router = SwitchGate(d_model, n_experts, noise_scale=noise_scale)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize expert weights."""
        scale_in = (1.0 / self.d_model) ** 0.5
        scale_out = (1.0 / self.mlp_dim) ** 0.5

        for expert in self.experts:
            if isinstance(expert, nn.Sequential) and len(expert) >= 3:
                linear1 = expert[0]
                linear2 = expert[2]
                if isinstance(linear1, nn.Linear) and isinstance(linear2, nn.Linear):
                    nn.init.uniform_(linear1.weight, -scale_in, scale_in)
                    nn.init.uniform_(linear2.weight, -scale_out, scale_out)
                    linear1.bias.data.zero_()
                    linear2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MoE FFN to input.

        Args:
            x: Input tensor of shape (batch, time, d_model).

        Returns:
            Output tensor of shape (batch, time, d_model).
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = rearrange(x, "b t d -> (b t) d")

        route_probs, _ = self.router(x_flat)
        route_prob_max, route_indices = torch.max(route_probs, dim=-1)

        output = torch.zeros_like(x_flat)

        for i in range(self.n_experts):
            mask = route_indices == i
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = self.experts[i](expert_input)
                output[mask] = expert_output * route_prob_max[mask].unsqueeze(-1)

        return rearrange(output, "(b t) d -> b t d", b=batch_size)


class DenseFFN(nn.Module):
    """Standard dense Feed-Forward Network.

    Args:
        d_model: Model dimension.
        mlp_dim: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, mlp_dim, bias=True)
        self.w2 = nn.Linear(mlp_dim, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        self._init_weights(d_model, mlp_dim)

    def _init_weights(self, d_model: int, mlp_dim: int) -> None:
        """Initialize weights."""
        scale_in = (1.0 / d_model) ** 0.5
        scale_out = (1.0 / mlp_dim) ** 0.5
        nn.init.uniform_(self.w1.weight, -scale_in, scale_in)
        nn.init.uniform_(self.w2.weight, -scale_out, scale_out)
        self.w1.bias.data.zero_()
        self.w2.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN to input."""
        x = self.w1(x)
        x = nnf.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with optional MoE FFN.

    Supports both pre-norm and post-norm configurations.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        mlp_dim: Hidden dimension of FFN.
        dropout: Dropout probability.
        layer_dropout: Layer-level dropout probability (not used in forward).
        ln_position: PRE or POST layer norm position.
        mask_mode: CAUSAL or NON_CAUSAL attention masking.
        bias: Whether to use bias in linear layers.
        ln_epsilon: Epsilon for layer normalization.
        n_experts: Number of experts (0 for dense FFN).
        router: Optional shared router for MoE.
        moe_jitter_eps: Noise scale for router.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        layer_dropout: float = 0.0,
        ln_position: LayerNormPosition = LayerNormPosition.PRE,
        mask_mode: MaskMode = MaskMode.NON_CAUSAL,
        bias: bool = False,
        ln_epsilon: float = 1e-5,
        n_experts: int = 0,
        router: SwitchGate | None = None,
        moe_jitter_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.ln_position = ln_position
        self.n_experts = n_experts

        self.ln1 = nn.LayerNorm(d_model, eps=ln_epsilon)
        self.ln2 = nn.LayerNorm(d_model, eps=ln_epsilon)

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            mask_mode=mask_mode,
        )

        if n_experts > 0:
            self.ffn: nn.Module = MoEFFN(
                d_model=d_model,
                mlp_dim=mlp_dim,
                n_experts=n_experts,
                router=router,
                noise_scale=moe_jitter_eps,
            )
        else:
            self.ffn = DenseFFN(
                d_model=d_model,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor of shape (batch, time, d_model).
            x_length: Sequence lengths for attention masking.

        Returns:
            Output tensor of shape (batch, time, d_model).
        """
        if self.ln_position == LayerNormPosition.POST:
            x = self.ln1(self.self_attention(x, x, x_length) + x)
            x = self.ln2(self.ffn(x) + x)
        else:
            y = self.ln1(x)
            x = self.self_attention(y, y, x_length) + x
            x = self.ffn(self.ln2(x)) + x

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with stacked blocks and optional shared routing.

    The n_experts string format "4x2-4x4-4x8" specifies:
    - First 4 layers: 2 experts each
    - Next 4 layers: 4 experts each
    - Next 4 layers: 8 experts each

    When share_router=True, all layers in a group share the same SwitchGate.

    Args:
        config: OmniRouterConfig with all encoder parameters.
    """

    def __init__(self, config: OmniRouterConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_blocks = config.n_layers
        self.ln_position = config.ln_position
        self.content_scale = config.content_scale

        self.cape: CAPE1d | None = None
        if config.cape_config is not None:
            self.cape = CAPE1d(d_model=config.d_model, config=config.cape_config)

        expert_configs, router_indices = self._parse_expert_config(
            config.n_experts, config.n_layers
        )

        routers: list[SwitchGate | None] = self._create_routers(
            config, expert_configs, config.share_router
        )

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                layer_dropout=config.layer_dropout,
                ln_position=config.ln_position,
                mask_mode=config.mask_mode,
                bias=config.bias,
                ln_epsilon=config.ln_epsilon,
                n_experts=expert_configs[i],
                router=routers[router_indices[i]] if expert_configs[i] > 0 else None,
                moe_jitter_eps=config.moe_jitter_eps,
            )
            for i in range(config.n_layers)
        ])

        if config.ln_position == LayerNormPosition.PRE:
            self.final_ln: nn.Module = nn.LayerNorm(config.d_model, eps=config.ln_epsilon)
        else:
            self.final_ln = nn.Identity()

    def _parse_expert_config(
        self, n_experts_str: str, n_layers: int
    ) -> tuple[list[int], list[int]]:
        """Parse expert configuration string.

        Args:
            n_experts_str: String like "4x2-4x4-4x8" or "0".
            n_layers: Total number of layers.

        Returns:
            expert_configs: List of expert counts per layer.
            router_indices: List of router group indices per layer.
        """
        if n_experts_str == "0" or str(n_experts_str) == "0":
            return [0] * n_layers, list(range(n_layers))

        expert_configs: list[int] = []
        router_indices: list[int] = []

        groups = n_experts_str.split("-")
        for group_idx, group in enumerate(groups):
            parts = group.split("x")
            num_layers = int(parts[0])
            num_experts = int(parts[1])

            for _ in range(num_layers):
                expert_configs.append(num_experts)
                router_indices.append(group_idx)

        if len(expert_configs) != n_layers:
            raise ValueError(
                f"Expert config '{n_experts_str}' specifies {len(expert_configs)} "
                f"layers but model has {n_layers} layers"
            )

        return expert_configs, router_indices

    def _create_routers(
        self,
        config: OmniRouterConfig,
        expert_configs: list[int],
        share_router: bool,
    ) -> list[SwitchGate | None]:
        """Create router instances for each expert group.

        Args:
            config: Model configuration.
            expert_configs: List of expert counts per layer.
            share_router: Whether to share routers within groups.

        Returns:
            List of routers (one per group if shared, else None).
        """
        if not share_router:
            return [None] * len(expert_configs)

        routers: list[SwitchGate | None] = []

        groups = config.n_experts.split("-") if config.n_experts != "0" else []

        for group in groups:
            parts = group.split("x")
            num_experts = int(parts[1])

            if num_experts > 0:
                router = SwitchGate(
                    config.d_model,
                    num_experts,
                    noise_scale=config.moe_jitter_eps,
                )
            else:
                router = None
            routers.append(router)

        if not routers:
            routers = [None]

        return routers

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        return self.d_model

    def forward(
        self, x: torch.Tensor, x_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transformer encoder.

        Args:
            x: Input tensor of shape (batch, time, d_model).
            x_length: Sequence lengths of shape (batch,).

        Returns:
            Output tensor and lengths.
        """
        if self.cape is not None:
            pe = self.cape(x, x_lengths=x_length)
            x = self.content_scale * x + pe

        for block in self.blocks:
            x = block(x, x_length)

        x = self.final_ln(x)

        return x, x_length.int()


class OmniRouterEncoder(Encoder):
    """Full Omni-Router encoder with frame stacking and transformer.

    Combines:
    - Frame stacking for temporal downsampling
    - Linear projection to model dimension
    - Transformer encoder with optional MoE and shared routing
    - CAPE positional embeddings

    Args:
        config: OmniRouterConfig with all encoder parameters.
    """

    def __init__(self, config: OmniRouterConfig) -> None:
        super().__init__(config)
        self.omni_config = config
        self.stacking = config.stacking
        self._downsample = config.stacking

        self.input_proj = nn.Linear(
            config.n_mels * config.stacking,
            config.d_model,
            bias=config.bias,
        )

        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.dropout)

    def _stack_frames(
        self, x: torch.Tensor, x_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack consecutive frames for temporal downsampling.

        Args:
            x: Input tensor of shape (batch, time, n_mels).
            x_length: Sequence lengths.

        Returns:
            Stacked tensor and updated lengths.
        """
        if self.stacking <= 1:
            return x, x_length

        batch, time, feat = x.shape
        to_pad = (self.stacking - time % self.stacking) % self.stacking

        if to_pad > 0:
            padding = torch.zeros(
                batch, to_pad, feat,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, padding], dim=1)

        x = rearrange(
            x,
            "b (t s) c -> b t (s c)",
            s=self.stacking,
        )

        x_length = x_length // self.stacking + int(to_pad > 0)

        return x, x_length

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode audio features.

        Args:
            features: Input mel features of shape (batch, time, n_mels).
            lengths: Optional sequence lengths of shape (batch,).
            streaming: Whether to use causal/streaming mode (ignored, uses config).

        Returns:
            Encoded features and output lengths.
        """
        if lengths is None:
            lengths = torch.full(
                (features.size(0),),
                features.size(1),
                dtype=torch.long,
                device=features.device,
            )

        x, x_length = self._stack_frames(features, lengths)
        x = self.input_proj(x)
        x = self.dropout(x)
        x, x_length = self.encoder(x, x_length)

        return x, x_length

    @property
    def downsample_factor(self) -> int:
        """Return the time dimension downsampling factor."""
        return self._downsample
