"""Configuration for Omni-Router MoE ASR models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from asr_lab.models.base import DecoderType, ModelConfig, ModelType


class LayerNormPosition(Enum):
    """Position of layer normalization in transformer blocks."""

    PRE = "pre"
    POST = "post"


class MaskMode(Enum):
    """Attention masking mode."""

    CAUSAL = "causal"
    NON_CAUSAL = "non_causal"


@dataclass
class CAPEConfig:
    """Configuration for CAPE (Continuous Augmented Positional Embeddings).

    Attributes:
        max_global_shift: Maximum global position shift during training augmentation.
        max_local_shift: Maximum local position shift during training augmentation.
        max_global_scaling: Maximum global scaling factor during training augmentation.
        normalize: Whether to normalize positions by centering.
        freq_scale: Frequency scaling factor for sinusoidal embeddings.
        positions_delta: Delta between consecutive positions (for variable frame rates).
    """

    max_global_shift: float = 0.0
    max_local_shift: float = 0.0
    max_global_scaling: float = 1.0
    normalize: bool = False
    freq_scale: float = 1.0
    positions_delta: float | None = None


@dataclass
class OmniRouterConfig(ModelConfig):
    """Configuration for Omni-Router MoE ASR model.

    The key innovation is the `n_experts` format and `share_router` flag:
    - n_experts="4x2-4x4-4x8" means first 4 layers have 2 experts, next 4 have 4, next 4 have 8
    - share_router=True shares the same router across all layers in each group

    Attributes:
        n_heads: Number of attention heads.
        mlp_dim: Hidden dimension of feed-forward network.
        stacking: Frame stacking factor for input downsampling.
        content_scale: Scaling factor for content before adding positional embeddings.
        layer_dropout: Layer-level dropout probability.
        ln_position: Position of layer normalization (PRE or POST).
        mask_mode: Attention masking mode (CAUSAL or NON_CAUSAL).
        ln_epsilon: Epsilon for layer normalization.
        bias: Whether to use bias in linear layers.
        n_experts: Expert configuration string like "4x2-4x4-4x8".
        share_router: Whether to share router across layers in each expert group.
        load_balance_loss_weight: Weight for load balancing loss.
        moe_jitter_eps: Noise scale for router during training.
        cape_config: Configuration for CAPE positional embeddings.
    """

    n_heads: int = 8
    mlp_dim: int = 2048
    stacking: int = 4
    content_scale: float = 1.0
    layer_dropout: float = 0.0
    ln_position: LayerNormPosition = LayerNormPosition.PRE
    mask_mode: MaskMode = MaskMode.NON_CAUSAL
    ln_epsilon: float = 1e-5
    bias: bool = False

    n_experts: str = "0"
    share_router: bool = True
    load_balance_loss_weight: float = 0.0
    moe_jitter_eps: float = 0.0

    cape_config: CAPEConfig = field(default_factory=CAPEConfig)

    def __post_init__(self) -> None:
        self.model_type = ModelType.SSM  # Using SSM as placeholder; could add OMNI_ROUTER
        self.extra = {
            "n_heads": self.n_heads,
            "mlp_dim": self.mlp_dim,
            "stacking": self.stacking,
            "content_scale": self.content_scale,
            "layer_dropout": self.layer_dropout,
            "ln_position": self.ln_position.value,
            "mask_mode": self.mask_mode.value,
            "ln_epsilon": self.ln_epsilon,
            "bias": self.bias,
            "n_experts": self.n_experts,
            "share_router": self.share_router,
            "load_balance_loss_weight": self.load_balance_loss_weight,
            "moe_jitter_eps": self.moe_jitter_eps,
            "cape_config": {
                "max_global_shift": self.cape_config.max_global_shift,
                "max_local_shift": self.cape_config.max_local_shift,
                "max_global_scaling": self.cape_config.max_global_scaling,
                "normalize": self.cape_config.normalize,
                "freq_scale": self.cape_config.freq_scale,
                "positions_delta": self.cape_config.positions_delta,
            },
        }

    @classmethod
    def small(cls) -> "OmniRouterConfig":
        """Small dense model (~85M params) - equivalent to dense-asr-libriheavy-0.08b.

        No MoE, suitable for baseline comparisons and smaller deployments.
        """
        return cls(
            d_model=512,
            n_layers=12,
            n_heads=8,
            mlp_dim=2048,
            vocab_size=4096,
            stacking=4,
            dropout=0.1,
            n_experts="0",
            share_router=False,
            ln_position=LayerNormPosition.PRE,
            cape_config=CAPEConfig(
                max_global_shift=5.0,
                max_local_shift=0.5,
                normalize=True,
            ),
        )

    @classmethod
    def base(cls) -> "OmniRouterConfig":
        """Base MoE model (~250M params) - equivalent to omni-router-asr-libriheavy-0.5b.

        4 experts per layer with shared routing for improved specialization.
        """
        return cls(
            d_model=768,
            n_layers=16,
            n_heads=12,
            mlp_dim=3072,
            vocab_size=4096,
            stacking=4,
            dropout=0.1,
            n_experts="4x4-4x4-4x4-4x4",
            share_router=True,
            load_balance_loss_weight=0.01,
            moe_jitter_eps=0.1,
            ln_position=LayerNormPosition.PRE,
            cape_config=CAPEConfig(
                max_global_shift=5.0,
                max_local_shift=0.5,
                normalize=True,
            ),
        )

    @classmethod
    def large(cls) -> "OmniRouterConfig":
        """Large MoE model (~613M params) - omni-router-speechcrawl-streaming-asr-0.6b.

        Hierarchical expert configuration with increasing experts in deeper layers.
        Uses shared routing within each expert group.
        """
        return cls(
            d_model=1024,
            n_layers=24,
            n_heads=16,
            mlp_dim=4096,
            vocab_size=4096,
            stacking=4,
            dropout=0.1,
            n_experts="6x2-6x4-6x8-6x16",
            share_router=True,
            load_balance_loss_weight=0.01,
            moe_jitter_eps=0.1,
            ln_position=LayerNormPosition.PRE,
            mask_mode=MaskMode.CAUSAL,
            cape_config=CAPEConfig(
                max_global_shift=5.0,
                max_local_shift=0.5,
                normalize=True,
            ),
        )

    @classmethod
    def streaming(cls) -> "OmniRouterConfig":
        """Streaming-optimized MoE model with causal masking.

        Suitable for real-time ASR applications.
        """
        return cls(
            d_model=768,
            n_layers=18,
            n_heads=12,
            mlp_dim=3072,
            vocab_size=4096,
            stacking=4,
            dropout=0.1,
            n_experts="6x4-6x4-6x8",
            share_router=True,
            load_balance_loss_weight=0.01,
            moe_jitter_eps=0.1,
            ln_position=LayerNormPosition.PRE,
            mask_mode=MaskMode.CAUSAL,
            cape_config=CAPEConfig(
                max_global_shift=0.0,
                max_local_shift=0.0,
                normalize=False,
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmniRouterConfig":
        """Create config from dictionary."""
        data = data.copy()
        data.pop("model_type", None)
        decoder_type = DecoderType(data.pop("decoder_type", "ctc"))

        if "ln_position" in data:
            data["ln_position"] = LayerNormPosition(data["ln_position"])
        if "mask_mode" in data:
            data["mask_mode"] = MaskMode(data["mask_mode"])
        if "cape_config" in data and isinstance(data["cape_config"], dict):
            data["cape_config"] = CAPEConfig(**data["cape_config"])

        return cls(decoder_type=decoder_type, **data)
