"""Omni-Router MoE ASR model.

This module implements Apple's Omni-Router MoE architecture from ASRU 2025,
which uses shared routing across MoE layers for improved expert specialization.
"""

from asr_lab.models.omni_router.config import OmniRouterConfig
from asr_lab.models.omni_router.encoder import OmniRouterEncoder
from asr_lab.models.omni_router.model import OmniRouterASRModel

__all__ = [
    "OmniRouterConfig",
    "OmniRouterEncoder",
    "OmniRouterASRModel",
]
