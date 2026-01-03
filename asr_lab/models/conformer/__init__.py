"""Fast Conformer ASR model.

This module implements an efficient Conformer-based ASR with:
- Depthwise separable convolution subsampling
- Limited context self-attention
- Macaron-style feed-forward modules
"""

from asr_lab.models.conformer.config import ConformerConfig
from asr_lab.models.conformer.encoder import ConformerEncoder
from asr_lab.models.conformer.model import ConformerASRModel

__all__ = [
    "ConformerConfig",
    "ConformerEncoder",
    "ConformerASRModel",
]
