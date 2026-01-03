# ASR Lab - Project Guide

## Overview

ASR Lab is a modular ASR (Automatic Speech Recognition) training framework supporting multiple architectures: SSM (Mamba2), Whisper-like, and Fast Conformer. The codebase is designed for reproducible training with multi-precision support.

## Quick Reference

```bash
# Install dependencies
uv sync --extra cuda --extra training

# Run training
uv run python scripts/train.py --model ssm --config base

# Run evaluation
uv run python scripts/evaluate.py --checkpoint outputs/best.pt --dataset fleurs

# Run tests
uv run pytest

# Lint code
uv run ruff check asr_lab/

# Type check
uv run mypy asr_lab/
```

## Project Structure

```
asr_lab/
├── models/           # ASR model architectures
│   ├── base.py       # Abstract base classes (ASRModel, Encoder, Decoder)
│   ├── ssm/          # Mamba2 SSM-based ASR
│   ├── whisper/      # Whisper-like encoder-decoder
│   └── conformer/    # Fast Conformer with CTC/RNNT
├── audio/            # Audio processing
│   ├── features.py   # Mel spectrogram, MFCC extraction
│   └── augmentation.py  # SpecAugment, noise injection
├── tokenizers/       # Text tokenization
│   └── base.py       # Character and BPE tokenizers
├── training/         # Training utilities
│   ├── precision.py  # Multi-precision (FP32, BF16, FP8, MXFP8)
│   ├── trainer.py    # Training loop with DDP/FSDP
│   └── dataset.py    # Dataset classes
├── evaluation/       # Evaluation utilities
│   └── evaluator.py  # WER/CER evaluation
├── reproducibility/  # Reproducibility utilities
│   ├── seeding.py    # Deterministic seeding
│   ├── deterministic.py  # Deterministic mode
│   ├── hash.py       # Model/tensor hashing
│   └── environment.py    # Environment capture
└── utils/
    └── metrics.py    # WER, CER computation
scripts/
├── train.py          # Training entrypoint
└── evaluate.py       # Evaluation entrypoint
```

## Architecture Patterns

### Model Creation
All models follow the same pattern:
```python
from asr_lab.models.ssm import SSMASRModel, SSMConfig
from asr_lab.models.whisper import WhisperASRModel, WhisperConfig
from asr_lab.models.conformer import ConformerASRModel, ConformerConfig

# Use preset configurations
model = SSMASRModel(SSMConfig.base())
model = WhisperASRModel(WhisperConfig.small())
model = ConformerASRModel(ConformerConfig.fast_conformer())
```

### Base Classes
- `ASRModel` - Abstract base for all ASR models (in `models/base.py`)
- `Encoder` - Abstract encoder interface
- `Decoder` - Abstract decoder interface (CTCDecoder, TransducerDecoder)
- `ModelConfig` - Base configuration dataclass

### Precision Management
```python
from asr_lab.training import get_precision_manager, PrecisionMode

precision = get_precision_manager(PrecisionMode.BF16)
with precision.autocast():
    output = model(input)
```

## Code Style

- Python 3.11+ with full type hints
- Line length: 100 characters
- Linter: ruff (rules: E, F, W, I, N, UP, B, C4, ANN)
- Type checker: mypy (strict mode)
- All public functions need docstrings

## Key Dependencies

- `torch>=2.5.0` - Core ML framework
- `torchaudio>=2.5.0` - Audio processing
- `einops>=0.8.0` - Tensor operations
- `mamba-ssm>=2.0.0` - Mamba2 SSM (CUDA only)
- `flash-attn>=2.5.0` - Flash Attention 2 (CUDA only)

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=asr_lab

# Run specific test file
uv run pytest tests/test_models.py
```

## Common Tasks

### Adding a New Model
1. Create directory under `asr_lab/models/your_model/`
2. Implement `config.py` with `YourConfig(ModelConfig)`
3. Implement `encoder.py` with `YourEncoder(Encoder)`
4. Implement `model.py` with `YourASRModel(ASRModel)`
5. Export in `__init__.py`
6. Add to `asr_lab/models/__init__.py`

### Adding a New Precision Mode
1. Add enum value to `PrecisionMode` in `training/precision.py`
2. Implement precision manager class
3. Register in `get_precision_manager()`

## GPU Requirements

| Feature | Minimum GPU |
|---------|-------------|
| Basic training | Any CUDA GPU |
| Flash Attention | V100+ |
| BF16 precision | A100+ |
| FP8 precision | H100+ |
| MXFP8/MXFP4 | B200+ |

## Environment Variables

- `CUDA_VISIBLE_DEVICES` - GPU selection
- `CUBLAS_WORKSPACE_CONFIG` - Set to `:4096:8` for deterministic cuBLAS
- `PYTHONHASHSEED` - Set for reproducible hashing
