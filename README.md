# ASR Lab

A modular, reproducible ASR training framework supporting multiple architectures.

## Features

- **Multiple Architectures**: SSM (Mamba2), Whisper-like, Fast Conformer
- **Reproducible Training**: Integration with Microsoft RepDL for bitwise reproducibility
- **Multi-Precision**: FP32, FP16, BF16, FP8 (H100), MXFP8/MXFP4 (B200)
- **Streaming + Offline**: Both real-time and high-accuracy inference modes
- **Modular Design**: Mix and match encoders, decoders, and training strategies

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ASR Lab
cd asr-lab
uv sync --extra cuda --extra training

# Train SSM model on LibriSpeech (15 hours)
uv run python scripts/train.py \
    --model ssm \
    --config base \
    --dataset librispeech \
    --train-split train.clean.100 \
    --epochs 100

# Evaluate on FLEURS
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --dataset fleurs \
    --language en_us
```

## Supported Models

| Model | Architecture | Params | Use Case |
|-------|-------------|--------|----------|
| **SSM** | Mamba2 + Flash Attention | 5M-150M | Streaming ASR |
| **Whisper** | Encoder-Decoder Transformer | 5M-100M | High-accuracy offline |
| **Conformer** | Fast Conformer + CTC | 5M-100M | Balanced performance |

## Model Configurations

### SSM (State Space Model)

```python
from asr_lab.models.ssm import SSMASRModel, SSMConfig

# Create base model (~19M params)
model = SSMASRModel(SSMConfig.base())

# A100-optimized (~50M params)
model = SSMASRModel(SSMConfig.a100_optimized())

# B200-optimized (~150M params)
model = SSMASRModel(SSMConfig.b200_optimized())
```

### Whisper-like

```python
from asr_lab.models.whisper import WhisperASRModel, WhisperConfig

# Create base model (~19M params)
model = WhisperASRModel(WhisperConfig.base())

# Small model (~50M params)
model = WhisperASRModel(WhisperConfig.small())
```

### Fast Conformer

```python
from asr_lab.models.conformer import ConformerASRModel, ConformerConfig

# Create base model (~19M params)
model = ConformerASRModel(ConformerConfig.base())

# Fast Conformer with limited context attention (2.4x faster)
model = ConformerASRModel(ConformerConfig.fast_conformer())
```

## Training

### Single GPU

```bash
uv run python scripts/train.py \
    --model ssm \
    --config base \
    --dataset librispeech \
    --batch-size 32 \
    --precision bf16
```

### Multi-GPU (DDP)

```bash
uv run torchrun --nproc-per-node 8 scripts/train.py \
    --model conformer \
    --config large \
    --batch-size 16 \
    --gradient-accumulation 4
```

### Multi-GPU (FSDP)

```bash
uv run torchrun --nproc-per-node 8 scripts/train.py \
    --model ssm \
    --config b200_optimized \
    --fsdp \
    --batch-size 4
```

### Precision Options

| Precision | GPU | Command |
|-----------|-----|---------|
| FP32 | Any | `--precision fp32` |
| FP16 | V100+ | `--precision fp16` |
| BF16 | A100+ | `--precision bf16` |
| FP8 | H100+ | `--precision fp8` |
| MXFP8 | B200+ | `--precision mxfp8` |
| MXFP4 | B200+ | `--precision mxfp4` |

## Evaluation

### FLEURS (102 Languages)

```bash
# Single language
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --dataset fleurs \
    --language en_us

# Multiple languages
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --dataset fleurs \
    --language en_us,de_de,fr_fr

# All 102 languages
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --dataset fleurs \
    --language all
```

### LibriSpeech

```bash
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --dataset librispeech \
    --split test.clean
```

### Streaming vs Offline Comparison

```bash
uv run python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_best.pt \
    --compare-streaming
```

## Project Structure

```
asr-lab/
├── asr_lab/
│   ├── models/
│   │   ├── base.py           # Abstract base classes
│   │   ├── ssm/              # SSM (Mamba2) model
│   │   ├── whisper/          # Whisper-like model
│   │   └── conformer/        # Fast Conformer model
│   ├── audio/
│   │   ├── features.py       # Feature extraction
│   │   └── augmentation.py   # SpecAugment, etc.
│   ├── tokenizers/
│   │   └── base.py           # Character & BPE tokenizers
│   ├── training/
│   │   ├── precision.py      # Multi-precision support
│   │   ├── trainer.py        # Training loop
│   │   └── dataset.py        # Dataset utilities
│   ├── evaluation/
│   │   └── evaluator.py      # Evaluation utilities
│   └── utils/
│       └── metrics.py        # WER, CER computation
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── docs/                     # Documentation
├── configs/                  # YAML configurations
├── tests/                    # Unit tests
└── pyproject.toml            # Project configuration
```

## Reproducible Training

ASR Lab integrates with [Microsoft RepDL](https://github.com/microsoft/RepDL) for bitwise reproducible training across different hardware:

```bash
# Install with reproducibility support
uv sync --extra reproducible

# Train with reproducible operations
uv run python scripts/train.py \
    --model ssm \
    --reproducible
```

## Installation

### With uv (Recommended)

```bash
# Base installation
uv sync

# With CUDA optimizations
uv sync --extra cuda

# With training dependencies
uv sync --extra training

# Full installation
uv sync --all-extras
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `mamba` | Mamba SSM CUDA kernels |
| `flash` | Flash Attention 2 |
| `cuda` | All CUDA optimizations |
| `training` | Datasets, TensorBoard, W&B |
| `transformer-engine` | NVIDIA FP8 support |
| `torchao` | TorchAO (MXFP8, MXFP4) |
| `reproducible` | Microsoft RepDL |
| `dev` | Development tools |

## Python API

```python
import torch
from asr_lab.models.ssm import SSMASRModel, SSMConfig
from asr_lab.audio.features import MelSpectrogramExtractor
from asr_lab.tokenizers import CharacterTokenizer
from asr_lab.training import get_precision_manager

# Create model
config = SSMConfig.base()
model = SSMASRModel(config).cuda().eval()

# Setup components
feature_extractor = MelSpectrogramExtractor.whisper_style().cuda()
tokenizer = CharacterTokenizer()
precision = get_precision_manager("bf16")

# Transcribe
import torchaudio
waveform, sr = torchaudio.load("audio.wav")
waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

with torch.no_grad(), precision.autocast():
    features = feature_extractor(waveform.cuda())
    tokens = model.transcribe(features)
    text = tokenizer.decode(tokens[0])

print(text)
```

## References

- [Microsoft RepDL](https://github.com/microsoft/RepDL) - Reproducible Deep Learning
- [Mamba](https://arxiv.org/abs/2312.00752) - Linear-Time Sequence Modeling
- [Mamba-2](https://arxiv.org/abs/2405.21060) - Structured State Space Duality
- [Fast Conformer](https://arxiv.org/abs/2305.05084) - Efficient Speech Recognition
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Efficient Attention
- [Whisper](https://arxiv.org/abs/2212.04356) - Robust Speech Recognition

## License

MIT
