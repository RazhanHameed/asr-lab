"""Multi-dataset support for ASR training.

This module provides utilities for loading and combining multiple ASR datasets
to create large-scale training sets similar to the ABR model training setup.
"""

import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from asr_lab.audio.features import FeatureExtractor, load_audio
from asr_lab.tokenizers.base import Tokenizer


@dataclass
class DatasetInfo:
    """Information about an ASR dataset.

    Attributes:
        name: Dataset name
        hf_name: HuggingFace dataset name
        hf_config: HuggingFace dataset configuration
        splits: Available splits
        hours: Approximate hours of audio
        language: Language code
    """

    name: str
    hf_name: str
    hf_config: str | None = None
    splits: list[str] = field(default_factory=list)
    hours: float = 0.0
    language: str = "en"


# Dataset registry matching ABR training setup
DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "librispeech_clean": DatasetInfo(
        name="librispeech_clean",
        hf_name="librispeech_asr",
        hf_config="clean",
        splits=["train.100", "train.360"],
        hours=460.0,
        language="en",
    ),
    "librispeech_other": DatasetInfo(
        name="librispeech_other",
        hf_name="librispeech_asr",
        hf_config="other",
        splits=["train.500"],
        hours=500.0,
        language="en",
    ),
    "voxpopuli": DatasetInfo(
        name="voxpopuli",
        hf_name="facebook/voxpopuli",
        hf_config="en",
        splits=["train"],
        hours=540.0,
        language="en",
    ),
    "gigaspeech": DatasetInfo(
        name="gigaspeech",
        hf_name="speechcolab/gigaspeech",
        hf_config="l",
        splits=["train"],
        hours=10000.0,
        language="en",
    ),
    "common_voice": DatasetInfo(
        name="common_voice",
        hf_name="mozilla-foundation/common_voice_16_1",
        hf_config="en",
        splits=["train"],
        hours=2500.0,
        language="en",
    ),
    "tedlium": DatasetInfo(
        name="tedlium",
        hf_name="LIUM/tedlium",
        hf_config="release3",
        splits=["train"],
        hours=450.0,
        language="en",
    ),
    "ami_ihm": DatasetInfo(
        name="ami_ihm",
        hf_name="edinburghcstr/ami",
        hf_config="ihm",
        splits=["train"],
        hours=100.0,
        language="en",
    ),
    "spgispeech": DatasetInfo(
        name="spgispeech",
        hf_name="kensho/spgispeech",
        hf_config=None,
        splits=["train"],
        hours=5000.0,
        language="en",
    ),
    "earnings22": DatasetInfo(
        name="earnings22",
        hf_name="revdotcom/earnings22",
        hf_config=None,
        splits=["train"],
        hours=119.0,
        language="en",
    ),
    "peoples_speech": DatasetInfo(
        name="peoples_speech",
        hf_name="MLCommons/peoples_speech",
        hf_config=None,
        splits=["train"],
        hours=30000.0,
        language="en",
    ),
    "mls": DatasetInfo(
        name="mls",
        hf_name="facebook/multilingual_librispeech",
        hf_config="english",
        splits=["train"],
        hours=44000.0,
        language="en",
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """Get dataset information by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


@dataclass
class ASRDatasetConfig:
    """Configuration for ASR dataset loading.

    Attributes:
        manifest_paths: List of manifest file paths
        sample_rate: Audio sample rate
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        max_text_len: Maximum text length
        shuffle: Whether to shuffle data
        seed: Random seed
    """

    manifest_paths: list[str | Path] = field(default_factory=list)
    sample_rate: int = 16000
    max_duration: float = 30.0
    min_duration: float = 0.5
    max_text_len: int = 512
    shuffle: bool = True
    seed: int = 42


@dataclass
class ASRSample:
    """Single ASR sample."""

    audio_path: str
    text: str
    duration: float
    dataset: str = ""


class CombinedASRDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Combined ASR dataset from multiple manifest files.

    Uses streaming/iterable approach for memory efficiency with large datasets.
    Supports weighted sampling across datasets.
    """

    def __init__(
        self,
        config: ASRDatasetConfig,
        feature_extractor: FeatureExtractor,
        tokenizer: Tokenizer,
        dataset_weights: dict[str, float] | None = None,
        augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.augment_fn = augment_fn

        # Load all manifests
        self.samples_by_dataset: dict[str, list[ASRSample]] = {}
        self._load_manifests()

        # Compute sampling weights
        self.dataset_weights = dataset_weights or {}
        self._compute_weights()

    def _load_manifests(self) -> None:
        """Load all manifest files."""
        for manifest_path in self.config.manifest_paths:
            path = Path(manifest_path)
            if not path.exists():
                print(f"Warning: Manifest not found: {path}")
                continue

            dataset_name = path.stem
            samples: list[ASRSample] = []

            with open(path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    duration = data.get("duration", 0.0)

                    # Filter by duration
                    if duration < self.config.min_duration:
                        continue
                    if duration > self.config.max_duration:
                        continue

                    # Filter by text length
                    text = data.get("text", "").strip()
                    if len(text) > self.config.max_text_len:
                        continue
                    if len(text) == 0:
                        continue

                    samples.append(
                        ASRSample(
                            audio_path=data["audio_filepath"],
                            text=text.lower(),
                            duration=duration,
                            dataset=dataset_name,
                        )
                    )

            if samples:
                self.samples_by_dataset[dataset_name] = samples
                print(f"Loaded {len(samples)} samples from {dataset_name}")

    def _compute_weights(self) -> None:
        """Compute dataset sampling probabilities."""
        total_hours = sum(
            sum(s.duration for s in samples) / 3600
            for samples in self.samples_by_dataset.values()
        )

        self.dataset_probs: dict[str, float] = {}
        for name, samples in self.samples_by_dataset.items():
            hours = sum(s.duration for s in samples) / 3600
            weight = self.dataset_weights.get(name, 1.0)
            self.dataset_probs[name] = (hours / total_hours) * weight

        # Normalize
        total_prob = sum(self.dataset_probs.values())
        for name in self.dataset_probs:
            self.dataset_probs[name] /= total_prob

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over samples with weighted sampling."""
        worker_info = get_worker_info()
        seed = self.config.seed
        if worker_info is not None:
            seed += worker_info.id

        rng = random.Random(seed)

        # Create shuffled indices for each dataset
        dataset_indices: dict[str, list[int]] = {}
        for name, samples in self.samples_by_dataset.items():
            indices = list(range(len(samples)))
            if self.config.shuffle:
                rng.shuffle(indices)
            dataset_indices[name] = indices

        dataset_positions: dict[str, int] = {name: 0 for name in self.samples_by_dataset}
        dataset_names = list(self.samples_by_dataset.keys())
        dataset_probs = [self.dataset_probs[name] for name in dataset_names]

        while True:
            # Sample dataset based on weights
            dataset_name = rng.choices(dataset_names, weights=dataset_probs, k=1)[0]
            samples = self.samples_by_dataset[dataset_name]
            indices = dataset_indices[dataset_name]
            pos = dataset_positions[dataset_name]

            # Check if we've exhausted this dataset
            if pos >= len(indices):
                # Reshuffle and restart
                if self.config.shuffle:
                    rng.shuffle(indices)
                dataset_positions[dataset_name] = 0
                pos = 0

            sample = samples[indices[pos]]
            dataset_positions[dataset_name] = pos + 1

            try:
                yield self._process_sample(sample)
            except Exception as e:
                print(f"Error processing {sample.audio_path}: {e}")
                continue

    def _process_sample(self, sample: ASRSample) -> dict[str, torch.Tensor]:
        """Process a single sample."""
        # Load audio
        waveform = load_audio(sample.audio_path, self.config.sample_rate)

        # Extract features
        features = self.feature_extractor(waveform)

        # Apply augmentation (SpecAugment operates on features, not waveform)
        if self.augment_fn is not None:
            features = self.augment_fn(features)

        # Tokenize text
        tokens = torch.tensor(self.tokenizer.encode(sample.text), dtype=torch.long)

        return {
            "features": features.squeeze(0),
            "feature_length": torch.tensor(features.size(1)),
            "tokens": tokens,
            "token_length": torch.tensor(len(tokens)),
        }


class MapStyleASRDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style ASR dataset for smaller datasets or validation."""

    def __init__(
        self,
        manifest_path: str | Path,
        feature_extractor: FeatureExtractor,
        tokenizer: Tokenizer,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

        self.samples: list[ASRSample] = []
        with open(manifest_path) as f:
            for line in f:
                data = json.loads(line.strip())
                duration = data.get("duration", 0.0)
                if duration > max_duration or duration < 0.1:
                    continue
                text = data.get("text", "").strip()
                if not text:
                    continue
                self.samples.append(
                    ASRSample(
                        audio_path=data["audio_filepath"],
                        text=text.lower(),
                        duration=duration,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        waveform = load_audio(sample.audio_path, self.sample_rate)
        features = self.feature_extractor(waveform)
        tokens = torch.tensor(self.tokenizer.encode(sample.text), dtype=torch.long)

        return {
            "features": features.squeeze(0),
            "feature_length": torch.tensor(features.size(1)),
            "tokens": tokens,
            "token_length": torch.tensor(len(tokens)),
        }


def create_combined_dataset(
    manifest_paths: list[str | Path],
    feature_extractor: FeatureExtractor,
    tokenizer: Tokenizer,
    max_duration: float = 30.0,
    shuffle: bool = True,
    seed: int = 42,
) -> CombinedASRDataset:
    """Create a combined dataset from multiple manifests.

    Args:
        manifest_paths: List of manifest file paths
        feature_extractor: Feature extractor
        tokenizer: Tokenizer
        max_duration: Maximum audio duration in seconds
        shuffle: Whether to shuffle data
        seed: Random seed

    Returns:
        Combined ASR dataset
    """
    config = ASRDatasetConfig(
        manifest_paths=list(manifest_paths),
        max_duration=max_duration,
        shuffle=shuffle,
        seed=seed,
    )
    return CombinedASRDataset(config, feature_extractor, tokenizer)
