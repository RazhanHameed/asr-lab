"""Dataset utilities for ASR training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from asr_lab.audio.features import FeatureExtractor, load_audio
from asr_lab.tokenizers.base import Tokenizer


@dataclass
class ASRSample:
    """Single ASR training sample.

    Attributes:
        audio_path: Path to audio file
        text: Transcript text
        duration: Audio duration in seconds (optional)
        language: Language code (optional)
    """

    audio_path: str
    text: str
    duration: float | None = None
    language: str | None = None


class ASRDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for ASR training.

    Loads audio files and extracts features on-the-fly.
    """

    def __init__(
        self,
        samples: list[ASRSample],
        feature_extractor: FeatureExtractor,
        tokenizer: Tokenizer,
        sample_rate: int = 16000,
        max_audio_len: float | None = None,
        max_text_len: int | None = None,
        augment_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.augment_fn = augment_fn

        # Filter samples by length if specified
        if max_audio_len is not None or max_text_len is not None:
            self.samples = self._filter_samples()

    def _filter_samples(self) -> list[ASRSample]:
        """Filter samples by length constraints."""
        filtered = []
        for sample in self.samples:
            if self.max_audio_len is not None and sample.duration is not None:
                if sample.duration > self.max_audio_len:
                    continue
            if self.max_text_len is not None:
                if len(sample.text) > self.max_text_len:
                    continue
            filtered.append(sample)
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        waveform = load_audio(sample.audio_path, self.sample_rate)

        # Apply augmentation
        if self.augment_fn is not None:
            waveform = self.augment_fn(waveform)

        # Extract features
        features = self.feature_extractor(waveform)

        # Tokenize text
        tokens = torch.tensor(self.tokenizer.encode(sample.text), dtype=torch.long)

        return {
            "features": features.squeeze(0),  # (time, n_features)
            "feature_length": torch.tensor(features.size(1)),
            "tokens": tokens,
            "token_length": torch.tensor(len(tokens)),
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        feature_extractor: FeatureExtractor,
        tokenizer: Tokenizer,
        **kwargs: object,
    ) -> "ASRDataset":
        """Create dataset from a manifest file.

        Manifest format (JSON lines):
        {"audio_filepath": "path/to/audio.wav", "text": "transcript", "duration": 3.5}

        Args:
            manifest_path: Path to manifest file
            feature_extractor: Feature extractor
            tokenizer: Tokenizer
            **kwargs: Additional arguments for ASRDataset

        Returns:
            ASRDataset instance
        """
        import json

        samples = []
        with open(manifest_path) as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(
                    ASRSample(
                        audio_path=data["audio_filepath"],
                        text=data["text"],
                        duration=data.get("duration"),
                        language=data.get("language"),
                    )
                )

        return cls(samples, feature_extractor, tokenizer, **kwargs)


class ASRCollator:
    """Collate function for ASR batches with padding."""

    def __init__(
        self,
        feature_pad_value: float = 0.0,
        token_pad_value: int = 0,
    ) -> None:
        self.feature_pad_value = feature_pad_value
        self.token_pad_value = token_pad_value

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate batch with padding.

        Args:
            batch: List of samples from ASRDataset

        Returns:
            Collated batch with:
                - features: (batch, max_time, n_features)
                - feature_lengths: (batch,)
                - tokens: (batch, max_tokens)
                - token_lengths: (batch,)
        """
        # Get max lengths
        max_feature_len = max(s["features"].size(0) for s in batch)
        max_token_len = max(s["tokens"].size(0) for s in batch)
        n_features = batch[0]["features"].size(1)

        batch_size = len(batch)

        # Create padded tensors
        features = torch.full(
            (batch_size, max_feature_len, n_features),
            self.feature_pad_value,
            dtype=batch[0]["features"].dtype,
        )
        tokens = torch.full(
            (batch_size, max_token_len),
            self.token_pad_value,
            dtype=torch.long,
        )
        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Fill tensors
        for i, sample in enumerate(batch):
            feat_len = sample["features"].size(0)
            tok_len = sample["tokens"].size(0)

            features[i, :feat_len] = sample["features"]
            tokens[i, :tok_len] = sample["tokens"]
            feature_lengths[i] = feat_len
            token_lengths[i] = tok_len

        return {
            "features": features,
            "feature_lengths": feature_lengths,
            "tokens": tokens,
            "token_lengths": token_lengths,
        }


def create_librispeech_dataset(
    root: str | Path,
    split: str,
    feature_extractor: FeatureExtractor,
    tokenizer: Tokenizer,
    **kwargs: object,
) -> ASRDataset:
    """Create dataset from LibriSpeech.

    Args:
        root: Root directory for LibriSpeech
        split: Split name (train-clean-100, dev-clean, test-clean, etc.)
        feature_extractor: Feature extractor
        tokenizer: Tokenizer
        **kwargs: Additional arguments

    Returns:
        ASRDataset for LibriSpeech
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("librispeech_asr", split=split, cache_dir=str(root))

        samples = []
        for item in ds:
            samples.append(
                ASRSample(
                    audio_path=item["audio"]["path"],
                    text=item["text"].lower(),
                    duration=len(item["audio"]["array"]) / item["audio"]["sampling_rate"],
                )
            )

        return ASRDataset(samples, feature_extractor, tokenizer, **kwargs)

    except ImportError as e:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        ) from e
