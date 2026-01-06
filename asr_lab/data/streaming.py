"""Streaming dataset utilities for large-scale ASR training."""

from typing import Iterator, Callable
import numpy as np
import torch
from torch.utils.data import IterableDataset

from asr_lab.audio.features import FeatureExtractor
from asr_lab.tokenizers.base import Tokenizer


class StreamingASRDataset(IterableDataset):
    """Streaming dataset for ASR training from HuggingFace datasets.

    Supports duration-limited epochs for large datasets like YODAS-Granary.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        hf_subset: str,
        hf_split: str,
        feature_extractor: FeatureExtractor,
        tokenizer: Tokenizer,
        text_key: str = "text",
        audio_key: str = "audio",
        duration_key: str = "duration",
        max_epoch_hours: float | None = None,
        max_audio_duration: float = 30.0,
        min_audio_duration: float = 0.5,
        shuffle_buffer_size: int = 10000,
        sample_rate: int = 16000,
        seed: int = 42,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset
        self.hf_split = hf_split
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.audio_key = audio_key
        self.duration_key = duration_key
        self.max_epoch_hours = max_epoch_hours
        self.max_epoch_seconds = max_epoch_hours * 3600 if max_epoch_hours else float('inf')
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        self.shuffle_buffer_size = shuffle_buffer_size
        self.sample_rate = sample_rate
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling reproducibility."""
        self._epoch = epoch

    def _load_hf_dataset(self) -> Iterator:
        """Load HuggingFace dataset with streaming and retry logic."""
        import time
        from datasets import load_dataset

        max_retries = 15
        for attempt in range(max_retries):
            try:
                ds = load_dataset(
                    self.hf_dataset_name,
                    self.hf_subset,
                    split=self.hf_split,
                    streaming=True,
                )
                # Shuffle with epoch-dependent seed
                ds = ds.shuffle(seed=self.seed + self._epoch, buffer_size=self.shuffle_buffer_size)
                return iter(ds)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt * 2, 120)  # Cap at 2 minutes
                    print(f"Dataset loading failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to load dataset after {max_retries} attempts: {e}")

    def _process_sample(self, item: dict) -> dict | None:
        """Process a single sample from the dataset."""
        try:
            # Check duration
            duration = item.get(self.duration_key, 0)
            if duration < self.min_audio_duration or duration > self.max_audio_duration:
                return None

            # Get text
            text = item.get(self.text_key, "")
            if not text or not isinstance(text, str):
                return None
            text = text.strip().lower()
            if not text:
                return None

            # Get audio
            audio_data = item.get(self.audio_key)
            if audio_data is None:
                return None

            # Handle AudioDecoder format (HuggingFace datasets 4.x with torchcodec)
            if hasattr(audio_data, 'metadata'):
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                audio_sr = audio_data.metadata.sample_rate
            elif isinstance(audio_data, dict):
                audio_array = np.array(audio_data.get("array"), dtype=np.float32)
                audio_sr = audio_data.get("sampling_rate", self.sample_rate)
            else:
                return None

            if audio_array is None or len(audio_array) == 0:
                return None

            # Resample if needed
            if audio_sr != self.sample_rate:
                import torchaudio.transforms as T
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                resampler = T.Resample(audio_sr, self.sample_rate)
                audio_array = resampler(audio_tensor).squeeze(0).numpy()

            # Extract features
            waveform = torch.from_numpy(audio_array)
            features = self.feature_extractor(waveform)

            # Tokenize text
            tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

            return {
                "features": features.squeeze(0),  # (time, n_features)
                "feature_length": torch.tensor(features.size(1)),
                "tokens": tokens,
                "token_length": torch.tensor(len(tokens)),
                "duration": duration,
            }
        except Exception:
            return None

    def __iter__(self) -> Iterator[dict]:
        """Iterate over samples with duration limit and retry on errors."""
        import time

        total_duration = 0.0
        consecutive_errors = 0
        max_consecutive_errors = 10

        while total_duration < self.max_epoch_seconds:
            try:
                ds_iter = self._load_hf_dataset()
                for item in ds_iter:
                    if total_duration >= self.max_epoch_seconds:
                        return

                    sample = self._process_sample(item)
                    if sample is None:
                        continue

                    total_duration += sample["duration"]
                    del sample["duration"]
                    consecutive_errors = 0  # Reset on success
                    yield sample

                # Normal end of dataset
                return

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(f"Too many consecutive errors: {e}")
                wait_time = min(2 ** consecutive_errors * 2, 120)
                print(f"Streaming error (attempt {consecutive_errors}): {e}")
                print(f"Retrying in {wait_time}s... (total duration so far: {total_duration/3600:.1f}h)")
                time.sleep(wait_time)


class StreamingASRCollator:
    """Collate function for streaming ASR batches with padding."""

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
        """Collate batch with padding."""
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


def create_yodas_english_dataset(
    feature_extractor: FeatureExtractor,
    tokenizer: Tokenizer,
    max_epoch_hours: float = 15000,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
) -> StreamingASRDataset:
    """Create streaming dataset for YODAS-Granary English."""
    return StreamingASRDataset(
        hf_dataset_name="espnet/yodas-granary",
        hf_subset="English",
        hf_split="asr_only",
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        text_key="text",
        audio_key="audio",
        duration_key="duration",
        max_epoch_hours=max_epoch_hours,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
