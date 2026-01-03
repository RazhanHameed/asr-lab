"""Manifest file utilities for ASR datasets.

This module provides utilities for creating and managing manifest files
for ASR training, including conversion from HuggingFace datasets.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torchaudio
from tqdm import tqdm


@dataclass
class ManifestEntry:
    """Single entry in a manifest file."""

    audio_filepath: str
    text: str
    duration: float


class ManifestWriter:
    """Writer for manifest files in JSON-lines format."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w")
        self._count = 0
        self._total_duration = 0.0

    def write(self, entry: ManifestEntry) -> None:
        """Write a single entry to the manifest."""
        data = {
            "audio_filepath": entry.audio_filepath,
            "text": entry.text,
            "duration": entry.duration,
        }
        self._file.write(json.dumps(data) + "\n")
        self._count += 1
        self._total_duration += entry.duration

    def close(self) -> None:
        """Close the manifest file."""
        self._file.close()
        hours = self._total_duration / 3600
        print(f"Wrote {self._count} entries ({hours:.1f} hours) to {self.output_path}")

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def create_manifest_from_hf(
    dataset_name: str,
    output_dir: str | Path,
    hf_name: str,
    hf_config: str | None = None,
    split: str = "train",
    audio_dir: str | Path | None = None,
    sample_rate: int = 16000,
    max_samples: int | None = None,
) -> Path:
    """Create manifest from HuggingFace dataset.

    Args:
        dataset_name: Name for the output manifest
        output_dir: Directory for manifest file
        hf_name: HuggingFace dataset name
        hf_config: HuggingFace dataset configuration
        split: Dataset split to use
        audio_dir: Directory to save extracted audio files
        sample_rate: Target sample rate
        max_samples: Maximum number of samples (for debugging)

    Returns:
        Path to created manifest file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if audio_dir is None:
        audio_dir = output_dir / "audio" / dataset_name
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{dataset_name}_{split}.json"

    print(f"Loading dataset: {hf_name} ({hf_config}) - {split}")

    # Load dataset with streaming for large datasets
    if hf_config:
        ds = load_dataset(hf_name, hf_config, split=split, streaming=True, trust_remote_code=True)
    else:
        ds = load_dataset(hf_name, split=split, streaming=True, trust_remote_code=True)

    with ManifestWriter(manifest_path) as writer:
        for idx, item in enumerate(tqdm(ds, desc=f"Processing {dataset_name}")):
            if max_samples and idx >= max_samples:
                break

            try:
                # Extract audio
                audio_data = item.get("audio", {})
                if isinstance(audio_data, dict):
                    audio_array = audio_data.get("array")
                    audio_sr = audio_data.get("sampling_rate", sample_rate)
                else:
                    continue

                if audio_array is None:
                    continue

                # Extract text
                text = item.get("text", item.get("sentence", item.get("transcription", "")))
                if not text or not isinstance(text, str):
                    continue

                text = text.strip()
                if not text:
                    continue

                # Save audio file
                audio_path = audio_dir / f"{dataset_name}_{idx:08d}.wav"

                # Resample if needed
                import torch
                import torchaudio.transforms as T

                audio_tensor = torch.tensor(audio_array).float()
                if audio_sr != sample_rate:
                    resampler = T.Resample(audio_sr, sample_rate)
                    audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)

                # Save
                torchaudio.save(str(audio_path), audio_tensor.unsqueeze(0), sample_rate)

                # Compute duration
                duration = len(audio_tensor) / sample_rate

                writer.write(
                    ManifestEntry(
                        audio_filepath=str(audio_path),
                        text=text,
                        duration=duration,
                    )
                )

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

    return manifest_path


def create_manifest_from_audio_dir(
    audio_dir: str | Path,
    transcript_file: str | Path,
    output_path: str | Path,
    audio_ext: str = ".wav",
) -> Path:
    """Create manifest from directory of audio files with transcript file.

    Args:
        audio_dir: Directory containing audio files
        transcript_file: File with transcripts (format: audio_id<tab>text)
        output_path: Output manifest path
        audio_ext: Audio file extension

    Returns:
        Path to created manifest file
    """
    audio_dir = Path(audio_dir)
    output_path = Path(output_path)

    # Load transcripts
    transcripts: dict[str, str] = {}
    with open(transcript_file) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

    with ManifestWriter(output_path) as writer:
        for audio_path in tqdm(sorted(audio_dir.glob(f"*{audio_ext}"))):
            audio_id = audio_path.stem
            text = transcripts.get(audio_id)
            if not text:
                continue

            # Get duration
            info = sf.info(str(audio_path))
            duration = info.duration

            writer.write(
                ManifestEntry(
                    audio_filepath=str(audio_path),
                    text=text,
                    duration=duration,
                )
            )

    return output_path


def merge_manifests(
    manifest_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Merge multiple manifest files into one.

    Args:
        manifest_paths: List of manifest file paths
        output_path: Output manifest path

    Returns:
        Path to merged manifest file
    """
    output_path = Path(output_path)

    with ManifestWriter(output_path) as writer:
        for manifest_path in manifest_paths:
            with open(manifest_path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    writer.write(
                        ManifestEntry(
                            audio_filepath=data["audio_filepath"],
                            text=data["text"],
                            duration=data["duration"],
                        )
                    )

    return output_path


def compute_manifest_stats(manifest_path: str | Path) -> dict:
    """Compute statistics for a manifest file.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Dictionary with statistics
    """
    total_duration = 0.0
    total_samples = 0
    text_lengths: list[int] = []

    with open(manifest_path) as f:
        for line in f:
            data = json.loads(line.strip())
            total_duration += data.get("duration", 0.0)
            total_samples += 1
            text_lengths.append(len(data.get("text", "")))

    return {
        "total_samples": total_samples,
        "total_hours": total_duration / 3600,
        "avg_duration": total_duration / total_samples if total_samples > 0 else 0,
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
    }
