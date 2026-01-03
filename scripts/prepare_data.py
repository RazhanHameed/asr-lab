#!/usr/bin/env python
"""Prepare ASR training data from HuggingFace datasets.

This script downloads and converts ASR datasets to manifest format for training.
Target: ~15,000 hours of English speech data matching ABR model training setup.

Usage:
    # Prepare all datasets
    python scripts/prepare_data.py --output_dir data/manifests --all

    # Prepare specific datasets
    python scripts/prepare_data.py --output_dir data/manifests --datasets librispeech voxpopuli

    # Small subset for testing
    python scripts/prepare_data.py --output_dir data/manifests --datasets librispeech --max_samples 1000
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    hf_name: str
    hf_config: str | None
    splits: dict[str, str]  # manifest_suffix -> hf_split
    text_key: str = "text"
    audio_key: str = "audio"
    requires_auth: bool = False


# Dataset configurations matching ABR training setup
DATASETS: dict[str, DatasetConfig] = {
    "librispeech_clean_100": DatasetConfig(
        name="librispeech_clean_100",
        hf_name="librispeech_asr",
        hf_config="clean",
        splits={
            "train": "train.100",
            "dev": "validation",
            "test": "test",
        },
    ),
    "librispeech_clean_360": DatasetConfig(
        name="librispeech_clean_360",
        hf_name="librispeech_asr",
        hf_config="clean",
        splits={
            "train": "train.360",
        },
    ),
    "librispeech_other": DatasetConfig(
        name="librispeech_other",
        hf_name="librispeech_asr",
        hf_config="other",
        splits={
            "train": "train.500",
            "dev": "validation",
            "test": "test",
        },
    ),
    "voxpopuli": DatasetConfig(
        name="voxpopuli",
        hf_name="facebook/voxpopuli",
        hf_config="en_asr",
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
        text_key="raw_text",
    ),
    "common_voice": DatasetConfig(
        name="common_voice",
        hf_name="mozilla-foundation/common_voice_16_1",
        hf_config="en",
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
        text_key="sentence",
        requires_auth=True,
    ),
    "tedlium": DatasetConfig(
        name="tedlium",
        hf_name="LIUM/tedlium",
        hf_config="release3",
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
    ),
    "gigaspeech": DatasetConfig(
        name="gigaspeech",
        hf_name="speechcolab/gigaspeech",
        hf_config="l",  # l=10k hours, m=1k hours, s=250 hours
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
        requires_auth=True,
    ),
    "ami": DatasetConfig(
        name="ami",
        hf_name="edinburghcstr/ami",
        hf_config="ihm",
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
    ),
    "spgispeech": DatasetConfig(
        name="spgispeech",
        hf_name="kensho/spgispeech",
        hf_config="L",  # L=5k hours
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
        text_key="transcript",
        requires_auth=True,
    ),
    "earnings22": DatasetConfig(
        name="earnings22",
        hf_name="revdotcom/earnings22",
        hf_config=None,
        splits={
            "train": "train",
            "test": "test",
        },
        text_key="sentence",
    ),
    "peoples_speech": DatasetConfig(
        name="peoples_speech",
        hf_name="MLCommons/peoples_speech",
        hf_config="clean",  # clean subset
        splits={
            "train": "train",
            "dev": "validation",
            "test": "test",
        },
        requires_auth=True,
    ),
}


def prepare_dataset(
    config: DatasetConfig,
    output_dir: Path,
    audio_dir: Path,
    sample_rate: int = 16000,
    max_samples: int | None = None,
    use_streaming: bool = True,
) -> dict[str, Path]:
    """Prepare a single dataset.

    Args:
        config: Dataset configuration
        output_dir: Directory for manifest files
        audio_dir: Directory for audio files
        sample_rate: Target sample rate
        max_samples: Maximum samples per split
        use_streaming: Use streaming for large datasets

    Returns:
        Dictionary mapping split names to manifest paths
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_audio_dir = audio_dir / config.name
    dataset_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: dict[str, Path] = {}

    for split_suffix, hf_split in config.splits.items():
        manifest_path = output_dir / f"{config.name}_{split_suffix}.json"
        manifest_paths[split_suffix] = manifest_path

        print(f"\nProcessing {config.name} - {split_suffix}")

        # Load dataset
        try:
            if config.hf_config:
                ds = load_dataset(
                    config.hf_name,
                    config.hf_config,
                    split=hf_split,
                    streaming=use_streaming,
                )
            else:
                ds = load_dataset(
                    config.hf_name,
                    split=hf_split,
                    streaming=use_streaming,
                )
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            continue

        # Process samples
        count = 0
        total_duration = 0.0

        with open(manifest_path, "w") as f:
            for idx, item in enumerate(tqdm(ds, desc=f"  Processing")):
                if max_samples and idx >= max_samples:
                    break

                try:
                    # Extract audio
                    audio_data = item.get(config.audio_key, {})
                    if isinstance(audio_data, dict):
                        audio_array = audio_data.get("array")
                        audio_sr = audio_data.get("sampling_rate", sample_rate)
                    else:
                        continue

                    if audio_array is None or len(audio_array) == 0:
                        continue

                    # Extract text
                    text = item.get(config.text_key, "")
                    if not text or not isinstance(text, str):
                        continue

                    text = text.strip()
                    if not text:
                        continue

                    # Clean text
                    text = text.lower()
                    text = " ".join(text.split())  # Normalize whitespace

                    # Convert to tensor and resample
                    audio_tensor = torch.tensor(audio_array).float()
                    if audio_sr != sample_rate:
                        resampler = T.Resample(audio_sr, sample_rate)
                        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)

                    # Compute duration
                    duration = len(audio_tensor) / sample_rate

                    # Skip very short or very long audio
                    if duration < 0.5 or duration > 30.0:
                        continue

                    # Save audio file
                    audio_path = dataset_audio_dir / f"{split_suffix}_{idx:08d}.wav"
                    torchaudio.save(str(audio_path), audio_tensor.unsqueeze(0), sample_rate)

                    # Write manifest entry
                    entry = {
                        "audio_filepath": str(audio_path),
                        "text": text,
                        "duration": duration,
                    }
                    f.write(json.dumps(entry) + "\n")

                    count += 1
                    total_duration += duration

                except Exception as e:
                    if idx < 10:  # Only print first few errors
                        print(f"  Error processing sample {idx}: {e}")
                    continue

        hours = total_duration / 3600
        print(f"  Saved {count:,} samples ({hours:.1f} hours) to {manifest_path}")

    return manifest_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASR training data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/manifests",
        help="Directory for manifest files",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audio",
        help="Directory for audio files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Datasets to prepare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all datasets",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per split (for testing)",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Don't use streaming (downloads full dataset first)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir)

    # Determine which datasets to prepare
    if args.all:
        dataset_names = list(DATASETS.keys())
    elif args.datasets:
        dataset_names = args.datasets
    else:
        print("Error: Specify --datasets or --all")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return

    print(f"Preparing {len(dataset_names)} datasets:")
    for name in dataset_names:
        print(f"  - {name}")

    # Prepare each dataset
    all_manifests: dict[str, dict[str, Path]] = {}

    for name in dataset_names:
        config = DATASETS[name]

        if config.requires_auth:
            print(f"\nNote: {name} requires HuggingFace authentication")
            print("Run: huggingface-cli login")

        try:
            manifests = prepare_dataset(
                config,
                output_dir,
                audio_dir,
                sample_rate=args.sample_rate,
                max_samples=args.max_samples,
                use_streaming=not args.no_streaming,
            )
            all_manifests[name] = manifests
        except Exception as e:
            print(f"\nError preparing {name}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_train_hours = 0.0
    for name, manifests in all_manifests.items():
        print(f"\n{name}:")
        for split, path in manifests.items():
            if path.exists():
                # Count samples and hours
                hours = 0.0
                count = 0
                with open(path) as f:
                    for line in f:
                        data = json.loads(line)
                        hours += data.get("duration", 0) / 3600
                        count += 1
                print(f"  {split}: {count:,} samples, {hours:.1f} hours")
                if split == "train":
                    total_train_hours += hours

    print(f"\nTotal training data: {total_train_hours:.1f} hours")
    print("\nDone!")


if __name__ == "__main__":
    main()
