#!/usr/bin/env python
"""Simple script to prepare LibriSpeech data for training.

This downloads LibriSpeech using torchaudio and creates manifest files.
"""

import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def prepare_librispeech(
    output_dir: str = "/data/razhan/asr-lab/data/manifests",
    split: str = "train-clean-100",
    root: str = "/data/razhan/asr-lab/data/librispeech",
    max_samples: int | None = None,
) -> Path:
    """Prepare LibriSpeech dataset.

    Args:
        output_dir: Directory for manifest files
        split: LibriSpeech split (train-clean-100, train-clean-360, etc.)
        root: Root directory for downloaded data
        max_samples: Maximum samples (for testing)

    Returns:
        Path to manifest file
    """
    from torchaudio.datasets import LIBRISPEECH

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map split names
    split_name = split.replace("-", "_")
    manifest_path = output_dir / f"librispeech_{split_name}.json"

    print(f"Downloading/loading LibriSpeech {split}...")
    dataset = LIBRISPEECH(root=root, url=split, download=True)

    print(f"Processing {len(dataset)} samples...")
    entries = []

    for idx, (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) in enumerate(
        tqdm(dataset, desc=f"Processing {split}")
    ):
        if max_samples and idx >= max_samples:
            break

        # Compute duration
        duration = waveform.size(1) / sample_rate

        # Skip very short or long audio
        if duration < 0.5 or duration > 30.0:
            continue

        # Get audio path
        audio_path = Path(root) / "LibriSpeech" / split / str(speaker_id) / str(chapter_id) / f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"

        if not audio_path.exists():
            # Try alternative path format
            audio_path = Path(root) / "LibriSpeech" / split / str(speaker_id) / str(chapter_id) / f"{speaker_id}-{chapter_id}-{utterance_id}.flac"

        entries.append({
            "audio_filepath": str(audio_path),
            "text": transcript.lower(),
            "duration": duration,
        })

    # Write manifest
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    hours = sum(e["duration"] for e in entries) / 3600
    print(f"Saved {len(entries)} samples ({hours:.1f} hours) to {manifest_path}")

    return manifest_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LibriSpeech data")
    parser.add_argument("--output_dir", type=str, default="/data/razhan/asr-lab/data/manifests")
    parser.add_argument("--root", type=str, default="/data/razhan/asr-lab/data/librispeech")
    parser.add_argument("--splits", type=str, nargs="+",
                       default=["train-clean-100"],
                       help="Splits to prepare")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    for split in args.splits:
        prepare_librispeech(
            output_dir=args.output_dir,
            split=split,
            root=args.root,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
