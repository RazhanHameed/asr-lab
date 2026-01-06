#!/usr/bin/env python
"""Fast parallel ASR data preparation using multiprocessing.

Optimized for machines with many CPU cores and large RAM.
Uses parallel processing at multiple levels for maximum throughput.

Usage:
    python scripts/prepare_data_fast.py --15k --workers 32
"""

import argparse
import json
import os
import queue
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import Manager, Process, Queue, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

# Disable tokenizers parallelism to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "false"


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    hf_name: str
    hf_config: str | None
    splits: dict[str, str]
    text_key: str = "text"
    audio_key: str = "audio"
    requires_auth: bool = False
    approx_hours: float = 0.0


DATASET_TARGET_HOURS: dict[str, float] = {
    "librispeech_clean_100": 100,
    "librispeech_clean_360": 360,
    "librispeech_other": 500,
    "mls_english": 400,
    "peoples_speech": 13500,
}

DATASETS: dict[str, DatasetConfig] = {
    "librispeech_clean_100": DatasetConfig(
        name="librispeech_clean_100",
        hf_name="librispeech_asr",
        hf_config="clean",
        splits={"train": "train.100", "dev": "validation", "test": "test"},
        approx_hours=100,
    ),
    "librispeech_clean_360": DatasetConfig(
        name="librispeech_clean_360",
        hf_name="librispeech_asr",
        hf_config="clean",
        splits={"train": "train.360"},
        approx_hours=360,
    ),
    "librispeech_other": DatasetConfig(
        name="librispeech_other",
        hf_name="librispeech_asr",
        hf_config="other",
        splits={"train": "train.500", "dev": "validation", "test": "test"},
        approx_hours=500,
    ),
    "mls_english": DatasetConfig(
        name="mls_english",
        hf_name="parler-tts/mls_eng",
        hf_config=None,
        splits={"train": "train", "dev": "dev", "test": "test"},
        text_key="transcript",
        approx_hours=44000,
    ),
    "peoples_speech": DatasetConfig(
        name="peoples_speech",
        hf_name="MLCommons/peoples_speech",
        hf_config="clean",
        splits={"train": "train"},
        text_key="text",
        requires_auth=False,
        approx_hours=12000,
    ),
}


def process_sample(args: tuple) -> dict | None:
    """Process a single sample - designed for parallel execution."""
    idx, item, audio_key, text_key, sample_rate, audio_dir, split_suffix = args

    try:
        # Extract audio - handle both AudioDecoder and dict formats
        audio_data = item.get(audio_key)
        if audio_data is None:
            return None

        # Handle new AudioDecoder format (HuggingFace datasets 4.x with torchcodec)
        if hasattr(audio_data, 'metadata'):
            audio_array = audio_data['array']
            audio_sr = audio_data.metadata.sample_rate
        # Handle legacy dict format
        elif isinstance(audio_data, dict):
            audio_array = audio_data.get("array")
            audio_sr = audio_data.get("sampling_rate", sample_rate)
        else:
            return None

        if audio_array is None or len(audio_array) == 0:
            return None

        # Extract text
        text = item.get(text_key, "")
        if not text or not isinstance(text, str):
            return None

        text = text.strip().lower()
        text = " ".join(text.split())
        if not text:
            return None

        # Convert to numpy
        audio_array = np.array(audio_array, dtype=np.float32)

        # Resample if needed
        if audio_sr != sample_rate:
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            resampler = T.Resample(audio_sr, sample_rate)
            audio_array = resampler(audio_tensor).squeeze(0).numpy()

        # Compute duration
        duration = len(audio_array) / sample_rate

        # Skip very short or very long audio
        if duration < 0.5 or duration > 30.0:
            return None

        # Save audio file using soundfile (faster than torchaudio)
        audio_path = audio_dir / f"{split_suffix}_{idx:08d}.wav"
        sf.write(str(audio_path), audio_array, sample_rate, subtype='PCM_16')

        return {
            "audio_filepath": str(audio_path),
            "text": text,
            "duration": duration,
        }
    except Exception:
        return None


def process_batch_threadpool(
    batch: list[tuple[int, dict]],
    audio_key: str,
    text_key: str,
    sample_rate: int,
    audio_dir: Path,
    split_suffix: str,
    num_threads: int = 8,
) -> list[dict]:
    """Process a batch of samples using thread pool."""
    args_list = [
        (idx, item, audio_key, text_key, sample_rate, audio_dir, split_suffix)
        for idx, item in batch
    ]

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_sample, args) for args in args_list]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return results


def prepare_dataset_fast(
    config: DatasetConfig,
    base_dir: Path,
    sample_rate: int = 16000,
    max_hours: float | None = None,
    num_workers: int = 8,
    batch_size: int = 500,
) -> dict[str, Path]:
    """Prepare a single dataset with parallel processing."""
    from datasets import load_dataset

    dataset_dir = base_dir / config.name
    output_dir = dataset_dir / "manifests"
    audio_dir = dataset_dir / "audio"

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: dict[str, Path] = {}

    for split_suffix, hf_split in config.splits.items():
        manifest_path = output_dir / f"{config.name}_{split_suffix}.json"
        manifest_paths[split_suffix] = manifest_path

        print(f"\n  Processing {config.name} - {split_suffix}")

        try:
            # Use streaming mode to avoid downloading entire dataset
            if config.hf_config:
                ds = load_dataset(
                    config.hf_name,
                    config.hf_config,
                    split=hf_split,
                    streaming=True,
                )
            else:
                ds = load_dataset(
                    config.hf_name,
                    split=hf_split,
                    streaming=True,
                )
        except Exception as e:
            print(f"    Error loading: {e}")
            continue

        # Calculate limits
        split_max_hours = max_hours if split_suffix == "train" else None
        max_seconds = split_max_hours * 3600 if split_max_hours else float('inf')

        total_duration = 0.0
        count = 0
        all_entries = []

        # Process in batches with parallel saving
        batch = []
        print(f"    Streaming dataset (target: {split_max_hours or 'unlimited'}h)")

        for idx, item in enumerate(ds):
            if total_duration >= max_seconds:
                print(f"    Reached {split_max_hours:.0f} hour limit at {idx:,} samples")
                break

            batch.append((idx, item))

            if len(batch) >= batch_size:
                # Process batch in parallel
                results = process_batch_threadpool(
                    batch,
                    config.audio_key,
                    config.text_key,
                    sample_rate,
                    audio_dir,
                    split_suffix,
                    num_threads=num_workers,
                )

                for entry in results:
                    total_duration += entry["duration"]
                    if total_duration > max_seconds:
                        break
                    all_entries.append(entry)
                    count += 1

                batch = []

                if idx % 2000 == 0 and idx > 0:
                    print(f"    Processed {idx:,} samples, {total_duration/3600:.1f} hours saved")

        # Process remaining batch
        if batch and total_duration < max_seconds:
            results = process_batch_threadpool(
                batch,
                config.audio_key,
                config.text_key,
                sample_rate,
                audio_dir,
                split_suffix,
                num_threads=num_workers,
            )
            for entry in results:
                if total_duration + entry["duration"] > max_seconds:
                    break
                total_duration += entry["duration"]
                all_entries.append(entry)
                count += 1

        # Write manifest (single write for speed)
        with open(manifest_path, "w") as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + "\n")

        hours = total_duration / 3600
        print(f"    Saved {count:,} samples ({hours:.1f} hours) to {manifest_path.name}")

    return manifest_paths


def prepare_dataset_worker(
    dataset_name: str,
    base_dir: str,
    sample_rate: int,
    max_hours: float | None,
    num_workers: int,
    result_queue: Queue,
) -> None:
    """Worker function for parallel dataset preparation."""
    try:
        config = DATASETS[dataset_name]
        manifests = prepare_dataset_fast(
            config,
            Path(base_dir),
            sample_rate=sample_rate,
            max_hours=max_hours,
            num_workers=num_workers,
        )
        result_queue.put((dataset_name, manifests, None))
    except Exception as e:
        result_queue.put((dataset_name, None, str(e)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast parallel ASR data preparation")
    parser.add_argument("--base_dir", type=str, default="/data/razhan/15k_hours")
    parser.add_argument("--datasets", type=str, nargs="+", choices=list(DATASETS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--15k", dest="fifteen_k", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--workers", type=int, default=min(32, cpu_count()))
    parser.add_argument("--parallel_datasets", type=int, default=2,
                        help="Number of datasets to process in parallel")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    target_hours: dict[str, float] = {}

    if args.fifteen_k:
        dataset_names = list(DATASET_TARGET_HOURS.keys())
        target_hours = DATASET_TARGET_HOURS.copy()
    elif args.all:
        dataset_names = list(DATASETS.keys())
    elif args.datasets:
        dataset_names = args.datasets
    else:
        print("Error: Specify --datasets, --all, or --15k")
        return

    if args.max_hours:
        for name in dataset_names:
            target_hours[name] = args.max_hours

    # Print plan
    print(f"\n{'='*60}")
    print(f"FAST PARALLEL DATA PREPARATION")
    print(f"{'='*60}")
    print(f"Workers per dataset: {args.workers}")
    print(f"Parallel datasets: {args.parallel_datasets}")
    print(f"Output: {base_dir}")
    print(f"{'='*60}")

    total_target = 0.0
    for name in dataset_names:
        config = DATASETS[name]
        max_h = target_hours.get(name)
        auth = " [AUTH]" if config.requires_auth else ""
        hours = max_h if max_h else config.approx_hours
        print(f"  {name}: {hours:.0f}h{auth}")
        total_target += hours
    print(f"{'='*60}")
    print(f"Total target: ~{total_target:.0f} hours")

    if args.dry_run:
        print("\n[Dry run - no data downloaded]")
        return

    print(f"\nProcessing {len(dataset_names)} datasets...")

    # Process datasets with limited parallelism to avoid memory issues
    result_queue: Queue = Queue()
    all_manifests: dict[str, dict[str, Path]] = {}

    # Process in batches of parallel_datasets
    for i in range(0, len(dataset_names), args.parallel_datasets):
        batch = dataset_names[i:i + args.parallel_datasets]
        processes = []

        for name in batch:
            print(f"\nStarting: {name}")
            p = Process(
                target=prepare_dataset_worker,
                args=(
                    name,
                    str(base_dir),
                    args.sample_rate,
                    target_hours.get(name),
                    args.workers // len(batch),  # Split workers among parallel datasets
                    result_queue,
                )
            )
            p.start()
            processes.append(p)

        # Wait for batch to complete
        for p in processes:
            p.join()

        # Collect results
        while not result_queue.empty():
            name, manifests, error = result_queue.get()
            if error:
                print(f"\nError preparing {name}: {error}")
            elif manifests:
                all_manifests[name] = manifests

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_train_hours = 0.0
    for name, manifests in all_manifests.items():
        print(f"\n{name}:")
        for split, path in manifests.items():
            if path.exists():
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

    print(f"\n{'='*60}")
    print(f"Total training data: {total_train_hours:.1f} hours")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
