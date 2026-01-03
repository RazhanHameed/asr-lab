#!/usr/bin/env python3
"""Training script for ASR Lab models.

Supports training SSM, Whisper, and Conformer models with various configurations.

Usage:
    # Train SSM model
    python scripts/train.py --model ssm --config base --dataset librispeech

    # Train Whisper model with FP8
    python scripts/train.py --model whisper --config base --precision fp8

    # Train Conformer with distributed training
    torchrun --nproc-per-node 8 scripts/train.py --model conformer --fsdp
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from asr_lab.models.ssm import SSMASRModel, SSMConfig
from asr_lab.models.whisper import WhisperASRModel, WhisperConfig
from asr_lab.models.conformer import ConformerASRModel, ConformerConfig
from asr_lab.audio.features import MelSpectrogramExtractor, FeatureConfig
from asr_lab.audio.augmentation import SpecAugment
from asr_lab.tokenizers.base import CharacterTokenizer
from asr_lab.training.dataset import ASRDataset, ASRCollator, ASRSample
from asr_lab.training.trainer import Trainer, TrainerConfig


def get_model(model_type: str, config_name: str) -> torch.nn.Module:
    """Create model from type and config name."""
    if model_type == "ssm":
        config_fn = getattr(SSMConfig, config_name, None)
        if config_fn is None:
            raise ValueError(f"Unknown SSM config: {config_name}")
        return SSMASRModel(config_fn())
    elif model_type == "whisper":
        config_fn = getattr(WhisperConfig, config_name, None)
        if config_fn is None:
            raise ValueError(f"Unknown Whisper config: {config_name}")
        return WhisperASRModel(config_fn())
    elif model_type == "conformer":
        config_fn = getattr(ConformerConfig, config_name, None)
        if config_fn is None:
            raise ValueError(f"Unknown Conformer config: {config_name}")
        return ConformerASRModel(config_fn())
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_dataset(
    dataset: str,
    split: str,
    feature_extractor: MelSpectrogramExtractor,
    tokenizer: CharacterTokenizer,
    data_dir: str | None = None,
) -> ASRDataset:
    """Load dataset by name."""
    if dataset == "librispeech":
        try:
            from datasets import load_dataset as hf_load_dataset

            ds = hf_load_dataset("librispeech_asr", split=split)
            samples = []
            for item in ds:
                samples.append(
                    ASRSample(
                        audio_path=item["audio"]["path"],
                        text=item["text"].lower(),
                        duration=len(item["audio"]["array"])
                        / item["audio"]["sampling_rate"],
                    )
                )
            return ASRDataset(samples, feature_extractor, tokenizer)
        except ImportError:
            raise ImportError(
                "datasets library required for LibriSpeech. "
                "Install with: pip install datasets"
            )

    elif dataset == "manifest":
        if data_dir is None:
            raise ValueError("--data-dir required for manifest dataset")
        return ASRDataset.from_manifest(
            Path(data_dir) / f"{split}.json",
            feature_extractor,
            tokenizer,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASR models")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="ssm",
        choices=["ssm", "whisper", "conformer"],
        help="Model architecture",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        help="Model configuration (small, base, large, a100_optimized, etc.)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "manifest"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train.clean.100",
        help="Training split",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="validation.clean",
        help="Validation split",
    )
    parser.add_argument("--data-dir", type=str, help="Data directory for manifest")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "fp8", "mxfp8", "mxfp4", "auto"],
        help="Training precision",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP")

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Create model
    print(f"Creating {args.model} model with {args.config} configuration...")
    model = get_model(args.model, args.config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create feature extractor
    feature_config = FeatureConfig(sample_rate=16000, n_mels=80)
    feature_extractor = MelSpectrogramExtractor(feature_config)

    # Create tokenizer
    tokenizer = CharacterTokenizer()

    # Create augmentation
    augment = SpecAugment()

    # Load datasets
    print(f"Loading {args.dataset} dataset...")
    train_dataset = load_dataset(
        args.dataset,
        args.train_split,
        feature_extractor,
        tokenizer,
        args.data_dir,
    )
    val_dataset = load_dataset(
        args.dataset,
        args.val_split,
        feature_extractor,
        tokenizer,
        args.data_dir,
    )

    # Create dataloaders
    collator = ASRCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
    )

    # Create trainer config
    trainer_config = TrainerConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        precision=args.precision,
        compile_model=args.compile,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_fsdp=args.fsdp,
        use_wandb=args.wandb,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("Starting training...")
    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Best eval loss: {results['best_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
