#!/usr/bin/env python3
"""Training script for ASR models using YODAS-Granary streaming dataset.

Supports streaming from HuggingFace with duration-limited epochs.

Usage:
    # Single GPU training with streaming
    python scripts/train_yodas.py --config configs/ssm_19m_yodas.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 scripts/train_yodas.py --config configs/ssm_19m_yodas.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Configure HuggingFace timeouts
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asr_lab.models.ssm import SSMASRModel, SSMConfig
from asr_lab.audio.features import MelSpectrogramExtractor, FeatureConfig
from asr_lab.tokenizers.base import BPETokenizer
from asr_lab.data.streaming import StreamingASRDataset, StreamingASRCollator
from asr_lab.training.precision import get_precision_manager


def setup_distributed() -> tuple[int, int, bool]:
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    precision_manager,
    device: torch.device,
    epoch: int,
    rank: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=rank != 0)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        token_lengths = batch["token_lengths"].to(device)

        # Forward pass with mixed precision
        with precision_manager.autocast():
            output = model(
                features,
                feature_lengths,
                targets=tokens,
                target_lengths=token_lengths,
            )
            loss = output["loss"] / gradient_accumulation_steps

        # Backward pass
        precision_manager.backward(loss)

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Optimizer step with gradient clipping
            precision_manager.optimizer_step(
                optimizer,
                clip_grad=max_grad_norm if max_grad_norm > 0 else None,
                model=model
            )
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if rank == 0 and batch_idx % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return {"train_loss": avg_loss, "num_batches": num_batches}


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: BPETokenizer,
    precision_manager,
    device: torch.device,
    rank: int = 0,
    max_samples: int = 500,
) -> dict:
    """Run validation, compute WER/CER, and print random samples."""
    import random
    from asr_lab.utils.metrics import compute_wer, compute_cer

    model.eval()
    total_loss = 0.0
    num_batches = 0
    references: list[str] = []
    hypotheses: list[str] = []

    # Get the underlying model for transcription
    transcribe_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch["features"].size(0) >= max_samples:
                break

            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            tokens = batch["tokens"].to(device)
            token_lengths = batch["token_lengths"].to(device)

            with precision_manager.autocast():
                output = model(
                    features,
                    feature_lengths,
                    targets=tokens,
                    target_lengths=token_lengths,
                )
                loss = output["loss"]

            total_loss += loss.item()
            num_batches += 1

            # Decode for WER
            predictions = transcribe_model.transcribe(features, feature_lengths)
            for i in range(len(tokens)):
                ref_tokens = tokens[i, :token_lengths[i]].tolist()
                references.append(tokenizer.decode(ref_tokens))
                hypotheses.append(tokenizer.decode(predictions[i]))

    avg_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)

    # Print 5 random samples on rank 0 (truncated to 5 words)
    if rank == 0 and len(references) >= 5:
        indices = random.sample(range(len(references)), 5)
        print("\n" + "=" * 60)
        print("Random Evaluation Samples (5 words max):")
        print("=" * 60)
        for idx in indices:
            ref_words = references[idx].split()[:5]
            hyp_words = hypotheses[idx].split()[:5]
            ref_trunc = " ".join(ref_words) + ("..." if len(references[idx].split()) > 5 else "")
            hyp_trunc = " ".join(hyp_words) + ("..." if len(hypotheses[idx].split()) > 5 else "")
            print(f"[{idx}] REF: {ref_trunc}")
            print(f"    HYP: {hyp_trunc}")
        print("=" * 60 + "\n")

    return {"val_loss": avg_loss, "wer": wer, "cer": cer, "num_samples": len(references)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASR on YODAS-Granary")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup distributed
    rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Training with {world_size} GPU(s)")
        print(f"Config: {args.config}")

    # Create feature extractor
    feature_config = FeatureConfig(
        sample_rate=config["features"]["sample_rate"],
        n_mels=config["features"]["n_mels"],
        n_fft=config["features"].get("n_fft", 400),
        hop_length=config["features"].get("hop_length", 160),
    )
    feature_extractor = MelSpectrogramExtractor(feature_config)

    # Create tokenizer
    tokenizer = BPETokenizer(config["tokenizer"]["model_path"], add_blank=True)

    # Create model
    model_config = SSMConfig.from_dict(config["model"])
    model = SSMASRModel(model_config)
    model = model.to(device)

    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # Create streaming dataset
    train_dataset = StreamingASRDataset(
        hf_dataset_name=config["data"]["dataset_name"],
        hf_subset=config["data"]["subset"],
        hf_split=config["data"]["split"],
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_epoch_hours=config["data"]["max_epoch_hours"],
        max_audio_duration=config["data"].get("max_audio_duration", 30.0),
        min_audio_duration=config["data"].get("min_audio_duration", 0.5),
        shuffle_buffer_size=config["data"].get("shuffle_buffer_size", 10000),
        seed=config["training"].get("seed", 42),
    )

    # Create dataloader
    collator = StreamingASRCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collator,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # Create scheduler
    total_steps = config["training"].get("total_steps", 100000)
    warmup_steps = config["training"].get("warmup_steps", 1000)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # Create precision manager
    precision_manager = get_precision_manager(config["training"].get("precision", "bf16"))

    # Resume if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if is_distributed:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    # Output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    if rank == 0:
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    # Training loop
    best_wer = float("inf")
    num_epochs = config["training"]["num_epochs"]

    for epoch in range(start_epoch, num_epochs):
        # Set epoch for shuffling
        train_dataset.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            precision_manager,
            device,
            epoch,
            rank,
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
            max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        )

        if rank == 0:
            print(f"Epoch {epoch}: Train Loss = {train_metrics['train_loss']:.4f}, "
                  f"Batches = {train_metrics['num_batches']}")

        # Validation
        if rank == 0:
            print("Running validation...")
        val_metrics = validate(
            model,
            train_loader,
            tokenizer,
            precision_manager,
            device,
            rank=rank,
            max_samples=config["training"].get("val_samples", 500),
        )

        if rank == 0:
            print(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}, "
                  f"WER = {val_metrics['wer']:.2f}%, CER = {val_metrics['cer']:.2f}%")

        # Save checkpoint
        if rank == 0:
            model_to_save = model.module if is_distributed else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            # Save periodic checkpoint
            if (epoch + 1) % config["training"].get("save_every", 1) == 0:
                torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")

            # Save latest
            torch.save(checkpoint, output_dir / "checkpoint_latest.pt")

            # Save best model based on WER
            if val_metrics["wer"] < best_wer:
                best_wer = val_metrics["wer"]
                torch.save(checkpoint, output_dir / "checkpoint_best.pt")
                print(f"New best model saved! WER: {best_wer:.2f}%")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
