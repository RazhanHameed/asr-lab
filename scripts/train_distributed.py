#!/usr/bin/env python
"""Distributed training script for SSM-ASR models.

This script supports training on multiple GPUs using PyTorch DistributedDataParallel.

Usage:
    # Single node, 2 GPUs
    torchrun --nproc_per_node=2 scripts/train_distributed.py --config configs/ssm_19m_a100.yaml

    # With specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_distributed.py --config configs/ssm_19m_a100.yaml
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

# Suppress CTC backward determinism warning (known issue)
warnings.filterwarnings("ignore", message=".*ctc_loss_backward_gpu.*")
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import yaml

from asr_lab.audio.augmentation import SpecAugment
from asr_lab.audio.features import FeatureConfig, MelSpectrogramExtractor
from asr_lab.data.datasets import ASRDatasetConfig, CombinedASRDataset, MapStyleASRDataset
from asr_lab.models.ssm import SSMASRModel, SSMConfig
from asr_lab.reproducibility import enable_deterministic_mode, set_seed, capture_environment, save_environment
from asr_lab.tokenizers.base import BPETokenizer
from asr_lab.training.dataset import ASRCollator
from asr_lab.training.precision import PrecisionMode, get_precision_manager


@dataclass
class TrainingState:
    """Tracks training progress."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_wer: float = float("inf")


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training.

    Returns:
        rank: Global rank of current process
        local_rank: Local rank (GPU index on this node)
        world_size: Total number of processes
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> SSMASRModel:
    """Create SSM-ASR model from config."""
    model_cfg = config["model"]

    ssm_config = SSMConfig(
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 18),
        d_state=model_cfg.get("d_state", 64),
        d_conv=model_cfg.get("d_conv", 4),
        expand=model_cfg.get("expand", 2),
        headdim=model_cfg.get("headdim", 64),
        attention_every_n_layers=model_cfg.get("attention_every_n_layers", 4),
        bidirectional=model_cfg.get("bidirectional", True),
        chunk_size=model_cfg.get("chunk_size", 256),
        vocab_size=model_cfg.get("vocab_size", 256),
        n_mels=model_cfg.get("n_mels", 80),
        dropout=model_cfg.get("dropout", 0.1),
        conv_channels=model_cfg.get("conv_channels", [256, 256]),
        conv_kernel_sizes=model_cfg.get("conv_kernel_sizes", [3, 3]),
        conv_strides=model_cfg.get("conv_strides", [2, 2]),
    )

    return SSMASRModel(ssm_config)


def create_feature_extractor(config: dict) -> MelSpectrogramExtractor:
    """Create feature extractor from config."""
    feat_cfg = config.get("features", {})

    feature_config = FeatureConfig(
        sample_rate=feat_cfg.get("sample_rate", 16000),
        n_mels=feat_cfg.get("n_mels", 80),
        n_fft=feat_cfg.get("n_fft", 400),
        hop_length=feat_cfg.get("hop_length", 160),
        win_length=feat_cfg.get("win_length", 400),
        f_min=feat_cfg.get("f_min", 0.0),
        f_max=feat_cfg.get("f_max", 8000.0),
        normalize=feat_cfg.get("normalize", True),
        log_mel=feat_cfg.get("log_mel", True),
    )

    return MelSpectrogramExtractor(feature_config)


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer with weight decay separation."""
    train_cfg = config["training"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_cfg.get("weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=train_cfg.get("learning_rate", 5e-4),
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler."""
    train_cfg = config["training"]
    warmup_steps = train_cfg.get("warmup_steps", 10000)
    min_lr = train_cfg.get("min_learning_rate", 1e-6)
    max_lr = train_cfg.get("learning_rate", 5e-4)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return max(min_lr / max_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: DDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    precision_manager: object,
    config: dict,
    state: TrainingState,
    rank: int,
    world_size: int = 1,
) -> float:
    """Train for one epoch."""
    model.train()
    train_cfg = config["training"]
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    logging_steps = train_cfg.get("logging_steps", 100)

    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0

    device = next(model.parameters()).device

    # IterableDataset doesn't have a length, so we track iterations manually
    max_steps_per_epoch = 28539 // (train_cfg.get("batch_size", 8) * world_size)  # Approx steps

    if rank == 0:
        print(f"  Max steps per epoch: {max_steps_per_epoch}")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= max_steps_per_epoch:
            break

        if batch_idx % 100 == 0 and rank == 0:
            print(f"  Step {batch_idx}/{max_steps_per_epoch}")

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with autocast
        with precision_manager.autocast():
            outputs = model(
                features=batch["features"],
                feature_lengths=batch["feature_lengths"],
                targets=batch["tokens"],
                target_lengths=batch["token_lengths"],
            )
            loss = outputs["loss"] / grad_accum_steps

        # Backward pass
        precision_manager.backward(loss)
        accumulated_loss += loss.item() * grad_accum_steps

        # Optimizer step
        if (batch_idx + 1) % grad_accum_steps == 0:
            precision_manager.optimizer_step(
                optimizer,
                clip_grad=max_grad_norm,
                model=model,
            )
            optimizer.zero_grad()
            scheduler.step()

            total_loss += accumulated_loss
            num_batches += 1
            state.global_step += 1

            # Logging
            if state.global_step % logging_steps == 0 and rank == 0:
                avg_loss = accumulated_loss
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {state.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")

            accumulated_loss = 0.0

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: DDP,
    val_loader: DataLoader,
    precision_manager: object,
    rank: int,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Evaluating", disable=rank != 0):
        batch = {k: v.to(device) for k, v in batch.items()}

        with precision_manager.autocast():
            outputs = model(
                features=batch["features"],
                feature_lengths=batch["feature_lengths"],
                targets=batch["tokens"],
                target_lengths=batch["token_lengths"],
            )
            total_loss += outputs["loss"].item()
            num_batches += 1

    # Reduce across processes
    loss_tensor = torch.tensor([total_loss, num_batches], device=device)
    dist.all_reduce(loss_tensor)
    total_loss, num_batches = loss_tensor.tolist()

    return {"val_loss": total_loss / max(num_batches, 1)}


def save_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    config: dict,
    output_dir: Path,
    name: str,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "state": {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_val_loss": state.best_val_loss,
            "best_wer": state.best_wer,
        },
        "config": config,
    }
    checkpoint_path = output_dir / f"checkpoint_{name}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    device: torch.device,
) -> None:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    saved_state = checkpoint.get("state", {})
    state.epoch = saved_state.get("epoch", 0)
    state.global_step = saved_state.get("global_step", 0)
    state.best_val_loss = saved_state.get("best_val_loss", float("inf"))
    state.best_wer = saved_state.get("best_wer", float("inf"))

    print(f"Loaded checkpoint from step {state.global_step}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed SSM-ASR Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]

    # Setup reproducibility
    seed = train_cfg.get("seed", 42)
    if train_cfg.get("deterministic", True):
        # Use warn_only=True because CTC loss backward doesn't have deterministic implementation
        enable_deterministic_mode(warn_only=True)
    set_seed(seed + rank)  # Different seed per rank for data augmentation

    # Create output directory
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config and environment
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        env = capture_environment()
        save_environment(env, output_dir / "environment.json")

    dist.barrier()

    # Create model
    model = create_model(config)
    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")

    model = model.to(device)

    # Compile model if requested
    if train_cfg.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=config.get("distributed", {}).get("find_unused_parameters", False),
        gradient_as_bucket_view=config.get("distributed", {}).get("gradient_as_bucket_view", True),
        static_graph=config.get("distributed", {}).get("static_graph", True),
    )

    # Create feature extractor
    feature_extractor = create_feature_extractor(config)

    # Create tokenizer
    tokenizer_cfg = config.get("tokenizer", {})
    tokenizer_path = tokenizer_cfg.get("model_path", "tokenizers/spm_256.model")
    if not Path(tokenizer_path).exists():
        if rank == 0:
            print(f"Warning: Tokenizer not found at {tokenizer_path}")
            print("You need to train a tokenizer first. See scripts/train_tokenizer.py")
        # Use character tokenizer as fallback
        from asr_lab.tokenizers.base import CharacterTokenizer
        tokenizer = CharacterTokenizer()
    else:
        tokenizer = BPETokenizer(tokenizer_path)

    # Create datasets
    data_cfg = config.get("data", {})

    # Training dataset (combined from multiple manifests)
    train_manifest_paths = data_cfg.get("train_manifests", [])
    if train_manifest_paths:
        train_config = ASRDatasetConfig(
            manifest_paths=train_manifest_paths,
            sample_rate=config.get("features", {}).get("sample_rate", 16000),
            max_duration=data_cfg.get("max_duration", 30.0),
            min_duration=data_cfg.get("min_duration", 0.5),
            max_text_len=data_cfg.get("max_text_length", 512),
            shuffle=True,
            seed=seed + rank,
        )

        # Augmentation
        augment_fn = None
        aug_cfg = config.get("augmentation", {}).get("spec_augment", {})
        if aug_cfg.get("enabled", False):
            spec_augment = SpecAugment(
                freq_mask_param=aug_cfg.get("freq_mask_param", 27),
                time_mask_param=aug_cfg.get("time_mask_param", 100),
                num_freq_masks=aug_cfg.get("num_freq_masks", 2),
                num_time_masks=aug_cfg.get("num_time_masks", 10),
            )
            augment_fn = spec_augment

        train_dataset = CombinedASRDataset(
            train_config,
            feature_extractor,
            tokenizer,
            dataset_weights=data_cfg.get("dataset_weights"),
            augment_fn=augment_fn,
        )
    else:
        raise ValueError("No training manifests specified in config")

    # Validation dataset
    val_manifest_paths = data_cfg.get("val_manifests", [])
    val_dataset = None
    if val_manifest_paths:
        val_dataset = MapStyleASRDataset(
            val_manifest_paths[0],
            feature_extractor,
            tokenizer,
            max_duration=data_cfg.get("max_duration", 30.0),
        )

    # Create data loaders
    collator = ASRCollator()

    # Note: IterableDataset with num_workers can cause issues
    # Using num_workers=0 for simplicity; for large scale, use MapStyleASRDataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 32),
            sampler=val_sampler,
            num_workers=4,
            collate_fn=collator,
            pin_memory=True,
        )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)

    # Estimate total steps
    epochs = train_cfg.get("epochs", 100)
    steps_per_epoch = 50000 // (train_cfg.get("batch_size", 32) * world_size)  # Rough estimate
    total_steps = epochs * steps_per_epoch

    scheduler = create_scheduler(optimizer, config, total_steps)

    # Setup precision
    precision_mode = train_cfg.get("precision", "bf16")
    precision_manager = get_precision_manager(precision_mode)

    # Training state
    state = TrainingState()

    # Resume from checkpoint
    if args.resume:
        load_checkpoint(Path(args.resume), model, optimizer, scheduler, state, device)

    # Training loop
    if rank == 0:
        print(f"\nStarting training:")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {train_cfg.get('batch_size', 32)}")
        print(f"  Gradient accumulation: {train_cfg.get('gradient_accumulation_steps', 1)}")
        effective_batch = train_cfg.get("batch_size", 32) * train_cfg.get("gradient_accumulation_steps", 1) * world_size
        print(f"  Effective batch size: {effective_batch}")
        print(f"  Precision: {precision_mode}")
        print(f"  Epochs: {epochs}")
        print()

    eval_steps = train_cfg.get("eval_steps", 2500)
    save_steps = train_cfg.get("save_steps", 5000)

    for epoch in range(state.epoch, epochs):
        state.epoch = epoch

        # Train epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            precision_manager, config, state, rank, world_size
        )

        if rank == 0:
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

        # Evaluate
        if val_loader and (epoch + 1) % 1 == 0:
            metrics = evaluate(model, val_loader, precision_manager, rank)
            if rank == 0:
                print(f"Epoch {epoch + 1} - Val Loss: {metrics['val_loss']:.4f}")

                if metrics["val_loss"] < state.best_val_loss:
                    state.best_val_loss = metrics["val_loss"]
                    save_checkpoint(model, optimizer, scheduler, state, config, output_dir, "best")

        # Save checkpoint
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, state, config, output_dir, f"epoch_{epoch + 1}")

        dist.barrier()

    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, state, config, output_dir, "final")
        print("\nTraining complete!")
        print(f"Best validation loss: {state.best_val_loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
