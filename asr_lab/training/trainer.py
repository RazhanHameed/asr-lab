"""Training utilities for ASR models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr_lab.models.base import ASRModel
from asr_lab.training.precision import PrecisionManager, PrecisionMode, get_precision_manager


@dataclass
class TrainerConfig:
    """Configuration for ASR trainer.

    Attributes:
        output_dir: Directory for checkpoints and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps
        max_grad_norm: Gradient clipping norm
        precision: Precision mode (fp32, fp16, bf16, fp8, mxfp8)
        compile_model: Whether to use torch.compile
        gradient_accumulation_steps: Gradient accumulation steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        use_fsdp: Use Fully Sharded Data Parallel
        use_wandb: Use Weights & Biases logging
    """

    output_dir: str = "outputs"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    compile_model: bool = False
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    use_fsdp: bool = False
    use_wandb: bool = False
    wandb_project: str = "asr-lab"
    seed: int = 42


class Trainer:
    """Trainer for ASR models with distributed training support."""

    def __init__(
        self,
        model: ASRModel,
        config: TrainerConfig,
        train_dataloader: DataLoader[dict[str, torch.Tensor]],
        eval_dataloader: DataLoader[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Set seed
        torch.manual_seed(config.seed)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup precision
        self.precision = get_precision_manager(config.precision)
        self.model = self.precision.prepare_model(self.model)

        # Compile model if requested
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Logging
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate params with/without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if available
        use_fused = torch.cuda.is_available() and hasattr(
            torch.optim.AdamW, "fused"
        )

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            fused=use_fused if use_fused else False,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup."""
        total_steps = len(self.train_dataloader) * self.config.epochs
        warmup_steps = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                config={
                    "model_params": self.model.count_parameters(),
                    **self.config.__dict__,
                },
            )
        except ImportError:
            print("Warning: wandb not available, disabling logging")

    def train(self) -> dict[str, float]:
        """Run training loop.

        Returns:
            Dictionary with final metrics
        """
        self.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            )

            for batch_idx, batch in enumerate(progress):
                loss = self._train_step(batch)
                epoch_loss += loss
                num_batches += 1

                # Update progress bar
                progress.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log({"train/loss": loss, "train/lr": self.scheduler.get_last_lr()[0]})

                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_loss = self.evaluate()
                    self._log({"eval/loss": eval_loss})

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best")

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                self.global_step += 1

            # End of epoch
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")

        # Save final checkpoint
        self.save_checkpoint("final")

        return {"final_loss": avg_loss, "best_eval_loss": self.best_eval_loss}

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with autocast
        with self.precision.autocast():
            outputs = self.model(
                features=batch["features"],
                feature_lengths=batch["feature_lengths"],
                targets=batch["tokens"],
                target_lengths=batch["token_lengths"],
            )
            loss = outputs["loss"]

            # Scale for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.precision.backward(loss)

        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.precision.optimizer_step(
                self.optimizer,
                clip_grad=self.config.max_grad_norm,
                model=self.model,
            )
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation.

        Returns:
            Average evaluation loss
        """
        if self.eval_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with self.precision.autocast():
                outputs = self.model(
                    features=batch["features"],
                    feature_lengths=batch["feature_lengths"],
                    targets=batch["tokens"],
                    target_lengths=batch["token_lengths"],
                )
                total_loss += outputs["loss"].item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.model.config.to_dict(),
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

    def _log(self, metrics: dict[str, float]) -> None:
        """Log metrics."""
        if self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=self.global_step)
