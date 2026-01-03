"""Evaluation utilities for ASR models."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr_lab.models.base import ASRModel
from asr_lab.tokenizers.base import Tokenizer
from asr_lab.training.precision import PrecisionManager, get_precision_manager
from asr_lab.utils.metrics import compute_wer, compute_cer


@dataclass
class EvaluationConfig:
    """Configuration for evaluation.

    Attributes:
        batch_size: Evaluation batch size
        precision: Precision mode
        compare_streaming: Compare streaming vs offline
        save_predictions: Save predictions to file
        output_dir: Output directory for results
    """

    batch_size: int = 32
    precision: str = "bf16"
    compare_streaming: bool = False
    save_predictions: bool = False
    output_dir: str = "results"


@dataclass
class EvaluationResult:
    """Evaluation results.

    Attributes:
        wer: Word Error Rate
        cer: Character Error Rate
        samples: Number of samples evaluated
        predictions: List of (reference, hypothesis) pairs
    """

    wer: float
    cer: float
    samples: int
    predictions: list[tuple[str, str]] | None = None


class Evaluator:
    """Evaluator for ASR models."""

    def __init__(
        self,
        model: ASRModel,
        tokenizer: Tokenizer,
        config: EvaluationConfig | None = None,
    ) -> None:
        if config is None:
            config = EvaluationConfig()

        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Setup precision
        self.precision = get_precision_manager(config.precision)

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader[dict[str, torch.Tensor]],
        streaming: bool = False,
    ) -> EvaluationResult:
        """Evaluate model on dataset.

        Args:
            dataloader: Evaluation dataloader
            streaming: Whether to use streaming mode

        Returns:
            EvaluationResult with WER, CER, and predictions
        """
        references: list[str] = []
        hypotheses: list[str] = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            features = batch["features"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            tokens = batch["tokens"]
            token_lengths = batch["token_lengths"]

            # Decode references
            for i in range(len(tokens)):
                ref_tokens = tokens[i, : token_lengths[i]].tolist()
                references.append(self.tokenizer.decode(ref_tokens))

            # Run inference
            with self.precision.autocast():
                predictions = self.model.transcribe(
                    features, feature_lengths, streaming=streaming
                )

            # Decode predictions
            for pred_tokens in predictions:
                hypotheses.append(self.tokenizer.decode(pred_tokens))

        # Compute metrics
        wer = compute_wer(references, hypotheses)
        cer = compute_cer(references, hypotheses)

        # Prepare predictions if requested
        predictions_list = None
        if self.config.save_predictions:
            predictions_list = list(zip(references, hypotheses))
            self._save_predictions(predictions_list, streaming)

        return EvaluationResult(
            wer=float(wer),
            cer=float(cer),
            samples=len(references),
            predictions=predictions_list,
        )

    def compare_streaming(
        self,
        dataloader: DataLoader[dict[str, torch.Tensor]],
    ) -> dict[str, EvaluationResult]:
        """Compare streaming vs offline performance.

        Args:
            dataloader: Evaluation dataloader

        Returns:
            Dictionary with 'offline' and 'streaming' results
        """
        offline_result = self.evaluate(dataloader, streaming=False)
        streaming_result = self.evaluate(dataloader, streaming=True)

        print("\nStreaming vs Offline Comparison:")
        print(f"{'Mode':<12} {'WER':<10} {'CER':<10}")
        print("-" * 32)
        print(f"{'Offline':<12} {offline_result.wer:.2%}    {offline_result.cer:.2%}")
        print(f"{'Streaming':<12} {streaming_result.wer:.2%}    {streaming_result.cer:.2%}")

        return {
            "offline": offline_result,
            "streaming": streaming_result,
        }

    def _save_predictions(
        self,
        predictions: list[tuple[str, str]],
        streaming: bool,
    ) -> None:
        """Save predictions to file."""
        mode = "streaming" if streaming else "offline"
        output_file = self.output_dir / f"predictions_{mode}.txt"

        with open(output_file, "w") as f:
            for ref, hyp in predictions:
                f.write(f"REF: {ref}\n")
                f.write(f"HYP: {hyp}\n")
                f.write("\n")

        print(f"Saved predictions to {output_file}")
