#!/usr/bin/env python
"""Train SentencePiece tokenizer for SSM-ASR.

This script trains a SentencePiece tokenizer on transcripts from ASR datasets,
matching the ABR model's 256 vocabulary size configuration.

Usage:
    python scripts/train_tokenizer.py \
        --manifests data/manifests/*.json \
        --output tokenizers/spm_256 \
        --vocab_size 256
"""

import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm
from tqdm import tqdm


def extract_texts_from_manifests(
    manifest_paths: list[Path],
    output_path: Path,
    lowercase: bool = True,
    max_samples: int | None = None,
) -> int:
    """Extract text transcripts from manifest files.

    Args:
        manifest_paths: List of manifest file paths
        output_path: Path to write extracted texts
        lowercase: Whether to lowercase text
        max_samples: Maximum number of samples to extract

    Returns:
        Number of texts extracted
    """
    count = 0
    seen_texts: set[str] = set()

    with open(output_path, "w") as out_f:
        for manifest_path in manifest_paths:
            if not manifest_path.exists():
                print(f"Warning: Manifest not found: {manifest_path}")
                continue

            print(f"Processing: {manifest_path}")

            with open(manifest_path) as f:
                for line in tqdm(f, desc=f"  Reading {manifest_path.name}"):
                    if max_samples and count >= max_samples:
                        break

                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "").strip()

                        if not text:
                            continue

                        if lowercase:
                            text = text.lower()

                        # Remove duplicates
                        if text in seen_texts:
                            continue
                        seen_texts.add(text)

                        out_f.write(text + "\n")
                        count += 1

                    except json.JSONDecodeError:
                        continue

            if max_samples and count >= max_samples:
                break

    return count


def train_sentencepiece(
    input_path: Path,
    model_prefix: str,
    vocab_size: int = 256,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    num_threads: int = 16,
) -> Path:
    """Train SentencePiece tokenizer.

    Args:
        input_path: Path to training text file
        model_prefix: Prefix for output model files
        vocab_size: Target vocabulary size
        model_type: Model type (bpe, unigram, char, word)
        character_coverage: Character coverage for training
        num_threads: Number of threads for training

    Returns:
        Path to trained model file
    """
    # Create output directory
    output_dir = Path(model_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Training parameters
        split_by_whitespace=True,
        split_by_number=True,
        split_digits=True,
        treat_whitespace_as_suffix=False,
        allow_whitespace_only_pieces=False,
        # Normalization
        normalization_rule_name="identity",  # No normalization
        remove_extra_whitespaces=True,
        add_dummy_prefix=False,
        # Limit memory usage
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
    )

    model_path = Path(f"{model_prefix}.model")
    print(f"Trained tokenizer saved to: {model_path}")

    return model_path


def verify_tokenizer(model_path: Path, test_texts: list[str]) -> None:
    """Verify tokenizer works correctly.

    Args:
        model_path: Path to SentencePiece model
        test_texts: List of test texts
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))

    print(f"\nTokenizer verification:")
    print(f"  Vocabulary size: {sp.GetPieceSize()}")
    print()

    for text in test_texts:
        tokens = sp.EncodeAsIds(text)
        pieces = sp.EncodeAsPieces(text)
        decoded = sp.DecodeIds(tokens)

        print(f"  Input:    '{text}'")
        print(f"  Tokens:   {tokens}")
        print(f"  Pieces:   {pieces}")
        print(f"  Decoded:  '{decoded}'")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for ASR")
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        required=True,
        help="Manifest files to extract texts from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizers/spm_256",
        help="Output model prefix",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=256,
        help="Vocabulary size (default: 256)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="Model type",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples for training",
    )
    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="Don't lowercase text",
    )
    args = parser.parse_args()

    # Resolve manifest paths
    manifest_paths = []
    for pattern in args.manifests:
        if "*" in pattern:
            manifest_paths.extend(Path(".").glob(pattern))
        else:
            manifest_paths.append(Path(pattern))

    if not manifest_paths:
        print("Error: No manifest files found")
        return

    print(f"Found {len(manifest_paths)} manifest files")

    # Extract texts to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)

    print(f"\nExtracting texts to: {tmp_path}")
    num_texts = extract_texts_from_manifests(
        manifest_paths,
        tmp_path,
        lowercase=not args.no_lowercase,
        max_samples=args.max_samples,
    )
    print(f"Extracted {num_texts:,} unique texts")

    # Train tokenizer
    print(f"\nTraining {args.model_type} tokenizer with vocab_size={args.vocab_size}")
    model_path = train_sentencepiece(
        tmp_path,
        args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )

    # Verify
    test_texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "automatic speech recognition",
        "i'm going to the store",
        "what time is it",
    ]
    verify_tokenizer(model_path, test_texts)

    # Cleanup
    tmp_path.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
