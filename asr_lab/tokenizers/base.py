"""Base tokenizer classes for ASR."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...

    @property
    @abstractmethod
    def pad_id(self) -> int:
        """Get padding token ID."""
        ...

    @property
    @abstractmethod
    def blank_id(self) -> int:
        """Get CTC blank token ID."""
        ...

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode batch of texts with padding.

        Args:
            texts: List of text strings

        Returns:
            tokens: Padded token tensor of shape (batch, max_len)
            lengths: Token lengths of shape (batch,)
        """
        encoded = [self.encode(text) for text in texts]
        lengths = torch.tensor([len(t) for t in encoded])
        max_len = int(lengths.max().item())

        tokens = torch.full((len(texts), max_len), self.pad_id, dtype=torch.long)
        for i, t in enumerate(encoded):
            tokens[i, : len(t)] = torch.tensor(t)

        return tokens, lengths

    def decode_batch(self, tokens: torch.Tensor) -> list[str]:
        """Decode batch of tokens.

        Args:
            tokens: Token tensor of shape (batch, seq_len)

        Returns:
            List of decoded text strings
        """
        return [self.decode(t.tolist()) for t in tokens]


class CharacterTokenizer(Tokenizer):
    """Simple character-level tokenizer.

    Supports ASCII letters, digits, and common punctuation.
    Special tokens: <blank> (0), <pad> (1), <unk> (2), <space> (3)
    """

    def __init__(
        self,
        chars: str | None = None,
        add_blank: bool = True,
    ) -> None:
        # Default character set
        if chars is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789.,!?'-"

        # Build vocabulary
        self._blank_id = 0
        self._pad_id = 1
        self._unk_id = 2
        self._space_id = 3

        self._id_to_char: dict[int, str] = {
            0: "<blank>",
            1: "<pad>",
            2: "<unk>",
            3: " ",
        }
        self._char_to_id: dict[str, int] = {
            "<blank>": 0,
            "<pad>": 1,
            "<unk>": 2,
            " ": 3,
        }

        for i, c in enumerate(chars, start=4):
            self._id_to_char[i] = c
            self._char_to_id[c] = i

        self._vocab_size = len(self._id_to_char)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        text = text.lower()
        return [self._char_to_id.get(c, self._unk_id) for c in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for t in tokens:
            if t in (self._blank_id, self._pad_id):
                continue
            char = self._id_to_char.get(t, "")
            if char not in ("<blank>", "<pad>", "<unk>"):
                chars.append(char)
        return "".join(chars)

    def decode_ctc(self, tokens: list[int]) -> str:
        """Decode CTC output with blank removal and deduplication."""
        chars = []
        prev_token = self._blank_id
        for t in tokens:
            if t != self._blank_id and t != prev_token:
                char = self._id_to_char.get(t, "")
                if char not in ("<blank>", "<pad>", "<unk>"):
                    chars.append(char)
            prev_token = t
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def blank_id(self) -> int:
        return self._blank_id


class BPETokenizer(Tokenizer):
    """Byte-pair encoding tokenizer using SentencePiece.

    This tokenizer provides subword tokenization which is more efficient
    than character-level tokenization for many languages.
    """

    def __init__(
        self,
        model_path: str | Path,
        add_blank: bool = True,
    ) -> None:
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "sentencepiece is required for BPETokenizer. "
                "Install with: pip install sentencepiece"
            ) from e

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

        self._add_blank = add_blank
        # SentencePiece reserves ID 0 for <unk>, 1 for <s>, 2 for </s>
        # We'll use ID 0 as blank if add_blank=True (shift all IDs by 1)
        self._blank_id = 0 if add_blank else -1
        self._pad_id = 1 if add_blank else 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ids = self.sp.EncodeAsIds(text)
        if self._add_blank:
            # Shift IDs to make room for blank at 0
            ids = [i + 1 for i in ids]
        return ids

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        if self._add_blank:
            # Remove blank tokens and shift back
            tokens = [t - 1 for t in tokens if t > 0]
        return self.sp.DecodeIds(tokens)

    @property
    def vocab_size(self) -> int:
        size = self.sp.GetPieceSize()
        if self._add_blank:
            size += 1  # For blank token
        return size

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def blank_id(self) -> int:
        return self._blank_id

    @classmethod
    def train(
        cls,
        input_file: str | Path,
        model_prefix: str,
        vocab_size: int = 256,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> "BPETokenizer":
        """Train a new BPE tokenizer.

        Args:
            input_file: Path to training text file
            model_prefix: Prefix for output model files
            vocab_size: Target vocabulary size
            model_type: Model type (bpe, unigram, char, word)
            character_coverage: Character coverage for training

        Returns:
            Trained BPETokenizer
        """
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "sentencepiece is required for BPETokenizer. "
                "Install with: pip install sentencepiece"
            ) from e

        spm.SentencePieceTrainer.Train(
            input=str(input_file),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        return cls(f"{model_prefix}.model")
