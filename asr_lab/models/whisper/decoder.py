"""Whisper-style Transformer decoder with cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_lab.models.base import Decoder
from asr_lab.models.whisper.config import WhisperConfig
from asr_lab.models.whisper.encoder import MultiHeadAttention


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings."""

    def __init__(self, max_positions: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_positions, d_model)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(offset, offset + x.size(1), device=x.device)
        return x + self.embedding(positions)


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_flash: bool = True,
    ) -> None:
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout, use_flash, is_cross_attention=False
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout, use_flash=False, is_cross_attention=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention (causal)
        x = x + self.self_attn(self.norm1(x), is_causal=True)

        # Cross-attention
        x = x + self.cross_attn(
            self.norm2(x),
            encoder_output=encoder_output,
            mask=cross_attn_mask,
        )

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


class WhisperDecoder(Decoder):
    """Whisper-style Transformer decoder with cross-attention.

    This decoder implements:
    - Token embeddings with learned positional encoding
    - Causal self-attention
    - Cross-attention to encoder outputs
    - Autoregressive generation
    """

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__(config)
        self.whisper_config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = LearnedPositionalEmbedding(
            config.max_target_positions, config.d_model
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.d_model,
                config.n_heads,
                config.decoder_ffn_dim,
                config.dropout,
                config.use_flash_attention,
            )
            for _ in range(config.n_decoder_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_embeddings:
            self.output_proj.weight = self.embed_tokens.weight

        # Special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training or teacher-forced decoding."""
        if targets is None:
            # Inference: return logits for greedy decoding
            return {"encoder_output": encoder_output}

        # Embed target tokens
        x = self.embed_tokens(targets)
        x = self.pos_embed(x)

        # Create cross-attention mask from encoder lengths
        cross_attn_mask = None
        if encoder_lengths is not None:
            max_len = encoder_output.size(1)
            cross_attn_mask = (
                torch.arange(max_len, device=encoder_output.device).unsqueeze(0)
                < encoder_lengths.unsqueeze(1)
            )

        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, cross_attn_mask=cross_attn_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)

        result: dict[str, torch.Tensor] = {"logits": logits}

        # Compute loss if targets provided
        if target_lengths is not None:
            # Shift for autoregressive loss: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            # Create mask for padding
            max_len = shift_targets.size(1)
            target_mask = (
                torch.arange(max_len, device=targets.device).unsqueeze(0)
                < (target_lengths - 1).unsqueeze(1)
            )

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_targets.view(-1),
                ignore_index=self.pad_id,
                reduction="none",
            )
            loss = (loss.view_as(shift_targets) * target_mask).sum() / target_mask.sum()
            result["loss"] = loss

        return result

    def decode(
        self,
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor | None = None,
        max_length: int = 200,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # Start with BOS token
        tokens = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=device
        )

        # Create cross-attention mask
        cross_attn_mask = None
        if encoder_lengths is not None:
            max_len = encoder_output.size(1)
            cross_attn_mask = (
                torch.arange(max_len, device=device).unsqueeze(0)
                < encoder_lengths.unsqueeze(1)
            )

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Embed current tokens
            x = self.embed_tokens(tokens)
            x = self.pos_embed(x)

            # Process through decoder
            for layer in self.layers:
                x = layer(x, encoder_output, cross_attn_mask=cross_attn_mask)

            x = self.final_norm(x)
            logits = self.output_proj(x[:, -1])  # Only last position

            # Greedy selection
            next_tokens = logits.argmax(dim=-1, keepdim=True)

            # Check for EOS
            done = done | (next_tokens.squeeze(-1) == self.eos_id)

            # Append next tokens
            tokens = torch.cat([tokens, next_tokens], dim=1)

            if done.all():
                break

        # Convert to list of lists, removing BOS and EOS
        decoded: list[list[int]] = []
        for i in range(batch_size):
            seq = tokens[i].tolist()
            # Remove BOS
            if seq[0] == self.bos_id:
                seq = seq[1:]
            # Remove EOS and everything after
            if self.eos_id in seq:
                seq = seq[: seq.index(self.eos_id)]
            decoded.append(seq)

        return decoded
