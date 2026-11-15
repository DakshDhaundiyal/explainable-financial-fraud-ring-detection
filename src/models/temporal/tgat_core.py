"""Core TGAT components for leakage-safe temporal transaction modeling."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


VALID_TIME_ENCODING_MODES = ("raw_seconds", "log1p_seconds", "hours", "days")


class TimeEncoder(nn.Module):
    """Learnable cosine temporal encoding for non-negative time gaps."""

    def __init__(
        self,
        dimension: int,
        max_period: float = 86_400.0,
        time_encoding_mode: str = "raw_seconds",
    ) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if max_period <= 0:
            raise ValueError("max_period must be positive")
        if time_encoding_mode not in VALID_TIME_ENCODING_MODES:
            raise ValueError(
                f"Unsupported time_encoding_mode '{time_encoding_mode}'. "
                f"Expected one of {VALID_TIME_ENCODING_MODES}."
            )
        self.dimension = dimension
        self.max_period = float(max_period)
        self.time_encoding_mode = str(time_encoding_mode)
        frequencies = torch.logspace(
            math.log10(1.0 / self.max_period),
            math.log10(1.0),
            dimension,
        )
        self.frequency = nn.Parameter(frequencies)
        self.phase = nn.Parameter(torch.zeros(dimension))

    def scale_delta(self, delta_seconds: Tensor) -> Tensor:
        """Transform raw non-negative seconds according to time_encoding_mode."""
        delta = torch.nan_to_num(
            delta_seconds.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        delta = torch.clamp(delta, min=0.0)
        if self.time_encoding_mode == "raw_seconds":
            return delta
        if self.time_encoding_mode == "log1p_seconds":
            return torch.log1p(delta)
        if self.time_encoding_mode == "hours":
            return delta / 3600.0
        if self.time_encoding_mode == "days":
            return delta / 86_400.0
        raise ValueError(f"Unsupported time_encoding_mode: {self.time_encoding_mode}")

    def forward(self, delta_seconds: Tensor) -> Tensor:
        delta = self.scale_delta(delta_seconds)
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        return torch.cos(delta * self.frequency + self.phase)


class TemporalAttentionLayer(nn.Module):
    """Multi-head attention over strictly historical temporal neighbors."""

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        num_heads: int = 2,
        dropout: float = 0.2,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_time_encoding = use_time_encoding
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim + time_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim + time_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_state: Tensor,
        neighbor_states: Tensor,
        delta_seconds: Tensor,
        time_encoder: TimeEncoder,
        return_attention: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if neighbor_states.dim() != 3:
            raise ValueError("neighbor_states must have shape [B, K, H]")
        if delta_seconds.shape[:2] != neighbor_states.shape[:2]:
            raise ValueError("delta_seconds shape must match [B, K]")

        batch_size, neighbor_count, _ = neighbor_states.shape
        valid = delta_seconds > 0

        if self.use_time_encoding:
            time_features = time_encoder(delta_seconds.reshape(-1)).reshape(
                batch_size, neighbor_count, -1
            )
        else:
            # Keep the validity mask based on the real delta, but remove the
            # temporal information supplied to key/value projections.
            time_features = torch.zeros(
                batch_size,
                neighbor_count,
                time_encoder.dimension,
                dtype=neighbor_states.dtype,
                device=neighbor_states.device,
            )

        combined = torch.cat([neighbor_states, time_features], dim=-1)
        q = self.query(query_state).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(combined).view(
            batch_size, neighbor_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(combined).view(
            batch_size, neighbor_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

        logits = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(
            ~valid.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid.unsqueeze(1).to(weights.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights / normalizer

        attended = (weights.unsqueeze(-1) * v).sum(dim=2)
        attended = attended.reshape(batch_size, self.hidden_dim)
        result = self.norm(query_state + self.dropout(self.output(attended)))

        if return_attention:
            return result, weights.mean(dim=1)
        return result, None


class TGATTransactionModel(nn.Module):
    """TGAT-style account encoder with configurable ablation architecture."""

    def __init__(
        self,
        transaction_dim: int = 26,
        event_dim: int = 26,
        hidden_dim: int = 32,
        time_dim: int = 16,
        num_heads: int = 2,
        dropout: float = 0.2,
        num_layers: int = 2,
        use_time_encoding: bool = True,
        time_encoding_mode: str = "raw_seconds",
        max_period: float = 86_400.0,
    ) -> None:
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")
        self.transaction_dim = transaction_dim
        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_time_encoding = use_time_encoding
        self.time_encoding_mode = str(time_encoding_mode)
        self.max_period = float(max_period)

        self.node_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.time_encoder = TimeEncoder(
            time_dim,
            max_period=self.max_period,
            time_encoding_mode=self.time_encoding_mode,
        )
        self.attn1 = TemporalAttentionLayer(
            hidden_dim, time_dim, num_heads, dropout, use_time_encoding
        )
        self.attn2 = TemporalAttentionLayer(
            hidden_dim, time_dim, num_heads, dropout, use_time_encoding
        )
        self.transaction_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + transaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode_account(
        self,
        query_features: Tensor,
        neighbor_features: Tensor,
        delta_seconds: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Optional[list[Tensor]]]:
        query = self.node_encoder(query_features)
        neighbors = self.node_encoder(neighbor_features)
        query, attn1 = self.attn1(
            query, neighbors, delta_seconds, self.time_encoder, return_attention
        )
        if self.num_layers == 2:
            query, attn2 = self.attn2(
                query, neighbors, delta_seconds, self.time_encoder, return_attention
            )
        else:
            attn2 = None

        if return_attention:
            return query, [attn1, attn2]
        return query, None

    def forward(
        self,
        transaction_features: Tensor,
        sender_features: Tensor,
        sender_delta_seconds: Tensor,
        receiver_features: Tensor,
        receiver_delta_seconds: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Optional[dict[str, list[Tensor]]]]:
        sender_embedding, sender_attention = self.encode_account(
            sender_features[:, 0, :],
            sender_features[:, 1:, :],
            sender_delta_seconds,
            return_attention,
        )
        receiver_embedding, receiver_attention = self.encode_account(
            receiver_features[:, 0, :],
            receiver_features[:, 1:, :],
            receiver_delta_seconds,
            return_attention,
        )
        combined = torch.cat(
            [sender_embedding, receiver_embedding, transaction_features], dim=-1
        )
        representation = self.transaction_encoder(combined)
        logits = self.classifier(representation).squeeze(-1)

        if return_attention:
            return logits, {
                "sender": sender_attention or [],
                "receiver": receiver_attention or [],
            }
        return logits, None
