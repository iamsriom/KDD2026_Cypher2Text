#!/usr/bin/env python3
"""
CSEncoder — Cypher Structure Encoder Model.

Deep hybrid embedding model that fuses:
- Sequential semantic information (text)
- Directed structural information (AST topology)
- Magnetic Laplacian positional encodings
- IDF-weighted attention
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagneticPE(nn.Module):
    """
    Magnetic Positional Encoding module.

    Projects complex Magnetic Laplacian eigenvectors (real + imaginary parts)
    to hidden dimension.
    """

    def __init__(self, pe_dim: int, hidden_dim: int):
        super().__init__()
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(pe_dim, hidden_dim)

    def forward(self, magnetic_pe: torch.Tensor) -> torch.Tensor:
        return self.projection(magnetic_pe)


class GraphBiasedAttention(nn.Module):
    """
    Graph-Biased Self-Attention with Structural and Rarity Biases.

    scores = QK^T / sqrt(d_k) + structural_bias(D_ij) + rarity_weight * (IDF_i + IDF_j)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        use_structural_bias: bool = True,
        use_rarity_bias: bool = True,
        max_distance: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_structural_bias = use_structural_bias
        self.use_rarity_bias = use_rarity_bias
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if use_structural_bias:
            self.structural_embedding = nn.Embedding(max_distance + 1, num_heads)
        if use_rarity_bias:
            self.rarity_weight = nn.Parameter(torch.ones(1) * 0.1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        idf_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.use_structural_bias and dist_matrix is not None:
            dist_clamped = torch.clamp(
                dist_matrix.long(), 0, self.structural_embedding.num_embeddings - 1
            )
            structural_bias = self.structural_embedding(dist_clamped).permute(0, 3, 1, 2)
            scores = scores + structural_bias

        if self.use_rarity_bias and idf_scores is not None:
            idf_i = idf_scores.unsqueeze(1).unsqueeze(2)
            idf_j = idf_scores.unsqueeze(1).unsqueeze(3)
            rarity_bias = self.rarity_weight * (idf_i + idf_j)
            rarity_bias = rarity_bias.expand(-1, self.num_heads, -1, -1)
            scores = scores + rarity_bias

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)

        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )
        output = self.out_proj(output)
        return output, attention_weights


class CSEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        use_structural_bias: bool = True,
        use_rarity_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        self.attention = GraphBiasedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_structural_bias=use_structural_bias,
            use_rarity_bias=use_rarity_bias,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        idf_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output, _ = self.attention(
            hidden_states,
            dist_matrix=dist_matrix,
            idf_scores=idf_scores,
            attention_mask=attention_mask,
        )
        hidden_states = self.norm1(hidden_states + attn_output)
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + ffn_output)
        return hidden_states


class CSEncoder(nn.Module):
    """
    Cypher Structure Encoder: stack of graph-biased attention layers
    producing L2-normalized embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        pe_dim: int = 16,
        max_seq_len: int = 512,  # kept for API compatibility
        use_structural_bias: bool = True,
        use_rarity_bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.magnetic_pe = MagneticPE(pe_dim=pe_dim, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [
                CSEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    use_structural_bias=use_structural_bias,
                    use_rarity_bias=use_rarity_bias,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.pooling = "mean"

    def forward(
        self,
        input_ids: torch.Tensor,
        magnetic_pe: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        idf_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.magnetic_pe(magnetic_pe)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                dist_matrix=dist_matrix,
                idf_scores=idf_scores,
                attention_mask=attention_mask,
            )

        hidden_states = self.layer_norm(hidden_states)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states * mask_expanded
            seq_lengths = mask_expanded.sum(dim=1)
            pooled = hidden_states.sum(dim=1) / seq_lengths
        else:
            pooled = hidden_states.mean(dim=1)

        return F.normalize(pooled, p=2, dim=1)

