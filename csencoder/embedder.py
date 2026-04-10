from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from csencoder.model.hybrid_cypher_graph_encoder import CSEncoder
from csencoder.preprocessing.graph_builder import GraphBuilder
from csencoder.preprocessing.rarity_calc import IDFCalculator


PAD_ID = 0
UNK_ID = 1


@dataclass
class EmbedderConfig:
    hidden_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    vocab_size: int = 10000
    max_seq_len: int = 256
    max_distance: int = 50
    device: str = "cpu"


class CSEmbedder:
    """
    Cypher–sentence embedder (csencoder): wrapper around CSEncoder (Cypher Structure Encoder).

    Use this name (csencoder) everywhere. Self-contained for artifact and pipeline use.
    """

    def __init__(
        self,
        *,
        config: EmbedderConfig,
        idf_scores: Optional[Dict[str, float]] = None,
        token_to_id: Optional[Dict[str, int]] = None,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.graph_builder = GraphBuilder(parser=None)
        self.idf_scores = idf_scores or {}
        self.idf_calc = IDFCalculator()

        if token_to_id is None:
            token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
        self.token_to_id = dict(token_to_id)

        self.model = CSEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pe_dim=16,  # 2*k with k=8 in GraphBuilder
            max_seq_len=config.max_seq_len,
            use_structural_bias=True,
            use_rarity_bias=True,
        ).to(self.device)
        self.model.eval()

        if model_path:
            self.load_checkpoint(model_path)

    def load_checkpoint(self, model_path: str | Path) -> None:
        p = Path(model_path)
        state = torch.load(str(p), map_location=self.device)
        # allow both raw state_dict and {model_state_dict: ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)

    def maybe_extend_vocab(self, cyphers: Iterable[str]) -> None:
        """
        Optionally extend token_to_id on the fly (up to vocab_size).
        Useful for quick experiments; for strict reproducibility provide token_to_id.
        """
        for c in cyphers:
            tokens = self.graph_builder.tokenizer.tokenize(c)
            for t in tokens:
                if t not in self.token_to_id and len(self.token_to_id) < self.config.vocab_size:
                    self.token_to_id[t] = len(self.token_to_id)

    def save_vocab(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)

    def _encode_one(self, cypher: str) -> Tuple[np.ndarray, Dict]:
        built = self.graph_builder.build(cypher)
        tokens: List[str] = built["tokens"]
        if not tokens:
            return np.zeros((self.config.hidden_dim,), dtype=np.float32), built

        # truncate/pad
        tokens = tokens[: self.config.max_seq_len]
        L = len(tokens)
        ids = [self.token_to_id.get(t, UNK_ID) for t in tokens]
        attn = [1] * L

        # features
        dist = built["dist_matrix"][:L, :L]
        mag = built["magnetic_pe"][:L, :]
        if mag.shape[1] != 16:
            # safety: pad/trim PE dimension
            if mag.shape[1] < 16:
                mag = np.pad(mag, ((0, 0), (0, 16 - mag.shape[1])), mode="constant")
            else:
                mag = mag[:, :16]

        idf_vec = self.idf_calc.get_idf_vector(tokens, self.idf_scores) if self.idf_scores else np.zeros((L,), dtype=np.float32)

        # pad to max_seq_len
        maxL = self.config.max_seq_len
        pad_n = maxL - L
        if pad_n > 0:
            ids = ids + [PAD_ID] * pad_n
            attn = attn + [0] * pad_n
            idf_vec = np.pad(idf_vec, (0, pad_n), mode="constant")
            mag = np.pad(mag, ((0, pad_n), (0, 0)), mode="constant")
            dist = np.pad(dist, ((0, pad_n), (0, pad_n)), mode="constant")

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)
        dist_t = torch.tensor([dist], dtype=torch.float32, device=self.device)
        mag_t = torch.tensor([mag], dtype=torch.float32, device=self.device)
        idf_t = torch.tensor([idf_vec], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            emb = self.model(
                input_ids=input_ids,
                magnetic_pe=mag_t,
                dist_matrix=dist_t,
                idf_scores=idf_t,
                attention_mask=attention_mask,
            )
        out = emb[0].detach().cpu().numpy().astype(np.float32)
        return out, built

    def embed(self, cypher: str) -> np.ndarray:
        return self._encode_one(cypher)[0]

    def embed_many(self, cyphers: Sequence[str]) -> np.ndarray:
        self.maybe_extend_vocab(cyphers)
        embs = [self.embed(c) for c in cyphers]
        return np.stack(embs, axis=0)


# Backward compatibility
HCGEEmbedder = CSEmbedder

