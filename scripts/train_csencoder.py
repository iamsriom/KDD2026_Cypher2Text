#!/usr/bin/env python3
"""
Train csencoder (Cypher–Structure encoder) with a lightweight contrastive objective.

Uses neo4j/text2cypher-2024v1 from HuggingFace Hub only.
- Builds token vocabulary from training Cypher
- Loads IDF from data/models/idf_scores_text2cypher.json (run compute_idf_text2cypher.py first)
- Trains for N epochs using InfoNCE over (cypher, augmented_cypher) positives
- Saves:
    data/models/csencoder_model.pt
    data/models/token_to_id.json

Use --limit N to train on N rows only (e.g. --limit 10 for a quick test).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure repo root on sys.path when running as a script (so `python scripts/*.py` works).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from csencoder.model.hybrid_cypher_graph_encoder import CSEncoder
from csencoder.preprocessing.graph_builder import GraphBuilder
from csencoder.preprocessing.rarity_calc import IDFCalculator

try:
    from data.setup.config import DEFAULT_HF_TOKEN_PATH
except ImportError:
    DEFAULT_HF_TOKEN_PATH = Path("/home/ubuntu/Huggingface_api.txt")

LOGGER = logging.getLogger("train_csencoder")


PAD_ID = 0
UNK_ID = 1


class VariableRenamingAugmenter:
    def __init__(self) -> None:
        self.var_pattern = re.compile(r"\b([a-z][a-zA-Z0-9]*)\b")

    def augment(self, cypher: str) -> str:
        variables = set(self.var_pattern.findall(cypher))
        variables = {v for v in variables if len(v) <= 3 and v.isalpha()}
        if not variables:
            return cypher
        replacement = [chr(c) for c in range(ord("x"), ord("z") + 1)] + [chr(c) for c in range(ord("a"), ord("w") + 1)]
        available = [v for v in replacement if v not in variables]
        mapping: Dict[str, str] = {}
        for v in sorted(variables):
            if available:
                mapping[v] = available.pop(0)
        out = cypher
        for old, new in mapping.items():
            out = re.sub(rf"\b{re.escape(old)}\b", new, out)
        return out


class CypherPairs(Dataset):
    def __init__(self, cyphers: List[str], augmenter: VariableRenamingAugmenter) -> None:
        self.cyphers = cyphers
        self.aug = augmenter

    def __len__(self) -> int:
        return len(self.cyphers)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        c = self.cyphers[idx]
        return c, self.aug.augment(c)


def load_hf_train_cyphers(dataset_name: str, limit: int | None = None) -> List[str]:
    """Load training Cypher strings from neo4j/text2cypher-2024v1 train split."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    cypher_col = "cypher" if "cypher" in ds.column_names else "query"
    cyphers = [c.strip() for c in ds[cypher_col] if isinstance(c, str) and c.strip()]
    if limit is not None:
        cyphers = cyphers[:limit]
    return cyphers


def build_vocab(cyphers: List[str], vocab_size: int) -> Dict[str, int]:
    tok = GraphBuilder().tokenizer
    token_to_id: Dict[str, int] = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for c in cyphers:
        for t in tok.tokenize(c):
            if t not in token_to_id:
                token_to_id[t] = len(token_to_id)
                if len(token_to_id) >= vocab_size:
                    return token_to_id
    return token_to_id


def batchify(
    gb: GraphBuilder,
    token_to_id: Dict[str, int],
    idf_scores: Dict[str, float],
    max_seq_len: int,
    cyphers: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # returns: input_ids, attention_mask, magnetic_pe, dist_matrix, idf_vec
    ids_b, mask_b, mag_b, dist_b, idf_b = [], [], [], [], []
    idf_calc = IDFCalculator()
    for c in cyphers:
        built = gb.build(c)
        tokens = built["tokens"][:max_seq_len]
        L = len(tokens)
        ids = [token_to_id.get(t, UNK_ID) for t in tokens]
        mask = [1] * L
        mag = built["magnetic_pe"][:L, :]
        dist = built["dist_matrix"][:L, :L]
        if mag.shape[1] != 16:
            mag = mag[:, :16] if mag.shape[1] > 16 else np.pad(mag, ((0, 0), (0, 16 - mag.shape[1])), mode="constant")
        idf_vec = idf_calc.get_idf_vector(tokens, idf_scores) if idf_scores else np.zeros((L,), dtype=np.float32)

        pad_n = max_seq_len - L
        if pad_n > 0:
            ids += [PAD_ID] * pad_n
            mask += [0] * pad_n
            idf_vec = np.pad(idf_vec, (0, pad_n), mode="constant")
            mag = np.pad(mag, ((0, pad_n), (0, 0)), mode="constant")
            dist = np.pad(dist, ((0, pad_n), (0, pad_n)), mode="constant")

        ids_b.append(ids)
        mask_b.append(mask)
        mag_b.append(mag)
        dist_b.append(dist)
        idf_b.append(idf_vec)

    return (
        torch.tensor(ids_b, dtype=torch.long, device=device),
        torch.tensor(mask_b, dtype=torch.long, device=device),
        torch.tensor(np.stack(mag_b), dtype=torch.float32, device=device),
        torch.tensor(np.stack(dist_b), dtype=torch.float32, device=device),
        torch.tensor(np.stack(idf_b), dtype=torch.float32, device=device),
    )


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # z1,z2: [B,H] normalized
    B = z1.shape[0]
    logits = (z1 @ z2.T) / temperature  # [B,B]
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="neo4j/text2cypher-2024v1", help="HuggingFace dataset (train split).")
    parser.add_argument("--idf-path", type=Path, default=Path("/home/ubuntu/KDD2026_Cypher2Text/data/models/idf_scores_text2cypher.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ubuntu/KDD2026_Cypher2Text/data/models"))
    parser.add_argument("--limit", type=int, default=None, help="Limit training rows. Omit for full dataset; use e.g. --limit 10 for a quick test.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default 5 for full dataset).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default 32; use smaller if OOM).")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"))
    parser.add_argument("--hf-token-path", type=Path, default=DEFAULT_HF_TOKEN_PATH, help="Hugging Face token file.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # HuggingFace login for dataset
    if args.hf_token_path.exists():
        token = args.hf_token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            from huggingface_hub import login as hf_login
            hf_login(token=token, add_to_git_credential=False)
            LOGGER.info("Authenticated with Hugging Face for dataset %s", args.dataset_name)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cyphers = load_hf_train_cyphers(args.dataset_name, limit=args.limit)
    if not cyphers:
        LOGGER.error("No training cyphers loaded.")
        sys.exit(1)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # vocab
    token_to_id = build_vocab(cyphers, args.vocab_size)
    (out_dir / "token_to_id.json").write_text(json.dumps(token_to_id, indent=2), encoding="utf-8")
    LOGGER.info("Vocab size: %d (saved token_to_id.json)", len(token_to_id))

    # idf
    idf_scores: Dict[str, float] = {}
    if args.idf_path.exists():
        idf_scores = IDFCalculator().load(args.idf_path)
        LOGGER.info("Loaded IDF from %s (%d tokens)", args.idf_path, len(idf_scores))
    else:
        LOGGER.warning("IDF file not found at %s (rarity bias will be effectively off).", args.idf_path)

    device = torch.device(args.device)
    model = CSEncoder(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        pe_dim=16,
        max_seq_len=args.max_seq_len,
        use_structural_bias=True,
        use_rarity_bias=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gb = GraphBuilder()
    aug = VariableRenamingAugmenter()
    ds = CypherPairs(cyphers, aug)
    drop_last = len(ds) > args.batch_size  # keep last batch when only 10 queries
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = []
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for c1, c2 in pbar:
            # DataLoader returns lists of strings
            x1 = list(c1)
            x2 = list(c2)
            ids1, m1, pe1, d1, idf1 = batchify(gb, token_to_id, idf_scores, args.max_seq_len, x1, device)
            ids2, m2, pe2, d2, idf2 = batchify(gb, token_to_id, idf_scores, args.max_seq_len, x2, device)
            z1 = model(ids1, pe1, dist_matrix=d1, idf_scores=idf1, attention_mask=m1)
            z2 = model(ids2, pe2, dist_matrix=d2, idf_scores=idf2, attention_mask=m2)
            loss = info_nce(z1, z2, temperature=args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=np.mean(losses[-50:]))
        LOGGER.info("epoch %d: mean_loss=%.4f", epoch + 1, float(np.mean(losses)) if losses else float("nan"))

    elapsed = time.time() - t0
    ckpt = out_dir / "csencoder_model.pt"
    torch.save(model.state_dict(), ckpt)
    LOGGER.info("Saved checkpoint: %s", ckpt)
    LOGGER.info("Training complete. elapsed=%.1fs (epochs=%d)", elapsed, args.epochs)


if __name__ == "__main__":
    main()

