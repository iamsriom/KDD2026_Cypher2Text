#!/usr/bin/env python3
"""
Compute top-k cosine similarity between test and train embeddings stored in ChromaDB.

Reads from data/chromadb/text2cypher (produced by embed.py).
Outputs:
  data/results/similarity/similarity_results.json
  data/results/similarity/similarity_results.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from data.setup.config import CHROMADB_DIR, RESULTS_DIR
except ImportError:
    CHROMADB_DIR = REPO_ROOT / "data" / "chromadb"
    RESULTS_DIR = REPO_ROOT / "data" / "results"

LOGGER = logging.getLogger("similarity")


def load_split(
    db_root: Path, split: str
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], List[str]]:
    split_path = db_root / split
    client = chromadb.PersistentClient(path=str(split_path))
    coll = client.get_or_create_collection("embeddings")
    res = coll.get(include=["embeddings", "documents", "metadatas"])
    embs = np.array(res["embeddings"], dtype=np.float32)
    docs = res["documents"]
    metas = res["metadatas"] or [{} for _ in range(len(docs))]
    ids = res.get("ids", [])
    return embs, docs, metas, ids


def top_k(test_vec: np.ndarray, train_mat: np.ndarray, k: int) -> np.ndarray:
    sims = train_mat @ test_vec
    return np.argsort(sims)[::-1][:k]


def fmt_block(test_id: str, test_query: str, matches: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"TEST: {test_id}")
    lines.append("=" * 80)
    lines.append(test_query)
    lines.append("\nTop matches:")
    for m in matches:
        lines.append(f"- rank={m['rank']} sim={m['similarity']:.6f} id={m['train_id']}")
        if m.get("gold_translation"):
            lines.append(f"  nl_question: {m['gold_translation']}")
        lines.append(f"  cypher: {m['query']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-k similarity from ChromaDB (output of embed.py)."
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=CHROMADB_DIR / "text2cypher",
        help="ChromaDB root (default: data/chromadb/text2cypher)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / "similarity",
        help="Output directory",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    db_root = args.db_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_embs, train_docs, train_metas, train_ids = load_split(db_root, "train")
    test_embs, test_docs, test_metas, test_ids = load_split(db_root, "test")

    if train_embs.size == 0 or test_embs.size == 0:
        LOGGER.error("Missing embeddings. Run scripts/embed.py first.")
        sys.exit(1)

    train = train_embs / np.clip(np.linalg.norm(train_embs, axis=1, keepdims=True), 1e-12, None)
    test = test_embs / np.clip(np.linalg.norm(test_embs, axis=1, keepdims=True), 1e-12, None)

    results: List[Dict[str, Any]] = []
    txt_blocks: List[str] = []
    for i in tqdm(range(test.shape[0]), desc="similarity"):
        tid = test_ids[i] if i < len(test_ids) else f"test_{i}"
        idxs = top_k(test[i], train, args.top_k)
        matches: List[Dict[str, Any]] = []
        for rank, j in enumerate(idxs, 1):
            meta = train_metas[j] if j < len(train_metas) else {}
            nlq = meta.get("nl_question") or meta.get("question") or meta.get("NL Question") or ""
            matches.append(
                {
                    "rank": int(rank),
                    "train_index": int(j),
                    "train_id": train_ids[j] if j < len(train_ids) else f"train_{j}",
                    "similarity": float(train[j] @ test[i]),
                    "query": train_docs[j],
                    "gold_translation": nlq,
                    "metadata": meta,
                }
            )
        rec = {
            "test_index": int(i),
            "test_id": str(tid),
            "test_query": test_docs[i],
            "test_metadata": test_metas[i] if i < len(test_metas) else {},
            "top_matches": matches,
        }
        results.append(rec)
        txt_blocks.append(fmt_block(str(tid), test_docs[i], matches))

    (out_dir / "similarity_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "similarity_results.txt").write_text("\n".join(txt_blocks), encoding="utf-8")
    LOGGER.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
