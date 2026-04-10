#!/usr/bin/env python3
"""
Embed neo4j/text2cypher-2024v1 train/test Cypher queries and store in ChromaDB.

Loads from HuggingFace Hub dataset neo4j/text2cypher-2024v1.
Uses IDF from compute_idf_text2cypher.py and csencoder from train_csencoder.py.
Outputs:
  data/chromadb/text2cypher/train
  data/chromadb/text2cypher/test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data.setup.config import (
        DEFAULT_HF_DATASET_NAME,
        DEFAULT_HF_TOKEN_PATH,
        MODELS_DIR,
        CHROMADB_DIR,
    )
except ImportError:
    MODELS_DIR = REPO_ROOT / "data" / "models"
    CHROMADB_DIR = REPO_ROOT / "data" / "chromadb"
    DEFAULT_HF_DATASET_NAME = "neo4j/text2cypher-2024v1"
    DEFAULT_HF_TOKEN_PATH = Path.home() / "Huggingface_api.txt"

from csencoder.embedder import EmbedderConfig, CSEmbedder
from csencoder.preprocessing.rarity_calc import IDFCalculator

LOGGER = logging.getLogger("embed")


def ensure_hf_login(token_path: Path) -> None:
    token = os.getenv("HF_TOKEN")
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
    if not os.getenv("HF_TOKEN"):
        raise FileNotFoundError(f"Hugging Face token required. Set HF_TOKEN or create {token_path}")
    from huggingface_hub import login as hf_login
    hf_login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)


def load_hf_splits(
    dataset_name: str,
    limit: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    train_ds = ds["train"]
    test_ds = ds.get("test") or ds.get("validation")

    cypher_col = "cypher" if "cypher" in train_ds.column_names else "query"
    question_col = next(
        (c for c in ["question", "nl_question", "natural_language"] if c in train_ds.column_names),
        None,
    )

    def _split_to_instances(split_ds, split_name: str) -> List[Dict[str, Any]]:
        out = []
        n = len(split_ds)
        for i in range(n):
            c = split_ds[cypher_col][i]
            if not isinstance(c, str) or not c.strip():
                continue
            qid = f"{split_name}_{i}"
            if "id" in split_ds.column_names:
                qid = str(split_ds["id"][i])
            elif "qid" in split_ds.column_names:
                qid = str(split_ds["qid"][i])

            meta: Dict[str, Any] = {"qid": qid}
            for col in split_ds.column_names:
                if col == cypher_col:
                    continue
                v = split_ds[col][i]
                if v is None:
                    continue
                if isinstance(v, (dict, list)):
                    meta[col] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, (str, int, float, bool)):
                    meta[col] = v
                else:
                    meta[col] = str(v)

            if question_col and question_col in split_ds.column_names:
                meta["nl_question"] = split_ds[question_col][i] or ""

            out.append({"cypher": c.strip(), "id": qid, "metadata": meta})
            if limit is not None and len(out) >= limit:
                break
        return out

    train = _split_to_instances(train_ds, "train")
    test = _split_to_instances(test_ds, "test") if test_ds is not None else []
    return train, test


def store_collection(
    split_dir: Path,
    name: str,
    embeddings: np.ndarray,
    docs: List[str],
    metas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(split_dir))
    coll = client.get_or_create_collection(name=name)
    coll.add(
        embeddings=[e.tolist() for e in embeddings],
        documents=docs,
        metadatas=metas,
        ids=ids,
    )
    LOGGER.info("Stored %d items in %s (count=%d)", len(ids), split_dir, coll.count())


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed neo4j/text2cypher-2024v1 into ChromaDB.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_HF_DATASET_NAME,
        help="HuggingFace dataset (default: neo4j/text2cypher-2024v1)",
    )
    parser.add_argument(
        "--hf-token-path",
        type=Path,
        default=DEFAULT_HF_TOKEN_PATH,
        help="Hugging Face API token file",
    )
    parser.add_argument(
        "--idf-path",
        type=Path,
        default=MODELS_DIR / "idf_scores_text2cypher.json",
        help="IDF JSON from compute_idf_text2cypher.py",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "csencoder_model.pt",
        help="CSEncoder checkpoint from train_csencoder.py",
    )
    parser.add_argument(
        "--out-db",
        type=Path,
        default=CHROMADB_DIR / "text2cypher",
        help="ChromaDB output root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit per split. Omit for full; use e.g. --limit 50 for quick run.",
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ensure_hf_login(args.hf_token_path)

    train, test = load_hf_splits(args.dataset_name, limit=args.limit)
    LOGGER.info("Loaded train=%d test=%d from %s", len(train), len(test), args.dataset_name)
    if not train:
        LOGGER.error("No train data loaded.")
        sys.exit(1)

    idf_scores: Dict[str, float] = {}
    if args.idf_path.exists():
        idf_scores = IDFCalculator().load(args.idf_path)
        LOGGER.info("Loaded IDF: %d tokens from %s", len(idf_scores), args.idf_path)
    else:
        LOGGER.warning("IDF file not found at %s. Proceeding without rarity bias.", args.idf_path)

    token_to_id: Dict[str, int] | None = None
    token_path = MODELS_DIR / "token_to_id.json"
    if token_path.exists():
        token_to_id = json.loads(token_path.read_text(encoding="utf-8"))
        LOGGER.info("Loaded vocab: %d tokens from %s", len(token_to_id), token_path)

    model_path = args.model_path if args.model_path.exists() else None
    cfg = EmbedderConfig(max_seq_len=args.max_seq_len, device=args.device)
    embedder = CSEmbedder(
        config=cfg,
        idf_scores=idf_scores,
        token_to_id=token_to_id,
        model_path=model_path,
    )

    out_root = args.out_db.resolve()
    for split_name, rows in [("train", train), ("test", test)]:
        if not rows:
            continue
        docs = [r["cypher"] for r in rows]
        ids = [r["id"] for r in rows]
        metas = [r["metadata"] for r in rows]
        embs = embedder.embed_many(docs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        store_collection(out_root / split_name, "embeddings", embs, docs, metas, ids)

    LOGGER.info("Done. ChromaDB root: %s", out_root)


if __name__ == "__main__":
    main()
