#!/usr/bin/env python3
"""
Compute IDF scores from neo4j/text2cypher-2024v1 train split.

Uses HuggingFace Hub dataset only. Default: 10 training Cypher queries.
Output: data/models/idf_scores_text2cypher.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.setup.config import DATA_DIR, DEFAULT_HF_DATASET_NAME, DEFAULT_HF_TOKEN_PATH
from csencoder.preprocessing.rarity_calc import IDFCalculator

LOGGER = logging.getLogger("compute_idf_text2cypher")

DEFAULT_OUTPUT = DATA_DIR / "models" / "idf_scores_text2cypher.json"


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


def load_train_cyphers(dataset_name: str, limit: int | None = None) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    cypher_col = "cypher" if "cypher" in ds.column_names else "query"
    cyphers = [c.strip() for c in ds[cypher_col] if isinstance(c, str) and c.strip()]
    if limit is not None:
        cyphers = cyphers[:limit]
    return cyphers


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IDF from neo4j/text2cypher-2024v1 train.")
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write IDF JSON",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max training rows. Omit for full dataset; use e.g. --limit 10 for a quick run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ensure_hf_login(args.hf_token_path)
    cyphers = load_train_cyphers(args.dataset_name, limit=args.limit)
    if not cyphers:
        LOGGER.error("No train cyphers loaded from %s", args.dataset_name)
        sys.exit(1)

    calc = IDFCalculator()
    idf = calc.compute_idf(cyphers)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    calc.save(out)
    LOGGER.info("Wrote IDF (%d tokens) to %s", len(idf), out)


if __name__ == "__main__":
    main()
