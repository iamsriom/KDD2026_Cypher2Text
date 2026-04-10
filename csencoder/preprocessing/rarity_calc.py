#!/usr/bin/env python3
"""
Rarity Calculator for HybridCypherGraph-Encoder.

Computes Inverse Document Frequency (IDF) scores for tokens in the training corpus.
Used for IDF-weighted attention mechanism.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from csencoder.preprocessing.graph_builder import CypherTokenizer


class IDFCalculator:
    """Calculate IDF scores for tokens in a corpus."""

    def __init__(self):
        self.tokenizer = CypherTokenizer()
        self.token_df = Counter()
        self.total_docs = 0
        self.idf_scores: Dict[str, float] = {}

    def compute_idf(self, corpus: List[str]) -> Dict[str, float]:
        """
        IDF(t) = log(N / (1 + df(t)))
        """
        self.token_df = Counter()
        self.total_docs = len(corpus)

        for query in tqdm(corpus, desc="Computing document frequencies"):
            tokens = set(self.tokenizer.tokenize(query))
            for token in tokens:
                self.token_df[token] += 1

        idf_scores: Dict[str, float] = {}
        for token, df in self.token_df.items():
            idf_scores[token] = float(np.log(self.total_docs / (1.0 + df)))

        avg_idf = float(np.mean(list(idf_scores.values()))) if idf_scores else 0.0
        idf_scores["<UNK>"] = avg_idf
        self.idf_scores = idf_scores
        return idf_scores

    def save(self, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.idf_scores, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Path) -> Dict[str, float]:
        with open(filepath, "r", encoding="utf-8") as f:
            self.idf_scores = json.load(f)
        return self.idf_scores

    def get_idf_vector(self, tokens: List[str], idf_scores: Dict[str, float]) -> np.ndarray:
        return np.array(
            [idf_scores.get(token, idf_scores.get("<UNK>", 0.0)) for token in tokens],
            dtype=np.float32,
        )


def compute_idf_from_dataset(dataset_path: str, output_path: str | None = None) -> Dict[str, float]:
    import pandas as pd

    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".jsonl"):
        df = pd.read_json(dataset_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")

    if "cypher" not in df.columns:
        raise ValueError("Dataset must contain 'cypher' column")

    queries = df["cypher"].dropna().tolist()
    calculator = IDFCalculator()
    idf_scores = calculator.compute_idf(queries)
    if output_path:
        calculator.save(Path(output_path))
    return idf_scores

