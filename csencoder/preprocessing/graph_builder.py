#!/usr/bin/env python3
"""
Graph Builder for HybridCypherGraph-Encoder (HCGE).

Converts raw Cypher text into:
1. Tokenized text sequence
2. Directed graph structure (AST if available; regex fallback)
3. Shortest-Path-Distance (SPD) matrix
4. Magnetic Laplacian positional encodings (eigenvectors)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix


class CypherTokenizer:
    """Simple tokenizer for Cypher queries using regex patterns."""

    KEYWORDS = [
        "MATCH",
        "WHERE",
        "RETURN",
        "WITH",
        "CREATE",
        "MERGE",
        "DELETE",
        "SET",
        "REMOVE",
        "ORDER",
        "BY",
        "LIMIT",
        "SKIP",
        "UNWIND",
        "CALL",
        "AND",
        "OR",
        "NOT",
        "XOR",
        "IN",
        "STARTS",
        "ENDS",
        "CONTAINS",
        "IS",
        "NULL",
        "DISTINCT",
        "AS",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
    ]

    OPERATORS = ["=", "<>", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "%", "^"]

    def __init__(self):
        keyword_pattern = "|".join(rf"\b{kw}\b" for kw in self.KEYWORDS)
        operator_pattern = "|".join(
            re.escape(op) for op in sorted(self.OPERATORS, key=len, reverse=True)
        )
        self.pattern = re.compile(
            rf"({keyword_pattern})|({operator_pattern})|"
            r"([a-zA-Z_][a-zA-Z0-9_]*)|"
            r"(\d+\.?\d*)|"
            r"(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')|"
            r"([()\[\]{},:;.])"
        )

    def tokenize(self, cypher: str) -> List[str]:
        tokens = []
        for match in self.pattern.finditer(cypher):
            token = match.group(0)
            if token.strip():
                tokens.append(token.strip())
        return tokens


class ASTGraphBuilder:
    """Build a directed graph from a Cypher AST structure."""

    def __init__(self, parser=None):
        self.parser = parser
        self.tokenizer = CypherTokenizer()

    def build_graph_from_ast(self, ast: Dict[str, Any], tokens: List[str]) -> nx.DiGraph:
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type="token")
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type="sequence")
        if ast and isinstance(ast, dict):
            self._add_ast_edges(G, ast, tokens)
        return G

    def _add_ast_edges(
        self,
        G: nx.DiGraph,
        node: Dict[str, Any],
        tokens: List[str],
        parent_idx: int | None = None,
    ):
        node_text = node.get("text", "")
        if not node_text:
            return
        token_indices = []
        text_lower = node_text.lower()
        for i, token in enumerate(tokens):
            if token.lower() in text_lower:
                token_indices.append(i)

        if parent_idx is not None and token_indices:
            for child_idx in token_indices:
                if parent_idx != child_idx:
                    G.add_edge(parent_idx, child_idx, edge_type="hierarchy")

        if not token_indices:
            return
        current_rep = token_indices[0]
        for child in node.get("children", []):
            if isinstance(child, dict):
                self._add_ast_edges(G, child, tokens, parent_idx=current_rep)

    def build_graph_from_regex(self, cypher: str) -> nx.DiGraph:
        tokens = self.tokenizer.tokenize(cypher)
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            G.add_node(i, token=token, type="token")
        for i in range(len(tokens) - 1):
            G.add_edge(i, i + 1, edge_type="sequence")

        match_pattern = re.compile(
            r"MATCH\s+(.+?)(?=\b(?:WHERE|RETURN|WITH|$))", re.IGNORECASE
        )
        match_content = match_pattern.search(cypher)
        if match_content:
            content = match_content.group(1)
            node_pattern = re.compile(r"\(([^)]+)\)")
            rel_pattern = re.compile(r"-\[([^\]]+)\]-")
            nodes = node_pattern.findall(content)
            _rels = rel_pattern.findall(content)

            for i, token in enumerate(tokens):
                if any(node_part in token for node_part in nodes):
                    for j in range(i + 1, min(i + 10, len(tokens))):
                        if any(node_part in tokens[j] for node_part in nodes):
                            G.add_edge(i, j, edge_type="structural")
                            break
        return G


def compute_magnetic_laplacian(
    G: nx.DiGraph, q: float = 0.25, k: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(G)
    if n == 0:
        return np.array([]), np.array([])

    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes()), weight=None).toarray()
    A = A.astype(np.complex128)
    D = np.diag(np.sum(A.real, axis=1))
    A_asym = A - A.T
    phase = 2 * np.pi * q * A_asym
    exp_phase = np.exp(1j * phase)
    A_mag = A * exp_phase
    L_mag = D - A_mag
    L_sym = (L_mag + L_mag.conj().T) / 2
    L_sparse = csr_matrix(L_sym.real)
    try:
        eigenvalues, eigenvectors = eigh(L_sparse.toarray())
    except Exception:
        eigenvalues, eigenvectors = eigh(L_sym.real)

    idx = np.argsort(eigenvalues)[:k]
    eigenvals = eigenvalues[idx]
    eigenvecs = eigenvectors[:, idx]
    eigenvecs_complex = eigenvecs.astype(np.complex128)
    eigenvecs_combined = np.hstack([eigenvecs_complex.real, eigenvecs_complex.imag])
    return eigenvals, eigenvecs_combined


def compute_spd_matrix(G: nx.DiGraph) -> np.ndarray:
    n = len(G)
    if n == 0:
        return np.array([]).reshape(0, 0)
    dist_matrix = np.full((n, n), np.inf)
    for i in sorted(G.nodes()):
        for j in sorted(G.nodes()):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                try:
                    dist_matrix[i, j] = nx.shortest_path_length(G, i, j)
                except nx.NetworkXNoPath:
                    dist_matrix[i, j] = np.inf

    max_finite = np.max(dist_matrix[dist_matrix != np.inf])
    if np.isfinite(max_finite):
        dist_matrix[dist_matrix == np.inf] = max_finite + 1
    else:
        dist_matrix[dist_matrix == np.inf] = n
    return dist_matrix


class GraphBuilder:
    def __init__(self, parser=None):
        self.ast_builder = ASTGraphBuilder(parser=parser)
        self.tokenizer = CypherTokenizer()

    def build(self, cypher: str, ast: Dict[str, Any] | None = None) -> Dict[str, Any]:
        tokens = self.tokenizer.tokenize(cypher)
        if len(tokens) == 0:
            return {
                "tokens": [],
                "graph": nx.DiGraph(),
                "dist_matrix": np.array([]).reshape(0, 0),
                "magnetic_pe": np.array([]).reshape(0, 0),
            }

        if ast:
            G = self.ast_builder.build_graph_from_ast(ast, tokens)
        else:
            G = self.ast_builder.build_graph_from_regex(cypher)

        for i in range(len(tokens)):
            if i not in G:
                G.add_node(i, token=tokens[i], type="token")

        dist_matrix = compute_spd_matrix(G)
        eigenvals, magnetic_pe = compute_magnetic_laplacian(G, q=0.25, k=8)
        return {
            "tokens": tokens,
            "graph": G,
            "dist_matrix": dist_matrix,
            "magnetic_pe": magnetic_pe,
            "eigenvalues": eigenvals,
        }

