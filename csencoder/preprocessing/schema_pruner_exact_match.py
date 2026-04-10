#!/usr/bin/env python3
"""
Exact-Match Schema Pruner for Text2Cypher.

This module prunes a graph schema based on exact word-level matches with a natural language question.
It handles singular/plural forms, verb morphology, and ensures connectivity between matched nodes.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


def normalize_to_singular(word: str) -> str:
    """
    Normalize a word to its singular form (simple heuristic).
    """
    word_lower = word.lower()

    if word_lower.endswith("ies") and len(word_lower) > 3:
        return word_lower[:-3] + "y"
    if word_lower.endswith("es") and len(word_lower) > 2:
        if len(word_lower) > 3 and word_lower[-3] in "chshxz":
            return word_lower[:-2]
        if word_lower.endswith("oes"):
            return word_lower[:-2]
        return word_lower[:-2]
    if word_lower.endswith("s") and len(word_lower) > 1:
        return word_lower[:-1]
    return word_lower


def normalize_verb_form(verb: str) -> str:
    """
    Normalize a verb to its base form (simple heuristic).
    """
    verb_lower = verb.lower()

    if verb_lower.endswith("ing") and len(verb_lower) > 3:
        base = verb_lower[:-3]
        if len(base) > 1 and base[-1] == base[-2]:
            return base[:-1]
        return base
    if verb_lower.endswith("ed") and len(verb_lower) > 2:
        base = verb_lower[:-2]
        if len(base) > 1 and base[-1] == base[-2]:
            return base[:-1]
        return base
    if verb_lower.endswith("es") and len(verb_lower) > 2:
        return verb_lower[:-2]
    if verb_lower.endswith("s") and len(verb_lower) > 1:
        return verb_lower[:-1]
    return verb_lower


def get_word_variants(word: str) -> set[str]:
    """
    Generate all variants of a word (singular, plural, verb forms).
    """
    variants = {word.lower()}
    singular = normalize_to_singular(word)
    variants.add(singular)

    if singular:
        if singular.endswith("y") and len(singular) > 1:
            variants.add(singular[:-1] + "ies")
        elif singular[-1] in "chshxz":
            variants.add(singular + "es")
        elif singular.endswith("o"):
            variants.add(singular + "es")
        else:
            variants.add(singular + "s")

    verb_base = normalize_verb_form(word)
    variants.add(verb_base)
    variants.add(verb_base + "s")
    variants.add(verb_base + "ed")
    variants.add(verb_base + "ing")
    return variants


def tokenize_question(question: str) -> set[str]:
    """
    Lowercase the question and split it into tokens using non-alphanumeric characters as separators.
    Return a set of all word variants (singular, plural, verb forms) for flexible matching.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    all_variants: set[str] = set()
    for token in tokens:
        all_variants.update(get_word_variants(token))
        all_variants.add(token)
    return all_variants


def tokenize_name(name: str) -> list[str]:
    """
    Lowercase a schema name and split it into basic word tokens.
    """
    name = re.sub(r"[_\-\.,]", " ", name)
    if name.isupper():
        tokens = [name.lower()]
    else:
        name = re.sub(r"(?<!^)(?<! )([A-Z])", r" \1", name)
        tokens = name.split()
    return [t.lower() for t in tokens if t.strip()]


def matches_question(schema_token: str, question_variants: set[str]) -> bool:
    """
    Check if a schema token (or any of its variants) matches any question variant.
    """
    schema_variants = get_word_variants(schema_token)
    return bool(schema_variants & question_variants)


def rel_type_is_relevant(rel_type: str, question_variants: set[str], cypher_variants: set[str] | None = None) -> bool:
    """
    Check if a relationship type is relevant to the question and/or Cypher query.
    Uses lenient token/substring matching.
    Only returns True if the relationship type itself matches the question or Cypher.
    """
    if cypher_variants is None:
        cypher_variants = set()

    all_variants = question_variants | cypher_variants
    rel_tokens = tokenize_name(rel_type)
    rel_type_lower = rel_type.lower()

    if any(matches_question(token, all_variants) for token in rel_tokens):
        return True

    for variant in all_variants:
        if len(variant) > 2 and variant in rel_type_lower:
            return True
        if len(rel_type_lower) > 2 and rel_type_lower in variant:
            return True
    return False


def prune_schema_by_exact_match(question: str, schema: dict[str, Any], cypher: str | None = None) -> dict[str, Any]:
    """
    Perform Exact-Match pruning on the given base schema, based on the question and optionally Cypher query.
    """
    question_variants = tokenize_question(question)
    cypher_variants = tokenize_question(cypher) if cypher else set()

    original_node_count = len(schema.get("nodes", []))
    original_rel_count = len(schema.get("relationships", []))

    label_to_node: dict[str, dict[str, Any]] = {}
    for node in schema.get("nodes", []):
        label = node.get("label", "")
        if label:
            label_to_node[label] = node

    kept_relationships: list[dict[str, Any]] = []
    for rel in schema.get("relationships", []):
        rel_type = rel.get("type", "")
        if rel_type and rel_type_is_relevant(rel_type, question_variants, cypher_variants):
            kept_relationships.append(rel)

    required_node_labels: set[str] = set()
    for rel in kept_relationships:
        from_label = rel.get("from", "")
        to_label = rel.get("to", "")
        if from_label:
            required_node_labels.add(from_label)
        if to_label:
            required_node_labels.add(to_label)

    primary_node_labels: set[str] = set()
    for node in schema.get("nodes", []):
        label = node.get("label", "")
        if not label:
            continue
        label_tokens = tokenize_name(label)
        if any(matches_question(token, question_variants) for token in label_tokens):
            required_node_labels.add(label)
            primary_node_labels.add(label)

    if len(primary_node_labels) > 1:
        for rel in schema.get("relationships", []):
            from_label = rel.get("from", "")
            to_label = rel.get("to", "")
            rel_type = rel.get("type", "")
            if not rel_type or not from_label or not to_label:
                continue
            if from_label in primary_node_labels and to_label in primary_node_labels:
                already_kept = any(
                    r.get("type") == rel_type and r.get("from") == from_label and r.get("to") == to_label
                    for r in kept_relationships
                )
                if not already_kept:
                    kept_relationships.append(rel)
                    required_node_labels.add(from_label)
                    required_node_labels.add(to_label)

    kept_nodes: list[dict[str, Any]] = []
    for label in required_node_labels:
        if label in label_to_node:
            kept_nodes.append(label_to_node[label])
        elif label:
            kept_nodes.append({"label": label, "properties": []})

    final_required_labels = {node.get("label", "") for node in kept_nodes}
    for rel in kept_relationships:
        from_label = rel.get("from", "")
        to_label = rel.get("to", "")
        if from_label:
            final_required_labels.add(from_label)
        if to_label:
            final_required_labels.add(to_label)

    final_kept_nodes: list[dict[str, Any]] = []
    final_added_labels: set[str] = set()
    for label in final_required_labels:
        if label and label not in final_added_labels:
            if label in label_to_node:
                final_kept_nodes.append(label_to_node[label])
            else:
                final_kept_nodes.append({"label": label, "properties": []})
            final_added_labels.add(label)

    kept_nodes = final_kept_nodes

    if (original_node_count > 0 and len(kept_nodes) == 0) or (original_rel_count > 0 and len(kept_relationships) == 0):
        return schema.copy()

    return {"nodes": kept_nodes, "relationships": kept_relationships}


def main() -> None:
    """
    Command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Prune a graph schema based on exact word-level matches with a question."
    )
    parser.add_argument("--schema", type=str, required=True, help="Path to the input JSON file containing the base schema.")
    parser.add_argument("--question", type=str, required=True, help="The natural language question to filter for.")
    parser.add_argument("--output", type=str, default=None, help="Path to write pruned schema JSON. If omitted, prints to stdout.")
    args = parser.parse_args()

    with open(args.schema, "r", encoding="utf-8") as file_handle:
        schema = json.load(file_handle)

    pruned_schema = prune_schema_by_exact_match(args.question, schema)
    output_json = json.dumps(pruned_schema, indent=2, sort_keys=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file_handle:
            file_handle.write(output_json)
        print(f"Pruned schema saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
