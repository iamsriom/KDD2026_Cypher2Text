#!/usr/bin/env python3
"""
Text2Cypher (OpenAI) – convert generated NL questions back into Cypher.

Input:
  JSONL produced by scripts/cypher2text_cot.py (predicted_translation + test_id + test_cypher)

Output:
  JSONL with:
    - test_id
    - original_cypher
    - generated_cypher

Supports --dry-run (no API calls) so you can validate prompt construction.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger("text2cypher_openai")


def build_prompt(question: str, schema_text: str | None = None) -> str:
    schema_block = f"\nSchema:\n{schema_text}\n" if schema_text else ""
    return (
        "You are a Cypher expert. Write a single Cypher query that answers the question.\n"
        "Output ONLY the Cypher query, no explanation, no markdown.\n"
        f"{schema_block}\n"
        f"Question: {question}\n\n"
        "Cypher:"
    )


def extract_cypher(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # Remove fenced blocks if any
    m = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Otherwise first line onward
    return t


def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-jsonl",
        type=Path,
        default=Path("/home/ubuntu/KDD2026_Cypher2Text/data/results/cypher2text/cypher2text.jsonl"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("/home/ubuntu/KDD2026_Cypher2Text/data/results/text2cypher/text2cypher.jsonl"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--schema-text", type=Path, default=None, help="Optional plain-text schema to include.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    if not have_key and not args.dry_run:
        LOGGER.warning("OPENAI_API_KEY not set; forcing --dry-run.")
        args.dry_run = True

    schema_txt = args.schema_text.read_text(encoding="utf-8") if args.schema_text else None

    in_path = args.in_jsonl.resolve()
    out_path = args.out_jsonl.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="text2cypher"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = (r.get("predicted_translation") or "").strip()
            test_id = r.get("test_id")
            orig = r.get("test_cypher") or ""
            prompt = build_prompt(q, schema_txt)
            if args.dry_run:
                gen = ""
            else:
                gen = extract_cypher(call_openai(prompt, args.model))
            rec = {
                "test_id": test_id,
                "original_cypher": orig,
                "generated_cypher": gen,
                "prompt": prompt if args.dry_run else None,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    LOGGER.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

