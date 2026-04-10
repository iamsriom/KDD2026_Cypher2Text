# RUN.md

Reproducible run instructions for KDD reviewers and for anyone reproducing results.

---

## 1) Prerequisites

- Python 3.10+
- Hugging Face access to `neo4j/text2cypher-2024v1`
- Optional for LLM stages: OpenAI API key

From repo root:

```bash
make install
```

Authentication options:

- Set environment variable:
  ```bash
  export HF_TOKEN=<your_hf_token>
  ```
- Or pass `--hf-token-path /path/to/token_file` to scripts.

---

## 2) Fast Repro Path (Recommended First)

```bash
make quick
```

This runs:

1. `python scripts/data.py --limit 10 --skip-query-execution`
2. `python scripts/compute_idf_text2cypher.py --limit 10`
3. `python scripts/train_csencoder.py --limit 10 --epochs 2 --batch-size 4 --seed 42`

Expected outputs:

- `data/models/idf_scores_text2cypher.json`
- `data/models/csencoder_model.pt`
- `data/models/token_to_id.json`

---

## 3) Full Core Pipeline

```bash
make full
```

Equivalent explicit commands:

```bash
python scripts/data.py --skip-query-execution
python scripts/compute_idf_text2cypher.py
python scripts/train_csencoder.py --epochs 5 --batch-size 32 --seed 42
```

---

## 4) Optional Retrieval Stage

After the core pipeline:

```bash
python scripts/embed.py
python scripts/similarity.py
```

Generated artifacts:

- `data/chromadb/text2cypher/train`
- `data/chromadb/text2cypher/test`
- `data/results/similarity/similarity_results.json`
- `data/results/similarity/similarity_results.txt`

---

## 5) Optional Generation Stages

### Cypher2Text

```bash
python scripts/cypher2text_cot.py --help
```

Typical (OpenAI) run:

```bash
python scripts/cypher2text_cot.py \
  --models gpt4o-mini \
  --similarity-json data/results/similarity/similarity_results.json \
  --output-dir data/results/cypher2text
```

### Text2Cypher

```bash
python scripts/text2cypher_openai.py \
  --in-jsonl <cypher2text_output.jsonl> \
  --out-jsonl <text2cypher_output.jsonl> \
  --model gpt-4o-mini
```

Use `--dry-run` to validate prompts without API calls.

---

## 6) Reproducibility Notes

- Training is seeded via `--seed` (default `42`).
- Paths are repo-relative or home-relative (no hardcoded machine-specific `/home/...` defaults).
- Generated artifacts are not required to be committed; rerun scripts to recreate.

---

## 7) Troubleshooting

- **Missing HF auth**: set `HF_TOKEN` or provide `--hf-token-path`.
- **Slow install / platform mismatch**: use a clean virtual environment.
- **GPU issues**: run on CPU or set `CUDA_VISIBLE_DEVICES`.
- **No similarity output**: ensure `embed.py` ran successfully before `similarity.py`.
