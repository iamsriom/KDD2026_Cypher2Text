# Data setup (KDD2026_Cypher2Text)

This folder holds configuration for the KDD2026 data pipeline.

## Layout

- **`config.py`** – Path constants (`DATA_DIR`, `RESULTS_DIR`, default dataset dirs, etc.). Import from `data.setup.config` or adjust `sys.path` and use `from config import ...` when running from `data/setup/`.

## Pipeline

The main data script **`scripts/data.py`** loads **only** the HuggingFace dataset `neo4j/text2cypher-2024v1`. It:

1. **Loads train/test** from the Hub (requires HF token for gated access).
2. **Prints a data summary** – Shape, columns, distribution by `database_reference_alias`.
3. **Optional Neo4j execution** – Can run Cypher against Neo4j Labs demo DBs (do not use `--skip-query-execution`).

After training, **embed** (`scripts/embed.py`) and **similarity** (`scripts/similarity.py`) use the same HF neo4j dataset to build ChromaDB and compute top-k similarity (see README in repo root).

All outputs (results, models, ChromaDB) live under `data/` as defined in `config.py`.

## Quick start

```bash
# From repo root: load Hub dataset, summary only
python scripts/data.py --skip-query-execution

# With a row limit
python scripts/data.py --limit 10 --skip-query-execution

# Run up to 5 test queries against Neo4j
python scripts/data.py --query-limit 5
```

## Dependencies

- `datasets` (HuggingFace)
- `pandas`
- For Neo4j execution: `neo4j`, and HF token for `neo4j/text2cypher-2024v1`
