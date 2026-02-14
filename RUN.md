# How to Run the Code (KDD 2026 Code Submission)

This document describes how to run the CSEncoder pipeline end-to-end and how each component works.

---

## 1. Prerequisites

- **Python 3.10+**
- **HuggingFace access**: The main pipeline uses the dataset `neo4j/text2cypher-2024v1` (gated). You must:
  - Accept the dataset terms on [HuggingFace](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)
  - Provide a token either by:
    - Setting the environment variable `HF_TOKEN`, or
    - Placing your token in a file and passing `--hf-token-path /path/to/token` to the scripts that need it (default path used by scripts: `$HOME/Huggingface_api.txt` or the path in `data/setup/config.py`)

Install dependencies from the repository root:

```bash
cd /path/to/KDD2026_Cypher2Text
pip install -r requirements.txt
```

---

## 2. How the Pipeline Works

The system performs **Cypher-to-Text** and **Text-to-Cypher** translation using a hybrid Cypher encoder (CSEncoder) and optional retrieval over embeddings.

### High-level flow

1. **Data**  
   Train/test Cypher (and metadata) are loaded from the HuggingFace dataset `neo4j/text2cypher-2024v1`.

2. **IDF (Inverse Document Frequency)**  
   IDF scores are computed from the **training** Cypher corpus. These are used to weight rare tokens in the encoder (IDF-weighted attention).

3. **Training CSEncoder**  
   The encoder is trained with contrastive learning: positive pairs are (cypher, lightly augmented cypher). The model learns embeddings so that similar Cypher queries are close in vector space.

4. **Embedding & similarity (optional)**  
   For few-shot prompting, you can embed train/test Cypher from the HF dataset, store them in ChromaDB, and compute top-k similar training examples per test query. Use `embed.py` (after Steps 1–3) and `similarity.py`.

5. **Cypher2Text**  
   Given a test Cypher query, the system (optionally) retrieves similar training examples and uses an LLM (e.g. CodeLlama or GPT-4o-mini) with Chain-of-Thought to produce a natural language question.

6. **Text2Cypher**  
   The generated natural language question can be sent to an LLM (e.g. OpenAI) with the schema to produce a Cypher query again (round-trip or evaluation).

### What each script does

| Script | Purpose | Main inputs | Main outputs |
|--------|--------|-------------|--------------|
| `data.py` | Load HF dataset, print summary, optional Neo4j execution | HF dataset name, HF token | Console summary; optional CSV under `data/dataset/` |
| `compute_idf_text2cypher.py` | Compute IDF from HF train split | HF train Cypher | `data/models/idf_scores_text2cypher.json` |
| `train_csencoder.py` | Train CSEncoder (contrastive) | Train Cypher, IDF file | `data/models/csencoder_model.pt`, `data/models/token_to_id.json` |
| `embed.py` | Embed HF neo4j dataset → ChromaDB | HF dataset, IDF, model checkpoint | `data/chromadb/text2cypher/{train,test}` |
| `similarity.py` | Top-k similarity train vs test | ChromaDB (train + test) | `data/results/similarity/similarity_results.json` |
| `cypher2text_cot.py` | Cypher → NL (LLM + CoT) | Similarity results, pruned schemas, test set | JSON/TXT under output dir |
| `text2cypher_openai.py` | NL → Cypher (LLM) | JSONL from Cypher2Text | JSONL with generated Cypher |

---

## 3. Step-by-Step: Running the Code

All commands are run from the **repository root**: `KDD2026_Cypher2Text/`.

### Step 1: Load data and inspect (optional)

Download the dataset and print a summary (no training yet):

```bash
python scripts/data.py --skip-query-execution
```

- Uses `neo4j/text2cypher-2024v1` and the **entire** train/test by default.
- **To limit rows** (e.g. quick run): add `--limit 10` (or any number).
- `--skip-query-execution` skips running Cypher against Neo4j (recommended if you only need the encoder pipeline).

### Step 2: Compute IDF scores

IDF is computed from the **training** Cypher only. Default: **full train split**.

```bash
python scripts/compute_idf_text2cypher.py
```

- **Output:** `data/models/idf_scores_text2cypher.json`
- **To limit rows:** add `--limit 10` (or any number) for a quick run.
- Requires HuggingFace login (HF_TOKEN or `--hf-token-path`).

### Step 3: Train CSEncoder

Train the hybrid encoder using the same HF train split and the IDF file from Step 2. Default: **full train split**.

```bash
python scripts/train_csencoder.py --epochs 5 --batch-size 32
```

- **Inputs:** HF train split, `data/models/idf_scores_text2cypher.json`
- **Outputs:**
  - `data/models/csencoder_model.pt`
  - `data/models/token_to_id.json`
- **To limit rows:** add `--limit 10` for a quick test.
- For full dataset, use `--epochs` (e.g. 5–10) and `--batch-size` (e.g. 32) as above.

### Step 4 (optional): Embed and similarity

After Steps 1–3, embed the HF neo4j dataset into ChromaDB and compute top-k similarity:

```bash
python scripts/embed.py --limit 50
python scripts/similarity.py
```

Omit `--limit` for the full dataset. Uses `data/models/idf_scores_text2cypher.json`, `data/models/csencoder_model.pt`, and `data/models/token_to_id.json` from the main pipeline.

### Step 5: Cypher2Text (LLM)

`cypher2text_cot.py` translates Cypher → natural language using an LLM and (optionally) similar examples and pruned schemas. It has its own defaults (e.g. paths to util2/util6); override as needed:

```bash
python scripts/cypher2text_cot.py --help
```

Run with your chosen model and paths (e.g. `--output-dir`, `--similarity-json`, schema dirs).

### Step 6: Text2Cypher (LLM)

Takes the JSONL produced by Cypher2Text and converts natural language back to Cypher (e.g. for round-trip or evaluation):

```bash
python scripts/text2cypher_openai.py --in-jsonl <cypher2text_output.jsonl> --out-jsonl <output.jsonl> --dry-run
```

Use `--dry-run` to check prompts without API calls.

---

## 4. Configuration and Paths

- **Central config:** `data/setup/config.py` defines:
  - `REPO_ROOT`, `DATA_DIR`, `MODELS_DIR`, `RESULTS_DIR`, `CHROMADB_DIR`
  - `DEFAULT_HF_DATASET_NAME` = `neo4j/text2cypher-2024v1`
  - `DEFAULT_HF_TOKEN_PATH` (override via env or CLI)
- Scripts that need the HF dataset or token import these or take `--dataset-name` and `--hf-token-path`.

---

## 5. Expected Generated Artifacts

After a full run of the HF pipeline (Steps 1–3), you should have:

- `data/models/idf_scores_text2cypher.json`
- `data/models/csencoder_model.pt`
- `data/models/token_to_id.json`

Optional (embed + similarity):

- `data/chromadb/text2cypher/train`, `data/chromadb/text2cypher/test`
- `data/results/similarity/similarity_results.json`, `similarity_results.txt`

These paths are ignored by git (see `.gitignore`) so the repository stays clean for submission; reviewers re-run the pipeline to generate them.

---

## 6. Limiting data size (quick runs)

By default, all scripts use the **entire** dataset. To run on a subset (e.g. for debugging or quick checks), pass **`--limit N`** in the terminal:

| Script | Example |
|--------|--------|
| `data.py` | `python scripts/data.py --limit 10 --skip-query-execution` |
| `compute_idf_text2cypher.py` | `python scripts/compute_idf_text2cypher.py --limit 10` |
| `train_csencoder.py` | `python scripts/train_csencoder.py --limit 10 --epochs 2 --batch-size 4` |
| `embed.py` | `python scripts/embed.py --limit 50` |
| `similarity.py` | `python scripts/similarity.py` |

Omit `--limit` for the full dataset.

### Minimal quick run (10 rows)

```bash
python scripts/data.py --limit 10 --skip-query-execution
python scripts/compute_idf_text2cypher.py --limit 10
python scripts/train_csencoder.py --limit 10 --epochs 2 --batch-size 4
```

Then check that `data/models/` contains `idf_scores_text2cypher.json`, `csencoder_model.pt`, and `token_to_id.json`.

---

## 7. Troubleshooting

- **HuggingFace token:** If you see permission or gated-dataset errors, set `HF_TOKEN` or pass `--hf-token-path` to `data.py`, `compute_idf_text2cypher.py`, and `train_csencoder.py`.
- **CUDA:** Training uses GPU if available; otherwise CPU. Set `CUDA_VISIBLE_DEVICES` if needed.
- **Embed & similarity:** `embed.py` and `similarity.py` use the HF neo4j dataset; run after the main pipeline (Steps 1–3).
