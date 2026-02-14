# CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation

**KDD 2026 Code Submission.** This repository contains the implementation of **CSEncoder (Cypher Structure Encoder)**, a hybrid embedding method for Cypher queries that combines sequential semantics, directed structural information, Magnetic Laplacian positional encodings, and IDF-weighted attention.

## How to run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set HuggingFace token (for dataset neo4j/text2cypher-2024v1)
export HF_TOKEN=your_token   # or use --hf-token-path

# 3. Pipeline (quick run with --limit 10; omit for full run)
python scripts/data.py --skip-query-execution
python scripts/compute_idf_text2cypher.py
python scripts/train_csencoder.py --epochs 5 --batch-size 32
```

**Full instructions:** [RUN.md](RUN.md)

---

## For KDD Reviewers

- **Reproducibility:** All steps to reproduce the pipeline are in **[RUN.md](RUN.md)**.
- **No pre-committed artifacts:** No pre-trained weights or generated outputs are committed. Run the three main scripts (data → IDF → train) to produce `data/models/` outputs.
- **Quick run:** Use the commands below for a minimal run (~5–15 minutes on CPU):
  ```bash
  python scripts/data.py --limit 10 --skip-query-execution
  python scripts/compute_idf_text2cypher.py --limit 10
  python scripts/train_csencoder.py --limit 10 --epochs 2 --batch-size 4
  ```
- **Resource Availability:** Upon acceptance, a DOI will be provided. See the "Resource Availability" section below.

## Resource Availability

Upon acceptance, the source code will be made publicly available in an archival repository with a persistent DOI. The camera-ready paper will include the Resource Availability statement per KDD 2026 artifact badging guidelines.

---

## Environment Requirements

- **Python:** 3.10+
- **Hardware:** CPU sufficient for quick runs; GPU recommended for full training
- **Disk:** ~2GB for dependencies and model outputs
- **API access:** HuggingFace token (for gated dataset `neo4j/text2cypher-2024v1`); optional: OpenAI/Anthropic for translation scripts

---

## Overview

CSEncoder is designed for **Cypher-to-Text** and **Text-to-Cypher** translation tasks. The system:

1. **Trains a hybrid embedding model** (CSEncoder) using contrastive learning on the HuggingFace dataset `neo4j/text2cypher-2024v1`
2. **Optionally generates embeddings** and stores them in ChromaDB for similarity search
3. **Translates Cypher → natural language** (Cypher2Text) and **natural language → Cypher** (Text2Cypher) using LLMs and optional retrieval

**Detailed run instructions and how each script works:** see **[RUN.md](RUN.md)**.

## Repository Structure

```
KDD2026_Cypher2Text/
├── README.md                          # This file
├── RUN.md                             # How to run the code (detailed)
├── requirements.txt
├── csencoder/                         # Core library (model + preprocessing + embedder)
│   ├── embedder.py
│   ├── model/
│   │   └── hybrid_cypher_graph_encoder.py
│   └── preprocessing/
│       ├── graph_builder.py
│       └── rarity_calc.py
├── data/
│   ├── setup/config.py                # Paths and dataset defaults
│   ├── models/                        # Generated: IDF JSON, csencoder checkpoints
│   ├── chromadb/                      # Generated: ChromaDB from embed.py
│   └── results/                       # Generated: similarity/translation outputs
└── scripts/                           # Runnable entrypoints
    ├── data.py                        # Load HF dataset (neo4j/text2cypher-2024v1)
    ├── compute_idf_text2cypher.py     # IDF from HF train split
    ├── train_csencoder.py             # Train CSEncoder on HF data
    ├── embed.py                       # Embed HF neo4j dataset → ChromaDB
    ├── similarity.py                  # Top-k similarity from ChromaDB
    ├── cypher2text_cot.py             # Cypher → NL (LLM + CoT)
    └── text2cypher_openai.py          # NL → Cypher (LLM)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **HuggingFace access:** Accept the dataset terms at [neo4j/text2cypher-2024v1](https://huggingface.co/datasets/neo4j/text2cypher-2024v1), then set `HF_TOKEN` or pass `--hf-token-path` to the scripts that need it.

3. **Run the pipeline** (from repository root). Default uses the full dataset; add `--limit N` for a quick run:

   ```bash
   python scripts/data.py --skip-query-execution
   python scripts/compute_idf_text2cypher.py
   python scripts/train_csencoder.py --epochs 5 --batch-size 32
   ```

4. **Expected outputs** after Steps 1–3:
   - `data/models/idf_scores_text2cypher.json`
   - `data/models/csencoder_model.pt`
   - `data/models/token_to_id.json`

For full commands, all CLI options (including `--limit`), embed/similarity, and Cypher2Text/Text2Cypher usage, see **[RUN.md](RUN.md)**.

## Key Components

### 1. CSEncoder (Hybrid Cypher Structure Encoder)

The core model architecture (`csencoder/model/hybrid_cypher_graph_encoder.py`) that combines:
- **Sequential semantics**: Token embeddings
- **Structural information**: Graph-biased attention with shortest-path-distance bias
- **Magnetic Laplacian PE**: Positional encodings from directed graph structure
- **IDF-weighted attention**: Rarity bias for discriminative tokens

### 2. Graph Builder

Preprocessing module (`csencoder/preprocessing/graph_builder.py`) that:
- Tokenizes Cypher queries
- Builds directed graphs from AST or regex patterns
- Computes shortest-path-distance matrices
- Generates Magnetic Laplacian positional encodings

### 3. IDF Calculator

Rarity calculation (`csencoder/preprocessing/rarity_calc.py`) that:
- Computes Inverse Document Frequency (IDF) scores
- Used for IDF-weighted attention mechanism

### 4. Embedding & Storage

- **HF pipeline**: `train_csencoder.py` saves `data/models/csencoder_model.pt` and `data/models/token_to_id.json`.
- **Embed & similarity** (`scripts/embed.py`, `scripts/similarity.py`): embed `neo4j/text2cypher-2024v1` into ChromaDB and compute top-k similarity.

### 5. Similarity Search

Similarity computation (`scripts/similarity.py`) that:
- Finds top-k similar training examples
- Uses cosine similarity on embeddings

### 6. Translation Steps (LLM-Dependent)

Cypher2Text and Text2Cypher generation require LLM API access (e.g., OpenAI). For reviewability, both scripts support `--dry-run` to verify prompt construction and retrieved example injection without making API calls.

## Evaluation Metrics

The system supports:
- **BLEU-4**: N-gram overlap for translation quality
- **BERTScore**: Semantic similarity using DeBERTa
- **LLM-as-Judge**: Semantic equivalence using Claude
- **Exact Match**: String-level Cypher comparison
- **Execution Accuracy**: Result set comparison on Neo4j

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{csencoder2026,
  title={CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation},
  author={[Authors]},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2026}
}
```

## License

MIT License. See `LICENSE`.
