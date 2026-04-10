# CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation

KDD 2026 code submission for Cypher-to-Text and Text-to-Cypher translation.

## Quick Repro (Reviewer Path)

```bash
# from repository root
make install
export HF_TOKEN=<your_hf_token>
make quick
```

Expected artifacts:
- `data/models/idf_scores_text2cypher.json`
- `data/models/csencoder_model.pt`
- `data/models/token_to_id.json`

For full instructions and all optional stages, see `RUN.md`.

## What Is Included

- Core model: `csencoder/model/hybrid_cypher_graph_encoder.py`
- Preprocessing: `csencoder/preprocessing/graph_builder.py`, `csencoder/preprocessing/rarity_calc.py`
- Training/inference scripts under `scripts/`

No generated model outputs, result files, or credentials are committed.

## Main Pipeline

1. `scripts/data.py`: load/inspect HF dataset.
2. `scripts/compute_idf_text2cypher.py`: compute IDF from train split.
3. `scripts/train_csencoder.py`: train encoder.
4. Optional retrieval: `scripts/embed.py`, `scripts/similarity.py`.
5. Optional generation: `scripts/cypher2text_cot.py`, `scripts/text2cypher_openai.py`.

## Environment

- Python: 3.10+
- Quick run: CPU is sufficient
- Full run: GPU recommended
- Dataset: `neo4j/text2cypher-2024v1` on Hugging Face

## Citation

```bibtex
@misc{csencoder_kdd2026,
  title={CSEncoder: Hybrid Cypher Structure Encoder for Graph Query Translation},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
