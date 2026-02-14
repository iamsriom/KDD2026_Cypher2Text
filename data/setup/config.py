"""
Data setup configuration for KDD2026_Cypher2Text.

Central path constants and dataset defaults. Scripts under scripts/ and data/setup/
should use these so the pipeline stays consistent with the KDD folder layout.
"""

from pathlib import Path

# Repo root (parent of scripts/ and data/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data directory under repo
DATA_DIR = REPO_ROOT / "data"

# Default dataset source (HuggingFace Hub)
DEFAULT_HF_DATASET_NAME = "neo4j/text2cypher-2024v1"  # HuggingFace Hub neo4j Text2Cypher

# Outputs under data/
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = DATA_DIR / "models"
CHROMADB_DIR = DATA_DIR / "chromadb"
PRUNED_SCHEMAS_DIR = DATA_DIR / "pruned_schemas"

# Credentials (override via env or CLI)
DEFAULT_HF_TOKEN_PATH = Path("/home/ubuntu/Huggingface_api.txt")
DEFAULT_NEO4J_TIMEOUT_SECONDS = 60 * 10  # 10 minutes
