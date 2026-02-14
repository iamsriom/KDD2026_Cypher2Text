#!/usr/bin/env python3
"""
Data pipeline for KDD2026_Cypher2Text: load neo4j/text2cypher-2024v1, summary, optional Neo4j execution.

Uses only HuggingFace Hub dataset neo4j/text2cypher-2024v1 (like util6/LLMTranslations/data.py).
Summary includes distribution by database_reference_alias.
Optional: run Cypher against Neo4j Labs demo DBs.

Run from repo root: python scripts/data.py [options]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import pandas as pd

try:
    from typing import LiteralString
except ImportError:
    LiteralString = str  # type: ignore

# Repo root and config
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.setup.config import (
    DATA_DIR,
    DEFAULT_HF_DATASET_NAME,
    DEFAULT_HF_TOKEN_PATH,
    DEFAULT_NEO4J_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("kdd2026_data")
DEFAULT_QUERY_LIMIT = 5
DEFAULT_SLEEP_SECONDS = 0.0
QUERY_RUN_EXCEPTION = "query_run_exception"
NEO4J_DEMO_URI = "neo4j+s://demo.neo4jlabs.com"
JSONABLE_TYPES = (dict, list, tuple, str, int, float, bool, type(None))


def configure_logging(verbose: bool = False) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_virtualenv() -> None:
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        raise RuntimeError(
            "Run from the project virtualenv. Activate it first, e.g. source venv/bin/activate"
        )


def read_hf_token(token_path: Path) -> str:
    if not token_path.exists():
        raise FileNotFoundError(f"Hugging Face token file not found: {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Hugging Face token file {token_path} is empty")
    return token


def ensure_hf_login(token_path: Path) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        token = read_hf_token(token_path)
        os.environ["HF_TOKEN"] = token
    from huggingface_hub import login as hf_login
    hf_login(token=token, add_to_git_credential=False)
    LOGGER.debug("Authenticated with Hugging Face hub")


# ---------- HuggingFace Hub (neo4j/text2cypher-2024v1) ----------

def load_hf_split(dataset_name: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset
    LOGGER.info("Loading split '%s' from '%s'", split, dataset_name)
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    LOGGER.info("Loaded %s rows for split '%s'", len(df), split)
    return df


# ---------- Neo4j (for Hub dataset) ----------

class Neo4jConnector:
    def __init__(
        self,
        db_uri: str,
        db_username: str,
        db_password: str,
        db_name: str = "neo4j",
        neo4j_timeout_in_seconds: Optional[int] = None,
    ) -> None:
        self.db_uri = db_uri
        self.db_username = db_username
        self.db_password = db_password
        self.db_name = db_name
        self.neo4j_timeout_in_seconds = neo4j_timeout_in_seconds
        self.logger = logging.getLogger(f"Neo4jConnector[{db_name}]")
        self.driver = self._init_driver()

    def _init_driver(self):
        from neo4j import GraphDatabase
        self.logger.debug("Initialising Neo4j driver for %s", self.db_uri)
        return GraphDatabase.driver(self.db_uri, auth=(self.db_username, self.db_password))

    def _run_query(self, session, cypher_query: str, params: Optional[dict] = None) -> list[dict[str, Any]]:
        from neo4j import Query
        try:
            query = Query(cast(LiteralString, cypher_query), timeout=self.neo4j_timeout_in_seconds)
            result = session.run(query=query, parameters=params).data()
        except Exception as exc:
            self.logger.warning("Query run exception: %s", exc)
            result = [{QUERY_RUN_EXCEPTION: type(exc).__name__}]
        return result

    def execute_query(
        self,
        cypher_query: str,
        params: Optional[dict] = None,
        version_suffix: str = "CYPHER 5",
    ) -> list[dict[str, Any]]:
        try:
            versioned = f"{version_suffix}\n{cypher_query}"
            with self.driver.session(database=self.db_name) as session:
                return self._run_query(session=session, cypher_query=versioned, params=params)
        except Exception as exc:
            self.logger.warning("Exception in execute_query: %s", exc)
            raise


class DatabaseReferenceType(str, Enum):
    NEO4JLABS_DEMO_DB_MOVIES = "neo4jlabs_demo_db_movies"
    NEO4JLABS_DEMO_DB_COMPANIES = "neo4jlabs_demo_db_companies"
    NEO4JLABS_DEMO_DB_NETWORK = "neo4jlabs_demo_db_network"
    NEO4JLABS_DEMO_DB_RECOMMENDATIONS = "neo4jlabs_demo_db_recommendations"
    NEO4JLABS_DEMO_DB_BLUESKY = "neo4jlabs_demo_db_bluesky"
    NEO4JLABS_DEMO_DB_BUZZOVERFLOW = "neo4jlabs_demo_db_buzzoverflow"
    NEO4JLABS_DEMO_DB_FINCEN = "neo4jlabs_demo_db_fincen"
    NEO4JLABS_DEMO_DB_GAMEOFTHRONES = "neo4jlabs_demo_db_gameofthrones"
    NEO4JLABS_DEMO_DB_GRANDSTACK = "neo4jlabs_demo_db_grandstack"
    NEO4JLABS_DEMO_DB_NEOFLIX = "neo4jlabs_demo_db_neoflix"
    NEO4JLABS_DEMO_DB_NORTHWIND = "neo4jlabs_demo_db_northwind"
    NEO4JLABS_DEMO_DB_OFFSHORELEAKS = "neo4jlabs_demo_db_offshoreleaks"
    NEO4JLABS_DEMO_DB_OPENSTREETMAP = "neo4jlabs_demo_db_openstreetmap"
    NEO4JLABS_DEMO_DB_STACKOVERFLOW2 = "neo4jlabs_demo_db_stackoverflow2"
    NEO4JLABS_DEMO_DB_TWITCH = "neo4jlabs_demo_db_twitch"
    NEO4JLABS_DEMO_DB_TWITTER = "neo4jlabs_demo_db_twitter"
    NEO4JLABS_DEMO_DB_STACKOVERFLOW = "neo4jlabs_demo_db_stackoverflow"


DB_REFERENCE_TO_DBNAME: dict[DatabaseReferenceType, str] = {
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_MOVIES: "movies",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_COMPANIES: "companies",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_NETWORK: "network",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_RECOMMENDATIONS: "recommendations",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_BLUESKY: "neo4jlabs_demo_db_bluesky",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_BUZZOVERFLOW: "buzzoverflow",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_FINCEN: "fincen",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_GAMEOFTHRONES: "gameofthrones",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_GRANDSTACK: "grandstack",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_NEOFLIX: "neoflix",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_NORTHWIND: "northwind",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_OFFSHORELEAKS: "offshoreleaks",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_OPENSTREETMAP: "openstreetmap",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_STACKOVERFLOW2: "stackoverflow2",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_TWITCH: "twitch",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_TWITTER: "twitter",
    DatabaseReferenceType.NEO4JLABS_DEMO_DB_STACKOVERFLOW: "stackoverflow",
}


class Neo4JLabsDemoDatabaseReference:
    def __init__(self, db_alias: str | DatabaseReferenceType, neo4j_timeout_in_seconds: Optional[int] = None):
        try:
            enum_alias = db_alias if isinstance(db_alias, DatabaseReferenceType) else DatabaseReferenceType(db_alias)
        except ValueError as exc:
            raise KeyError(f"Unknown Neo4j Labs demo database alias: {db_alias}") from exc
        self.db_reference_alias = enum_alias
        self.db_name = DB_REFERENCE_TO_DBNAME[enum_alias]
        self.neo4j_uri = NEO4J_DEMO_URI
        self.db_connector = Neo4jConnector(
            db_uri=self.neo4j_uri,
            db_username=self.db_name,
            db_password=self.db_name,
            db_name=self.db_name,
            neo4j_timeout_in_seconds=neo4j_timeout_in_seconds,
        )


def clean_cypher_query(query: str) -> str:
    query = query.replace("\n", " ").replace("\r", " ").replace("\\n", " ")
    return " ".join(query.split()).strip()


def _sorted_nested_dict(input_dict: Any) -> Any:
    if isinstance(input_dict, dict):
        return {k: _sorted_nested_dict(v) for k, v in sorted(input_dict.items())}
    return input_dict


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_jsonable(v) for v in obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    try:
        from neo4j.time import Date as Neo4jDate
        from neo4j.time import DateTime as Neo4jDateTime
        if isinstance(obj, (Neo4jDate, Neo4jDateTime)):
            return obj.isoformat()
    except ImportError:
        pass
    if isinstance(obj, JSONABLE_TYPES):
        return obj
    raise TypeError(f"Object of type {type(obj)} is not JSON-serializable")


def _convert_dict_to_str(cypher_output: list[dict[str, Any]]) -> str:
    if not cypher_output:
        return "{}"
    return str(_sorted_nested_dict(cypher_output))


def convert_outputs_into_string(cypher_outputs: Sequence[Any]) -> list[str]:
    out = to_jsonable(cypher_outputs)
    if not isinstance(out, (list, tuple)):
        return [str(out)]
    return [_convert_dict_to_str(x) if isinstance(x, list) else str(x) for x in out]


def run_cypher_queries(
    cypher_queries: Sequence[str],
    db_connectors: Sequence[Neo4jConnector],
    sleep: Optional[float] = DEFAULT_SLEEP_SECONDS,
    clean_cypher: bool = False,
) -> list[list[dict[str, Any]] | str]:
    if len(cypher_queries) != len(db_connectors):
        raise ValueError("Number of queries must match number of connectors")
    outputs: list[list[dict[str, Any]] | str] = []
    for idx, (query, connector) in enumerate(zip(cypher_queries, db_connectors)):
        try:
            q = clean_cypher_query(query) if clean_cypher else query
            result = connector.execute_query(cypher_query=q)
            outputs.append(result)
            if sleep and idx < len(cypher_queries) - 1:
                time.sleep(sleep)
        except Exception as exc:
            LOGGER.warning("Query run exception: %s", exc)
            outputs.append(f"{{'{QUERY_RUN_EXCEPTION}':'{type(exc).__name__}'}}")
    return outputs


@dataclasses.dataclass
class CypherExecution:
    split: str
    row_index: int
    database_reference_alias: str
    cypher: str
    outputs_raw: list[dict[str, Any]] | str
    outputs_as_string: list[str]


def build_connectors_for_aliases(
    aliases: Iterable[str],
    neo4j_timeout_in_seconds: Optional[int] = DEFAULT_NEO4J_TIMEOUT_SECONDS,
) -> dict[str, Neo4jConnector]:
    connectors: dict[str, Neo4jConnector] = {}
    for alias in aliases:
        if alias not in connectors:
            connectors[alias] = Neo4JLabsDemoDatabaseReference(alias, neo4j_timeout_in_seconds).db_connector
    return connectors


def execute_queries_for_split(
    split_name: str,
    df: pd.DataFrame,
    limit: int,
    clean_cypher: bool,
    sleep: Optional[float],
) -> list[CypherExecution]:
    if "database_reference_alias" not in df.columns or "cypher" not in df.columns:
        LOGGER.warning(
            "Split '%s' missing required columns. Available: %s",
            split_name,
            list(df.columns),
        )
        return []
    executable_df = df[df["database_reference_alias"].notna()].head(limit)
    if executable_df.empty:
        LOGGER.info("No executable rows for split '%s'", split_name)
        return []
    connectors_map = build_connectors_for_aliases(executable_df["database_reference_alias"].unique())
    db_connectors = [connectors_map[a] for a in executable_df["database_reference_alias"]]
    outputs = run_cypher_queries(
        cypher_queries=executable_df["cypher"],
        db_connectors=db_connectors,
        sleep=sleep,
        clean_cypher=clean_cypher,
    )
    executions: list[CypherExecution] = []
    for (row_index, row), output in zip(executable_df.iterrows(), outputs):
        output_strings = convert_outputs_into_string(output if isinstance(output, list) else [output])
        executions.append(
            CypherExecution(
                split=split_name,
                row_index=row_index,
                database_reference_alias=row["database_reference_alias"],
                cypher=row["cypher"],
                outputs_raw=output,
                outputs_as_string=output_strings,
            )
        )
    return executions


# ---------- Summary ----------

def print_data_distribution(dataframes: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 80)
    print("DATASET SUMMARY (KDD2026 pipeline)")
    print("=" * 80)
    for split_name, df in dataframes.items():
        if df.empty:
            print(f"\n{split_name.upper()}: (empty)")
            continue
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Dtypes: {dict(df.dtypes)}")
        missing = df.isnull().sum()
        if missing.any():
            print(f"\n  Missing:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count} ({100 * count / len(df):.2f}%)")
        else:
            print(f"\n  Missing: None")
        if "graph" in df.columns:
            g = df["graph"].value_counts()
            print(f"\n  By graph ({len(g)}): {dict(g.head(10))}" + (" ..." if len(g) > 10 else ""))
        if "database_reference_alias" in df.columns:
            db = df["database_reference_alias"].value_counts()
            print(f"\n  By database ({len(db)}): {dict(db)}")
        print(f"\n  First 3 rows:")
        print(df.head(3).to_string())
    print("\n" + "=" * 80 + "\n")


# ---------- CLI ----------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KDD2026 data pipeline: load neo4j/text2cypher-2024v1, summary, optional Neo4j execution."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_HF_DATASET_NAME,
        help="HuggingFace dataset (default: neo4j/text2cypher-2024v1)",
    )
    parser.add_argument(
        "--hf-token-path",
        type=Path,
        default=DEFAULT_HF_TOKEN_PATH,
        help="Hugging Face API token file",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help="Max rows to execute per split when not skipping execution",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds between Neo4j query executions",
    )
    parser.add_argument(
        "--clean-cypher",
        action="store_true",
        help="Normalise whitespace in Cypher before execution",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load only test split",
    )
    parser.add_argument(
        "--skip-query-execution",
        action="store_true",
        help="Do not run Cypher against Neo4j (summary only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows per split when loading (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "dataset",
        help="Directory to save loaded splits (CSV). Default: data/dataset",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    ensure_virtualenv()
    ensure_hf_login(Path(args.hf_token_path))

    dataframes: dict[str, pd.DataFrame] = {}
    splits = ["test"] if args.skip_train else ["train", "test"]
    for split in splits:
        df = load_hf_split(args.dataset_name, split)
        if args.limit is not None:
            df = df.head(args.limit)
        dataframes[split] = df

    print_data_distribution(dataframes)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, df in dataframes.items():
            if df.empty:
                continue
            path = args.out_dir / f"{split_name}.csv"
            df.to_csv(path, index=False)
            LOGGER.info("Wrote %s", path)

    if args.skip_query_execution:
        LOGGER.info("Skipping query execution as requested.")
        return

    if "test" not in dataframes or dataframes["test"].empty:
        LOGGER.warning("No test split; skipping execution.")
        return

    executions = execute_queries_for_split(
        split_name="test",
        df=dataframes["test"],
        limit=args.query_limit,
        clean_cypher=args.clean_cypher,
        sleep=args.sleep,
    )
    if not executions:
        LOGGER.info("No query executions performed.")
        return
    for ex in executions:
        LOGGER.info(
            "Split=%s Row=%d Alias=%s\nCypher: %s\nOutputs: %s\n",
            ex.split, ex.row_index, ex.database_reference_alias, ex.cypher, ex.outputs_as_string,
        )


if __name__ == "__main__":
    main()
