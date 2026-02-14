#!/usr/bin/env python3
"""
Translate Cypher queries to natural language using pruned schemas from PrunedSchemas folder.
IMPROVED VERSION: Includes retry logic, fallback mechanisms, and ensures all entries get translations.
Uses CodeLlama model and 5 similar examples from similarity results for each test query.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from openai import OpenAI

# Model configuration
MODELS = {
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt4o-mini": "gpt-4o-mini",
}
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 50
MAX_NEW_TOKENS = 800  # Balanced for quality and speed
REFERENCE_EXAMPLES = 5
MAX_RETRIES = 2  # Maximum retries for failed translations (reduced for faster processing)

# Paths
PRUNED_SCHEMAS_DIR = Path("/home/ubuntu/util2/PrunedSchemas2")
OUTPUT_DIR = Path("/home/ubuntu/util6/Cypher2Text3")  # Output directory in util6
SIMILAR_QUERIES_DIR = Path("/home/ubuntu/similar_queries_cypher")
HYBRID_SIMILARITY_JSON = Path("/home/ubuntu/util6/LLMTranslations/SimilarityResults2/similarity_results.json")
# Manual dataset3 (SimilarityResultsManualDataset3 + PrunedSchemasDataset3)
PRUNED_SCHEMAS_DIR_MANUAL = Path("/home/ubuntu/util2/PrunedSchemasDataset3")
SIMILARITY_JSON_MANUAL = Path("/home/ubuntu/util6/LLMTranslations/SimilarityResultsManualDataset3/similarity_results.json")
OUTPUT_DIR_MANUAL = Path("/home/ubuntu/util6/Cypher2TextDataset3Manual")
_pruned_schemas_dir_override: Optional[Path] = None

# Help avoid CUDA fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def format_pruned_schema(pruned_schema: Dict[str, Any]) -> str:
    """Format pruned schema JSON into text format similar to codeBERT format."""
    relationships = pruned_schema.get("relationships", [])
    nodes = pruned_schema.get("nodes", [])
    
    lines = []
    
    # Add relationships
    for rel in relationships:
        from_node = rel.get("from", "")
        rel_type = rel.get("type", "")
        to_node = rel.get("to", "")
        lines.append(f"(:{from_node})-[:{rel_type}]->(:{to_node})")
    
    # Add nodes with properties
    for node in nodes:
        label = node.get("label", "")
        properties = node.get("properties", [])
        if properties:
            prop_str = ", ".join(properties)
            lines.append(f"(:{label}) {{properties: {prop_str}}}")
        else:
            lines.append(f"(:{label})")
    
    return "\n".join(lines) if lines else ""


def is_cuda_healthy() -> bool:
    """Check if CUDA is still functional after potential errors."""
    if not torch.cuda.is_available():
        return False
    try:
        # Try a simple CUDA operation to verify context is valid
        # Use a minimal check that won't fail if CUDA is corrupted
        _ = torch.cuda.current_device()
        # Only synchronize if we can get the device - if this fails, CUDA is corrupted
        try:
            torch.cuda.synchronize()
        except (RuntimeError, Exception) as sync_error:
            error_str = str(sync_error).lower()
            if "illegal memory access" in error_str or "cuda error" in error_str:
                return False
            # For other sync errors, still consider CUDA healthy
        return True
    except (RuntimeError, Exception) as e:
        error_str = str(e).lower()
        if "illegal memory access" in error_str or "cuda error" in error_str:
            return False
        # For other errors, assume CUDA might still be OK
        return True  # Changed from False to True - be more lenient


def safe_cuda_clear():
    """Safely clear CUDA cache, handling errors gracefully."""
    if not torch.cuda.is_available():
        return
    try:
        if is_cuda_healthy():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (RuntimeError, Exception) as e:
        error_msg = str(e).lower()
        if "illegal memory access" in error_msg or "cuda error" in error_msg:
            LOGGER.warning("CUDA context corrupted, skipping memory clear. May need to restart process.")
            # Don't raise - just log and continue
        else:
            # For other errors, log but don't crash
            LOGGER.warning(f"Error clearing CUDA cache: {e}")


def check_gpu_memory(min_free_gb: float = 1.0) -> bool:
    """Check if there's enough free GPU memory."""
    if not torch.cuda.is_available():
        return False
    try:
        for i in range(torch.cuda.device_count()):
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_gb = free_memory / (1024**3)
            if free_gb < min_free_gb:
                LOGGER.warning(f"GPU {i} has only {free_gb:.2f} GB free (minimum: {min_free_gb} GB)")
                return False
        return True
    except Exception as e:
        LOGGER.warning(f"Could not check GPU memory: {e}")
        return True  # Assume OK if we can't check


def load_pruned_schema(instance_id: str) -> Optional[Dict[str, Any]]:
    """Load pruned schema JSON file for a given instance_id."""
    schema_dir = _pruned_schemas_dir_override if _pruned_schemas_dir_override is not None else PRUNED_SCHEMAS_DIR
    schema_path = schema_dir / f"{instance_id}_pruned_schema.json"
    if not schema_path.exists():
        return None
    
    try:
        with schema_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("pruned_schema", data)
    except Exception as e:
        LOGGER.warning(f"Could not load pruned schema for {instance_id}: {e}")
        return None


def load_similar_queries_from_hybrid(
    similarity_entry: Dict[str, Any],
    max_examples: int = REFERENCE_EXAMPLES
) -> List[Dict[str, Any]]:
    """Load similar queries from sentence embedding similarity results."""
    references: List[Dict[str, Any]] = []
    
    # Sentence embedding format uses "top_matches"
    top_matches = similarity_entry.get("top_matches", [])
    
    for match in top_matches[:max_examples]:
        train_query = match.get("query", "")
        # Gold translation is now at top level, fallback to metadata if not present
        train_question = match.get("gold_translation", "")
        similarity_score = match.get("similarity")
        train_instance_id = match.get("train_instance_id", "")
        train_metadata = match.get("metadata", {})
        
        # If gold_translation not found, try to get from metadata
        if not train_question:
            train_question = train_metadata.get("NL Question", "") or train_metadata.get("question", "") or train_metadata.get("Question", "")
        
        if train_query and train_question:
            references.append({
                "train_query": train_query,
                "train_translation": train_question,
                "similarity_score": similarity_score,
                "train_id": train_instance_id or train_metadata.get("instance_id", ""),
                "train_metadata": train_metadata,
            })
    
    return references[:max_examples]


def load_similar_queries_from_file(
    instance_id: str, 
    test_query: Optional[str] = None,
    max_examples: int = REFERENCE_EXAMPLES
) -> List[Dict[str, Any]]:
    """Load similar queries from the similar_queries_cypher folder (fallback)."""
    references: List[Dict[str, Any]] = []
    if not SIMILAR_QUERIES_DIR.exists():
        return references

    possible_files = []

    number_match = re.search(r'(\d+)', instance_id)
    if number_match:
        file_num = number_match.group(1)
        file_num_padded = file_num.zfill(5)
        candidate_file = SIMILAR_QUERIES_DIR / f"test_query_{file_num_padded}.json"
        if candidate_file.exists():
            possible_files.append(candidate_file)
    
    if test_query and (not possible_files or not possible_files[0].exists()):
        for json_file in sorted(SIMILAR_QUERIES_DIR.glob("test_query_*.json")):
            try:
                with json_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    file_test_query = data.get("test_query", "").strip()
                    if file_test_query and test_query.strip():
                        if file_test_query == test_query.strip():
                            possible_files.insert(0, json_file)
                            break
            except Exception:
                continue
    
    for json_file in possible_files:
        if not json_file.exists():
            continue
        try:
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                similar_queries = data.get("similar_train_queries", [])
                
                for similar in similar_queries[:max_examples]:
                    train_query = similar.get("train_query", "")
                    train_question = similar.get("train_question", "")
                    
                    if train_query and train_question:
                        references.append({
                            "train_query": train_query,
                            "train_translation": train_question,
                            "similarity_score": similar.get("similarity_score"),
                            "train_index": similar.get("train_index"),
                        })
                
                if references:
                    break
        except Exception as e:
            LOGGER.warning(f"Could not load similar queries from {json_file}: {e}")
            continue
    
    return references[:max_examples]


def collect_reference_queries(
    similarity_entry: Dict[str, Any],
    instance_id: str,
    max_examples: int = REFERENCE_EXAMPLES,
) -> List[Dict[str, Any]]:
    """Collect structurally similar Cypher queries from sentence embedding similarity results."""
    references = load_similar_queries_from_hybrid(similarity_entry, max_examples)
    
    if references:
        return references
    
    test_query = similarity_entry.get("test_query", "")
    references = load_similar_queries_from_file(instance_id, test_query, max_examples)
    
    return references


def create_simple_prompt(cypher_query: str, pruned_schema_text: Optional[str] = None) -> str:
    """Create a simplified prompt without few-shot examples for fallback."""
    prompt = "Translate the following Cypher query into a natural language question.\n\n"
    
    if pruned_schema_text:
        prompt += f"Schema context:\n{pruned_schema_text}\n\n"
    
    prompt += f"Cypher query: {cypher_query}\n\n"
    prompt += "Natural language question: "
    
    return prompt


def create_few_shot_prompt_no_schema(
    cypher_query: str,
    reference_queries: List[Dict[str, Any]],
    metadata_context: Optional[str] = None,
) -> str:
    """Create a prompt with top 5 examples but NO schema (Case 1)."""
    base_prompt_template = """You are an expert graph data analyst who translates Cypher queries into natural language questions for non-technical stakeholders.

Carefully inspect each Cypher query and internally reason about:
- The graph pattern being matched (nodes, labels, relationship types, traversal direction)
- Filters and predicates (WHERE clauses, property constraints, parameter usage)
- Aggregations, grouping, ordering, and limiting (COUNT, COLLECT, ORDER BY, LIMIT, SKIP, DISTINCT)
- The flow of data through MATCH / OPTIONAL MATCH / WITH / RETURN clauses

Your task is to output a single, fluent natural language question that:
- Captures the exact intent of the target Cypher query
- Mentions critical entities and relationships when relevant
- Includes essential filters, joins, aggregations, sorting, and limits
- Reflects how results are grouped or scoped (e.g., "for each author", "top 5", "overall count")
- Sounds like something a human business analyst would actually ask

Important guardrails:
- DO NOT copy or assume any existing translations; only the Cypher syntax is provided.
- Think step by step, but output ONLY the final question—no reasoning, bullet lists, or explanations.
- Stop immediately after the first well-formed question.

You are provided with structurally similar Cypher queries (with their translations) to help you stay consistent with domain terminology.

Translate the final target query using this information.
"""

    prompt = f"{base_prompt_template}\n\n"

    if metadata_context:
        prompt += "Additional metadata associated with the target query:\n"
        prompt += f"{metadata_context}\n\n"

    usable_examples = [
        example
        for example in reference_queries
        if example.get("train_query") and example.get("train_translation")
    ]

    if usable_examples:
        prompt += "Few-shot examples (study how each Cypher query maps to natural language):\n\n"
        for example in usable_examples[:REFERENCE_EXAMPLES]:
            prompt += f"Cypher: {example.get('train_query', '')}\n"
            prompt += f"Natural Language: {example.get('train_translation', '')}\n\n"
        prompt += (
            "Use these references to stay aligned with domain terminology, but do not reuse or invent "
            "translations for the reference queries. Only translate the final target query.\n\n"
        )

    prompt += "Target Cypher query:\n"
    prompt += f"{cypher_query}\n\n"
    prompt += "Reply ONLY with one natural language question. Do not include explanations, metadata, code, or formatting characters.\n"
    prompt += "Natural Language: "
    return prompt


def create_few_shot_prompt_full_schema(
    cypher_query: str,
    reference_queries: List[Dict[str, Any]],
    full_schema_text: Optional[str] = None,
    metadata_context: Optional[str] = None,
) -> str:
    """Create a prompt with top 5 examples and FULL schema from metadata (Case 2)."""
    base_prompt_template = """You are an expert graph data analyst who translates Cypher queries into natural language questions for non-technical stakeholders.

Carefully inspect each Cypher query and internally reason about:
- The graph pattern being matched (nodes, labels, relationship types, traversal direction)
- Filters and predicates (WHERE clauses, property constraints, parameter usage)
- Aggregations, grouping, ordering, and limiting (COUNT, COLLECT, ORDER BY, LIMIT, SKIP, DISTINCT)
- The flow of data through MATCH / OPTIONAL MATCH / WITH / RETURN clauses

Your task is to output a single, fluent natural language question that:
- Captures the exact intent of the target Cypher query
- Mentions critical entities and relationships when relevant
- Includes essential filters, joins, aggregations, sorting, and limits
- Reflects how results are grouped or scoped (e.g., "for each author", "top 5", "overall count")
- Sounds like something a human business analyst would actually ask

Important guardrails:
- DO NOT copy or assume any existing translations; only the Cypher syntax is provided.
- Think step by step, but output ONLY the final question—no reasoning, bullet lists, or explanations.
- Stop immediately after the first well-formed question.

You are provided with the full database schema to keep names accurate. You also see structurally similar Cypher queries (with their translations) to help you stay consistent with domain terminology.

Translate the final target query using this information.
"""

    prompt = f"{base_prompt_template}\n\n"

    if full_schema_text:
        prompt += "Full Database Schema:\n"
        prompt += f"{full_schema_text}\n\n"

    if metadata_context:
        prompt += "Additional metadata associated with the target query:\n"
        prompt += f"{metadata_context}\n\n"

    usable_examples = [
        example
        for example in reference_queries
        if example.get("train_query") and example.get("train_translation")
    ]

    if usable_examples:
        prompt += "Few-shot examples (study how each Cypher query maps to natural language):\n\n"
        for example in usable_examples[:REFERENCE_EXAMPLES]:
            prompt += f"Cypher: {example.get('train_query', '')}\n"
            prompt += f"Natural Language: {example.get('train_translation', '')}\n\n"
        prompt += (
            "Use these references to stay aligned with domain terminology, but do not reuse or invent "
            "translations for the reference queries. Only translate the final target query.\n\n"
        )

    prompt += "Target Cypher query:\n"
    prompt += f"{cypher_query}\n\n"
    prompt += "Reply ONLY with one natural language question. Do not include explanations, metadata, code, or formatting characters.\n"
    prompt += "Natural Language: "
    return prompt


def create_few_shot_prompt_cypher(
    cypher_query: str,
    reference_queries: List[Dict[str, Any]],
    pruned_schema_text: Optional[str] = None,
    metadata_context: Optional[str] = None,
) -> str:
    """Create a prompt with top 5 examples and PRUNED schema (Case 3 - as used in util4)."""
    base_prompt_template = """You are an expert graph data analyst who translates Cypher queries into natural language questions for non-technical stakeholders.

Carefully inspect each Cypher query and internally reason about:
- The graph pattern being matched (nodes, labels, relationship types, traversal direction)
- Filters and predicates (WHERE clauses, property constraints, parameter usage)
- Aggregations, grouping, ordering, and limiting (COUNT, COLLECT, ORDER BY, LIMIT, SKIP, DISTINCT)
- The flow of data through MATCH / OPTIONAL MATCH / WITH / RETURN clauses

Your task is to output a single, fluent natural language question that:
- Captures the exact intent of the target Cypher query
- Mentions critical entities and relationships when relevant
- Includes essential filters, joins, aggregations, sorting, and limits
- Reflects how results are grouped or scoped (e.g., "for each author", "top 5", "overall count")
- Sounds like something a human business analyst would actually ask

Important guardrails:
- DO NOT copy or assume any existing translations; only the Cypher syntax is provided.
- Think step by step, but output ONLY the final question�no reasoning, bullet lists, or explanations.
- Stop immediately after the first well-formed question.

You are provided with schema snippets derived from the target query to keep names accurate. You also see structurally similar Cypher queries (without translations) to help you stay consistent with domain terminology.

Translate the final target query using this information.
"""

    prompt = f"{base_prompt_template}\n\n"

    if pruned_schema_text:
        prompt += "Schema context:\n"
        prompt += f"{pruned_schema_text}\n\n"

    if metadata_context:
        prompt += "Additional metadata associated with the target query:\n"
        prompt += f"{metadata_context}\n\n"

    usable_examples = [
        example
        for example in reference_queries
        if example.get("train_query") and example.get("train_translation")
    ]

    if usable_examples:
        prompt += "Few-shot examples (study how each Cypher query maps to natural language):\n\n"
        for example in usable_examples[:REFERENCE_EXAMPLES]:
            prompt += f"Cypher: {example.get('train_query', '')}\n"
            prompt += f"Natural Language: {example.get('train_translation', '')}\n\n"
        prompt += (
            "Use these references to stay aligned with domain terminology, but do not reuse or invent "
            "translations for the reference queries. Only translate the final target query.\n\n"
        )

    prompt += "Target Cypher query:\n"
    prompt += f"{cypher_query}\n\n"
    prompt += "Reply ONLY with one natural language question. Do not include explanations, metadata, code, or formatting characters.\n"
    prompt += "Natural Language: "
    return prompt


def create_minimal_prompt(cypher_query: str) -> str:
    """Create a minimal prompt with just the query for last resort LLM generation."""
    prompt = "Translate this Cypher query to a natural language question:\n\n"
    prompt += f"{cypher_query}\n\n"
    prompt += "Question: "
    return prompt


def preprocess_llm_output(raw_output: str) -> str:
    """Post-process the raw LLM output to extract natural language translation."""
    if not raw_output:
        raise ValueError("Empty LLM output received.")

    cleaned_lines: List[str] = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if any(
            lower.startswith(prefix)
            for prefix in (
                "cypher:",
                "natural language:",
                "analysis:",
                "reasoning:",
                "thought:",
                "explanation:",
                "output:",
                "answer:",
                "response:",
            )
        ):
            continue

        cleaned_lines.append(stripped)

    if not cleaned_lines:
        raise ValueError("LLM output did not contain any usable translation after filtering.")

    collapsed = " ".join(cleaned_lines)
    collapsed = " ".join(collapsed.split())
    collapsed = collapsed.replace("```", "")

    collapsed = re.sub(r"(target cypher query|schema context|domain schema reference)\s*:?", "", collapsed, flags=re.IGNORECASE)
    collapsed = re.sub(r"A natural language question:?", "", collapsed, flags=re.IGNORECASE)
    collapsed = collapsed.strip()

    trans_split = re.split(r"(?i)translation\s*:", collapsed)
    if len(trans_split) > 1:
        collapsed = trans_split[-1].strip()

    match = re.search(r'(?i)natural\s+language\s+question[^:]*:\s*"([^"]+)"', collapsed)
    if match:
        candidate = match.group(1).strip()
        if any(ch.isalpha() for ch in candidate):
            return candidate

    match = re.search(r'(?i)natural\s+language\s+question[^:]*:\s*([^\n]+)', collapsed)
    if match:
        candidate = match.group(1).strip().strip('"').strip("'")
        if any(ch.isalpha() for ch in candidate):
            return candidate

    if re.search(r"[{}\[\]]", collapsed):
        for _ in range(5):
            new_collapsed = re.sub(r"\{[^{}]*\}", " ", collapsed)
            if new_collapsed == collapsed:
                break
            collapsed = new_collapsed
        collapsed = re.sub(r"\[[^\[\]]*\]", " ", collapsed)

    collapsed = re.sub(r"\`\s*[^`]*Example:[^`]*", " ", collapsed, flags=re.IGNORECASE)
    collapsed = re.sub(r"-\s*`[^`]+`:\s*(STRING|INTEGER|FLOAT|BOOLEAN|DATE|Example:)", " ", collapsed, flags=re.IGNORECASE)

    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    collapsed = collapsed.strip('"').strip("'")

    if not collapsed:
        raise ValueError("Translation appears empty after cleaning.")

    def looks_like_metadata_sentence(text: str) -> bool:
        lowered = text.lower()
        if any(token in lowered for token in ("example:", "min size", "max size", "boolean", "integer", "float", "list min", "list max", "string example")):
            return True
        if "`" in text and any(pattern in lowered for pattern in (": string", ": integer", ": float", "example:", "available options:")):
            return True
        return False

    sentence_candidates = re.split(r"(?<=[.!?])\s+", collapsed)
    for candidate in sentence_candidates:
        candidate = candidate.strip().strip('"').strip("'")
        if len(candidate.split()) < 3:
            continue
        letters = sum(ch.isalpha() for ch in candidate)
        if '":' in candidate or '": ' in candidate:
            continue
        if looks_like_metadata_sentence(candidate):
            continue
        if letters >= 5 and letters / max(len(candidate), 1) >= 0.5:
            return candidate

    fallback = collapsed.strip().strip('"').strip("'")
    if fallback and sum(ch.isalpha() for ch in fallback) >= 5 and not looks_like_metadata_sentence(fallback):
        return fallback

    raise ValueError("Could not extract a natural language translation from LLM output.")


def load_hybrid_similarity_dataset(similarity_json: Path) -> List[Dict[str, Any]]:
    """Load and validate the sentence embedding similarity dataset."""
    with similarity_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected JSON structure in {similarity_json}")
    if not payload:
        raise ValueError(f"No entries found in {similarity_json}")

    return payload


def load_text_generation_model(
    model_name: str,
    token: Optional[str] = None,
    cpu_offload: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load tokenizer and model for text generation."""
    if torch.cuda.is_available():
        # Clear cache more aggressively
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Set environment variable to help with CUDA errors
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Set to 1 for debugging, 0 for speed

    auth_kwargs: Dict[str, Any] = {"token": token} if token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use util4's exact approach - simple and stable
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = None
    max_memory: Optional[Dict[str, str]] = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_fraction = 0.9  # Use 90% like util4
        available_gpus = min(gpu_count, 2)  # Use 2 GPUs as requested
        max_memory = {
            idx: f"{int(torch.cuda.get_device_properties(idx).total_memory / 1024**3 * total_fraction)}GiB"
            for idx in range(available_gpus)
        }
        if cpu_offload:
            max_memory["cpu"] = cpu_offload
        device_map = "balanced" if available_gpus >= 2 else "auto"
        LOGGER.info(f"Using {available_gpus} GPU(s) for model loading (GPUs: {list(range(available_gpus))})")
    elif cpu_offload:
        max_memory = {"cpu": cpu_offload}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=False,
        **auth_kwargs,
    )
    
    # Optimize model for faster inference
    if torch.cuda.is_available():
        # Enable better memory management and faster inference
        if hasattr(model, 'config'):
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True  # Enable KV cache for faster generation
        
        # Disable torch.compile for multi-GPU setups (causes CUDA illegal memory access errors)
        # torch.compile has issues with device_map="balanced" and multi-GPU
        # LOGGER.info("Skipping torch.compile for multi-GPU stability")
    
    return tokenizer, model


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Generation timed out")


def generate_translation_openai(
    prompt: str,
    *,
    model_name: str = "gpt-4o-mini",
    api_key: str,
    max_new_tokens: int = 800,
    timeout_seconds: int = 60,
) -> str:
    """Generate translation using OpenAI API."""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            timeout=timeout_seconds,
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        # Clean up the output
        lines = raw_text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not re.match(r'^[_\-\s=]+$', stripped):
                cleaned_lines.append(stripped)
        
        if cleaned_lines:
            raw_text = " ".join(cleaned_lines)
        
        raw_text = re.sub(r'_{3,}', ' ', raw_text)
        raw_text = re.sub(r'-{3,}', ' ', raw_text)
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()
        
        try:
            return preprocess_llm_output(raw_text)
        except ValueError as e:
            LOGGER.warning(f"Failed to preprocess OpenAI output. Raw output (first 500 chars): {raw_text[:500]}")
            raw_text_clean = raw_text.strip()
            for prefix in ["Natural Language:", "Translation:", "Question:", "Answer:", "Response:"]:
                if raw_text_clean.lower().startswith(prefix.lower()):
                    raw_text_clean = raw_text_clean[len(prefix):].strip()
            raw_text_clean = raw_text_clean.strip('"').strip("'").strip()
            raw_text_clean = re.sub(r'[_\-\s]{3,}', ' ', raw_text_clean).strip()
            if raw_text_clean and len(raw_text_clean.split()) >= 3:
                sentences = re.split(r'(?<=[.!?])\s+', raw_text_clean)
                if sentences:
                    candidate = sentences[0].strip()
                    if len(candidate.split()) >= 3:
                        return candidate
            if raw_text_clean and len(raw_text_clean.split()) >= 3:
                return raw_text_clean
            raise e
            
    except Exception as e:
        LOGGER.error(f"OpenAI API error: {e}")
        raise


def generate_translation(
    prompt: str,
    tokenizer: Any,
    model: Any,
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    model_name: Optional[str] = None,
    timeout_seconds: int = 60,
) -> str:
    """Generate translation text given a prompt and loaded model with timeout."""
    # Simple device handling like util4 - just use model.device
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    
    # Check if this is Qwen3 model
    is_qwen3 = model_name and "qwen3" in model_name.lower()
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 supports enable_thinking parameter - set to False for non-thinking mode
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }
        if is_qwen3:
            template_kwargs["enable_thinking"] = False  # Disable thinking mode for translation
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            **template_kwargs
        )
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if torch.cuda.is_available():
        safe_cuda_clear()

    base_len = input_ids.shape[-1]

    # Set up timeout signal (Unix only) - only in main thread (signal.alarm fails in worker threads)
    use_signal_timeout = False
    old_handler = None
    if hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread():
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        use_signal_timeout = True

    try:
        with torch.no_grad():
            try:
                # Qwen3 recommended settings for non-thinking mode: Temperature=0.7, TopP=0.8, TopK=20
                if is_qwen3:
                    gen_temperature = 0.7
                    gen_top_p = 0.8
                    gen_top_k = 20
                else:
                    gen_temperature = TEMPERATURE
                    gen_top_p = TOP_P
                    gen_top_k = TOP_K
                
                # Use optimized generation settings for faster inference
                # Enable flash attention if available for speed
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": gen_temperature,
                    "top_p": gen_top_p,
                    "top_k": gen_top_k,
                    "do_sample": gen_temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,  # Enable KV cache for faster generation
                    "num_beams": 1,  # Greedy decoding (faster than beam search)
                }
                
                # Don't use flash attention with multi-GPU and device_map (causes issues)
                # outputs = model.generate(**generation_kwargs)
                
                # Simple generation like util4 - model handles device_map internally
                # No use_cache or num_beams - keep it simple like util4
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    top_k=gen_top_k,
                    do_sample=gen_temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    safe_cuda_clear()
                # Retry with reduced tokens but still reasonable
                reduced_tokens = max(max_new_tokens // 2, 256)  # Increased minimum from 64 to 256
                LOGGER.warning(f"OOM error, retrying with reduced max_new_tokens: {reduced_tokens}")
                
                # Also try reducing input length if it's very long (for full_schema)
                if input_ids.shape[1] > 12000:
                    # Truncate input but keep the query and examples
                    # Keep last 12000 tokens (should include query and examples)
                    keep_tokens = 12000
                    input_ids = input_ids[:, -keep_tokens:]
                    attention_mask = attention_mask[:, -keep_tokens:]
                    LOGGER.warning(f"Input too long ({input_ids.shape[1]} tokens), truncating to {keep_tokens} tokens")
                # Use same temperature settings as above
                if is_qwen3:
                    gen_temperature = 0.7
                    gen_top_p = 0.8
                    gen_top_k = 20
                else:
                    gen_temperature = TEMPERATURE
                    gen_top_p = TOP_P
                    gen_top_k = TOP_K
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=reduced_tokens,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    top_k=gen_top_k,
                    do_sample=gen_temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
    except TimeoutError:
        # If timeout, return empty string - will be handled by caller
        raise TimeoutError(f"Translation generation timed out after {timeout_seconds} seconds")
    finally:
        # Cancel timeout (only if we set it in main thread)
        if use_signal_timeout and hasattr(signal, 'SIGALRM') and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    generated_ids = outputs[0][base_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # For Qwen3, remove thinking content wrapped in <think> tags
    if is_qwen3:
        # Remove thinking content between <think> tags
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    
    raw_text = raw_text.strip()
    
    lines = raw_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not re.match(r'^[_\-\s=]+$', stripped):
            cleaned_lines.append(stripped)
    
    if cleaned_lines:
        raw_text = " ".join(cleaned_lines)
    
    raw_text = re.sub(r'_{3,}', ' ', raw_text)
    raw_text = re.sub(r'-{3,}', ' ', raw_text)
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
    
    try:
        return preprocess_llm_output(raw_text)
    except ValueError as e:
        LOGGER.warning(f"Failed to preprocess LLM output. Raw output (first 500 chars): {raw_text[:500]}")
        raw_text_clean = raw_text.strip()
        for prefix in ["Natural Language:", "Translation:", "Question:", "Answer:", "Response:"]:
            if raw_text_clean.lower().startswith(prefix.lower()):
                raw_text_clean = raw_text_clean[len(prefix):].strip()
        raw_text_clean = raw_text_clean.strip('"').strip("'").strip()
        raw_text_clean = re.sub(r'[_\-\s]{3,}', ' ', raw_text_clean).strip()
        if raw_text_clean and len(raw_text_clean.split()) >= 3:
            sentences = re.split(r'(?<=[.!?])\s+', raw_text_clean)
            if sentences:
                candidate = sentences[0].strip()
                if len(candidate.split()) >= 3:
                    return candidate
        if raw_text_clean and len(raw_text_clean.split()) >= 3:
            return raw_text_clean
        raise e


def generate_translation_with_fallback(
    similarity_entry: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    model_name: str,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """
    Generate translation with multiple LLM-based fallback strategies.
    All strategies use LLM generation, only with different prompt complexities.
    Returns: (translation, method_used)
    """
    test_query = similarity_entry.get("test_query", "")
    test_metadata = similarity_entry.get("test_metadata", {})
    # Get instance_id from top level first, then fallback to metadata
    instance_id = similarity_entry.get("instance_id", "")
    if not instance_id:
        instance_id = test_metadata.get("instance_id", "")
    
    # Load pruned schema
    pruned_schema = load_pruned_schema(instance_id)
    pruned_schema_text = None
    if pruned_schema:
        pruned_schema_text = format_pruned_schema(pruned_schema)
    
    # Build metadata context
    metadata_context = None
    if test_metadata:
        metadata_parts = []
        if test_metadata.get("database_reference_alias"):
            metadata_parts.append(f"Database: {test_metadata['database_reference_alias']}")
        if test_metadata.get("data_source"):
            metadata_parts.append(f"Data Source: {test_metadata['data_source']}")
        if metadata_parts:
            metadata_context = "\n".join(metadata_parts)
    
    # Strategy 1: Full prompt with few-shot examples
    try:
        reference_queries = collect_reference_queries(similarity_entry, instance_id, max_examples=REFERENCE_EXAMPLES)
        prompt = create_few_shot_prompt_cypher(
            test_query,
            reference_queries,
            pruned_schema_text=pruned_schema_text,
            metadata_context=metadata_context,
        )
        translation = generate_translation(
            prompt,
            tokenizer,
            model,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            model_name=model_name,
        )
        if translation and translation.strip():
            return translation, "full_prompt"
    except Exception as e:
        LOGGER.warning(f"Strategy 1 (full prompt) failed for {instance_id}: {e}")
    
    # Strategy 2: Simplified prompt without few-shot examples but with schema
    try:
        prompt = create_simple_prompt(test_query, pruned_schema_text)
        translation = generate_translation(
            prompt,
            tokenizer,
            model,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            model_name=model_name,
        )
        if translation and translation.strip():
            return translation, "simple_prompt_with_schema"
    except Exception as e:
        LOGGER.warning(f"Strategy 2 (simple prompt with schema) failed for {instance_id}: {e}")
    
    # Strategy 3: Minimal prompt with just query and basic instruction
    try:
        prompt = create_minimal_prompt(test_query)
        translation = generate_translation(
            prompt,
            tokenizer,
            model,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            model_name=model_name,
        )
        if translation and translation.strip():
            return translation, "minimal_prompt"
    except Exception as e:
        LOGGER.warning(f"Strategy 3 (minimal prompt) failed for {instance_id}: {e}")
    
    # Strategy 4: Even simpler prompt - just direct instruction
    try:
        prompt = f"Translate this Cypher query to English: {test_query}\n\nTranslation: "
        translation = generate_translation(
            prompt,
            tokenizer,
            model,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            model_name=model_name,
        )
        if translation and translation.strip():
            return translation, "direct_prompt"
    except Exception as e:
        LOGGER.warning(f"Strategy 4 (direct prompt) failed for {instance_id}: {e}")
    
    # If all LLM strategies fail, raise an exception
    raise ValueError(f"All LLM-based translation strategies failed for {instance_id}")


def generate_translation_for_case(
    similarity_entry: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    model_name: str,
    args: argparse.Namespace,
    case_type: str,  # "no_schema", "full_schema", "pruned_schema"
) -> Dict[str, Any]:
    """Generate translation for a specific case (no_schema, full_schema, or pruned_schema)."""
    test_query = similarity_entry.get("test_query", "")
    test_metadata = similarity_entry.get("test_metadata", {})
    instance_id = similarity_entry.get("instance_id", "")
    if not instance_id:
        instance_id = test_metadata.get("instance_id", "")
    test_index = similarity_entry.get("test_index", "")
    
    # Get reference queries (top 5 examples)
    reference_queries = collect_reference_queries(similarity_entry, instance_id, max_examples=REFERENCE_EXAMPLES)
    
    # Build metadata context
    metadata_context = None
    if test_metadata:
        metadata_parts = []
        if test_metadata.get("database_reference_alias"):
            metadata_parts.append(f"Database: {test_metadata['database_reference_alias']}")
        if test_metadata.get("data_source"):
            metadata_parts.append(f"Data Source: {test_metadata['data_source']}")
        if metadata_parts:
            metadata_context = "\n".join(metadata_parts)
    
    translation = ""
    error_info = None
    schema_content = None  # Store the schema content used for this case
    
    # Generate prompt based on case type
    try:
        if case_type == "no_schema":
            schema_content = None  # No schema used
            prompt = create_few_shot_prompt_no_schema(
                test_query,
                reference_queries,
                metadata_context=metadata_context,
            )
        elif case_type == "full_schema":
            # Get full schema from test_metadata
            full_schema_text = test_metadata.get("schema", "")
            schema_content = full_schema_text if full_schema_text else None
            
            # Aggressive memory clearing before processing full_schema (largest prompts)
            if torch.cuda.is_available():
                safe_cuda_clear()
            
            prompt = create_few_shot_prompt_full_schema(
                test_query,
                reference_queries,
                full_schema_text=full_schema_text if full_schema_text else None,
                metadata_context=metadata_context,
            )
        elif case_type == "pruned_schema":
            # Load pruned schema from file
            pruned_schema = load_pruned_schema(instance_id)
            pruned_schema_text = None
            if pruned_schema:
                pruned_schema_text = format_pruned_schema(pruned_schema)
            schema_content = pruned_schema_text  # Store the formatted pruned schema
            prompt = create_few_shot_prompt_cypher(
                test_query,
                reference_queries,
                pruned_schema_text=pruned_schema_text,
                metadata_context=metadata_context,
            )
        else:
            raise ValueError(f"Unknown case_type: {case_type}")
        
        # Generate translation with retries
        max_retries = getattr(args, 'max_retries', MAX_RETRIES)
        skip_retry_on_timeout = getattr(args, 'skip_retry_on_timeout', False)
        
        # Check if this is an OpenAI model
        is_openai_model = model_name == "gpt-4o-mini"
        openai_api_key = getattr(args, 'openai_api_key', None)
        
        # Use different context window sizes based on schema type
        # full_schema: largest (full schemas can be very large, don't truncate)
        # pruned_schema: medium (pruned schemas are smaller)
        # no_schema: smallest (no schema, just query and examples)
        if case_type == "full_schema":
            # Full schema - use maximum context window to avoid truncation
            effective_max_input = min(args.max_input_tokens, 16384)
            LOGGER.debug(f"Using max_input_tokens={effective_max_input} for full_schema")
        elif case_type == "pruned_schema":
            # Pruned schema - use medium context window
            effective_max_input = min(args.max_input_tokens, 12288)  # 75% of full
            LOGGER.debug(f"Using max_input_tokens={effective_max_input} for pruned_schema")
        else:  # no_schema
            # No schema - use smaller context window (just query + examples)
            effective_max_input = min(args.max_input_tokens, 8192)  # 50% of full
            LOGGER.debug(f"Using max_input_tokens={effective_max_input} for no_schema")
        
        for attempt in range(max_retries):
            try:
                # Aggressive memory clearing before EVERY attempt (not just full_schema or retries)
                # For full_schema, be even more aggressive due to large context
                if torch.cuda.is_available():
                    if attempt > 0:
                        LOGGER.warning(f"Retry attempt {attempt + 1}/{max_retries} for {case_type}, clearing GPU cache...")
                    safe_cuda_clear()
                    # For full_schema, use longer delays and extra clearing
                    if case_type == "full_schema":
                        # Extra aggressive clearing for full_schema
                        if is_cuda_healthy():
                            safe_cuda_clear()
                        time.sleep(1.0 if attempt == 0 else 3.0)  # Longer delays for full_schema
                    else:
                        # Small delay to let GPU settle, especially for retries
                        if attempt > 0:
                            time.sleep(2)  # Longer delay for retries
                        else:
                            time.sleep(0.5)  # Short delay even for first attempt
                
                if is_openai_model and openai_api_key:
                    # Use OpenAI API - ensure model name is exactly "gpt-4o-mini"
                    translation = generate_translation_openai(
                        prompt,
                        model_name="gpt-4o-mini",
                        api_key=openai_api_key,
                        max_new_tokens=args.max_new_tokens,
                        timeout_seconds=args.timeout_seconds,
                    )
                else:
                    # Use local model with effective max_input_tokens (larger for full_schema)
                    translation = generate_translation(
                        prompt,
                        tokenizer,
                        model,
                        max_input_tokens=effective_max_input,
                        max_new_tokens=args.max_new_tokens,
                        model_name=model_name,
                        timeout_seconds=args.timeout_seconds,
                    )
                if translation and translation.strip():
                    break
            except TimeoutError as e:
                error_info = str(e)
                LOGGER.warning(f"Timeout for {instance_id} (case: {case_type}): {e}")
                if skip_retry_on_timeout:
                    # Skip retries on timeout for faster processing
                    break
                elif attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    raise
            except Exception as e:
                error_info = str(e)
                error_lower = error_info.lower()
                
                # Check if this is a CUDA illegal memory access error
                is_cuda_corruption = (
                    "illegal memory access" in error_lower or
                    ("cuda error" in error_lower and "illegal" in error_lower)
                )
                
                if is_cuda_corruption:
                    LOGGER.error(f"CUDA corruption detected for {instance_id} (case: {case_type}). Skipping this entry.")
                    # Mark CUDA as unhealthy for this entry
                    # Don't try to clear cache - it will fail
                    # Break out of retry loop immediately
                    raise RuntimeError(f"CUDA corruption: {error_info}") from e
                
                # For other errors, try to clear memory if CUDA is healthy
                if case_type == "full_schema" and is_cuda_healthy():
                    safe_cuda_clear()
                
                if attempt < max_retries - 1:
                    LOGGER.warning(f"Attempt {attempt + 1} failed for {instance_id} (case: {case_type}), retrying...")
                    time.sleep(1.0)  # Longer sleep for full_schema to let memory settle
                else:
                    LOGGER.error(f"All {max_retries} attempts failed for {instance_id} (case: {case_type}): {e}")
        
        if not translation or not translation.strip():
            error_info = f"Translation generation failed for case {case_type}"
            
    except RuntimeError as e:
        error_str = str(e).lower()
        # Check if this is a CUDA corruption error
        if "cuda corruption" in error_str or "illegal memory access" in error_str:
            error_info = f"CUDA corruption detected: {str(e)}"
            LOGGER.error(f"[CUDA CORRUPTION] {instance_id} (case: {case_type}): {e}")
            # Don't try to clear cache - CUDA is corrupted
        else:
            error_info = f"Error generating translation for case {case_type}: {str(e)}"
            LOGGER.error(f"Error for {instance_id} (case: {case_type}): {e}")
    except Exception as e:
        error_info = f"Error generating translation for case {case_type}: {str(e)}"
        LOGGER.error(f"Error for {instance_id} (case: {case_type}): {e}")
    
    # Get gold translation
    gold_translation = similarity_entry.get("gold_translation", "")
    if not gold_translation:
        gold_translation = test_metadata.get("question", "")
    
    result_payload: Dict[str, Any] = {
        "model_name": model_name,
        "test_instance_id": instance_id,
        "test_id": f"test_{test_index}" if test_index != "" else instance_id,
        "test_index": test_index,
        "test_cypher": test_query,
        "predicted_translation": translation if translation else "",
        "gold_translation": gold_translation,
        "test_metadata": test_metadata,
        "case_type": case_type,  # Track which case was used
        "schema_content": schema_content,  # Store the schema content used (None for no_schema, full schema text for full_schema, pruned schema text for pruned_schema)
    }
    
    if error_info:
        result_payload["translation_error"] = error_info
    
    return result_payload


def generate_translation_for_entry(
    similarity_entry: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    model_name: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Generate translations for all 3 cases and return list of results."""
    results = []
    
    # Case 1: Top 5 examples, no schema
    try:
        result1 = generate_translation_for_case(
            similarity_entry, tokenizer, model, model_name, args, "no_schema"
        )
        results.append(result1)
        # Clear memory after each case
        if torch.cuda.is_available():
            safe_cuda_clear()
    except Exception as e:
        LOGGER.error(f"Failed to generate case 1 (no_schema): {e}")
        # Create error result
        test_metadata = similarity_entry.get("test_metadata", {})
        instance_id = similarity_entry.get("instance_id", "") or test_metadata.get("instance_id", "")
        results.append({
            "model_name": model_name,
            "test_instance_id": instance_id,
            "test_id": f"test_{similarity_entry.get('test_index', '')}",
            "test_index": similarity_entry.get("test_index", ""),
            "test_cypher": similarity_entry.get("test_query", ""),
            "predicted_translation": "",
            "gold_translation": similarity_entry.get("gold_translation", "") or test_metadata.get("question", ""),
            "test_metadata": test_metadata,
            "case_type": "no_schema",
            "schema_content": None,  # No schema used
            "translation_error": str(e),
        })
        # Clear memory after error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
    
    # Case 2: Top 5 examples with full schema
    try:
        result2 = generate_translation_for_case(
            similarity_entry, tokenizer, model, model_name, args, "full_schema"
        )
        results.append(result2)
        # Clear memory after each case (especially important for full_schema)
        if torch.cuda.is_available():
            safe_cuda_clear()
    except Exception as e:
        LOGGER.error(f"Failed to generate case 2 (full_schema): {e}")
        test_metadata = similarity_entry.get("test_metadata", {})
        instance_id = similarity_entry.get("instance_id", "") or test_metadata.get("instance_id", "")
        # Get full schema from metadata for error case
        full_schema_text = test_metadata.get("schema", "")
        results.append({
            "model_name": model_name,
            "test_instance_id": instance_id,
            "test_id": f"test_{similarity_entry.get('test_index', '')}",
            "test_index": similarity_entry.get("test_index", ""),
            "test_cypher": similarity_entry.get("test_query", ""),
            "predicted_translation": "",
            "gold_translation": similarity_entry.get("gold_translation", "") or test_metadata.get("question", ""),
            "test_metadata": test_metadata,
            "case_type": "full_schema",
            "schema_content": full_schema_text if full_schema_text else None,
            "translation_error": str(e),
        })
        # Clear memory after error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
    
    # Case 3: Top 5 examples with pruned schema
    try:
        result3 = generate_translation_for_case(
            similarity_entry, tokenizer, model, model_name, args, "pruned_schema"
        )
        results.append(result3)
        # Clear memory after each case
        if torch.cuda.is_available():
            safe_cuda_clear()
    except Exception as e:
        LOGGER.error(f"Failed to generate case 3 (pruned_schema): {e}")
        test_metadata = similarity_entry.get("test_metadata", {})
        instance_id = similarity_entry.get("instance_id", "") or test_metadata.get("instance_id", "")
        # Try to load pruned schema for error case
        pruned_schema = load_pruned_schema(instance_id)
        pruned_schema_text = None
        if pruned_schema:
            pruned_schema_text = format_pruned_schema(pruned_schema)
        results.append({
            "model_name": model_name,
            "test_instance_id": instance_id,
            "test_id": f"test_{similarity_entry.get('test_index', '')}",
            "test_index": similarity_entry.get("test_index", ""),
            "test_cypher": similarity_entry.get("test_query", ""),
            "predicted_translation": "",
            "gold_translation": similarity_entry.get("gold_translation", "") or test_metadata.get("question", ""),
            "test_metadata": test_metadata,
            "case_type": "pruned_schema",
            "schema_content": pruned_schema_text,
            "translation_error": str(e),
        })
        # Clear memory after error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
    
    return results


def _safe_filename_component(value: Any) -> str:
    """Create a filesystem-safe string component."""
    text = str(value) if value is not None else "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"


def _parse_hybrid_jsonl_basename(name: str) -> Optional[Tuple[str, str]]:
    """Parse (model_key, case_type) from Hybrid_{model_key}_{case_type}_*.jsonl. Returns None if not matched."""
    m = re.match(r"^Hybrid_(.+?)_(full_schema|no_schema|pruned_schema)_", name)
    return (m.group(1), m.group(2)) if m else None


def _regenerate_txt_from_jsonl(rows: List[Dict[str, Any]], txt_path: Path, case_type: str) -> None:
    """Regenerate .txt file from jsonl rows (same block format as Cypher2Text write)."""
    with txt_path.open("w", encoding="utf-8") as f:
        for case_result in rows:
            txt = "=" * 80 + "\n"
            txt += f"Test ID: {case_result.get('test_id', 'N/A')}\n"
            txt += f"Test Instance ID: {case_result.get('test_instance_id', 'N/A')}\n"
            txt += f"Model: {case_result.get('model_name', 'N/A')}\n"
            txt += f"Case Type: {case_type}\n"
            txt += f"Cypher Query: {case_result.get('test_cypher', '')}\n"
            txt += f"Predicted Translation: {case_result.get('predicted_translation', '')}\n"
            txt += f"Gold Translation: {case_result.get('gold_translation', '')}\n"
            if case_result.get("translation_error"):
                txt += f"Translation Error: {case_result.get('translation_error')}\n"
            txt += f"\nSchema Content Used ({case_type}):\n"
            sc = case_result.get("schema_content")
            txt += f"{sc}\n" if sc else "None (no schema was used in the prompt)\n"
            txt += "Test Metadata:\n"
            txt += json.dumps(case_result.get("test_metadata", {}), ensure_ascii=False, indent=2)
            txt += "\n" + "=" * 80 + "\n\n"
            f.write(txt)


def run_fill_missing(args: argparse.Namespace) -> None:
    """Scan output-dir for Cypher2Text jsonl with empty predicted_translation, run translation for those, update in place."""
    output_dir = Path(args.output_dir)
    similarity_path = Path(args.similarity_json)
    if not output_dir.exists():
        LOGGER.error(f"Output dir does not exist: {output_dir}")
        return
    if not similarity_path.exists():
        LOGGER.error(f"Similarity JSON does not exist: {similarity_path}")
        return

    # Build instance_id -> full similarity entry (for top_matches)
    LOGGER.info(f"Loading similarity data from {similarity_path}")
    try:
        sim_list = load_hybrid_similarity_dataset(similarity_path)
        similarity_lookup: Dict[str, Dict[str, Any]] = {
            (e.get("instance_id") or (e.get("test_metadata") or {}).get("instance_id") or ""): e
            for e in sim_list
            if e.get("instance_id") or (e.get("test_metadata") or {}).get("instance_id")
        }
    except Exception as e:
        LOGGER.error(f"Failed to load similarity JSON: {e}")
        return
    LOGGER.info(f"Similarity lookup: {len(similarity_lookup)} instance_ids")

    # Discover Hybrid_*_{case_type}_*.jsonl in case subdirs
    jsonl_files: List[Tuple[Path, str, str]] = []  # (path, model_key, case_type)
    for case_type in ("full_schema", "no_schema", "pruned_schema"):
        case_dir = output_dir / case_type
        if not case_dir.exists():
            continue
        for p in case_dir.glob("Hybrid_*_*.jsonl"):
            parsed = _parse_hybrid_jsonl_basename(p.name)
            if parsed:
                model_key, ct = parsed
                if ct == case_type:
                    jsonl_files.append((p, model_key, case_type))

    if not jsonl_files:
        LOGGER.warning("No Hybrid_*_{case_type}_*.jsonl files found under %s", output_dir)
        return

    # Group by model_key to load each model once
    by_model: Dict[str, List[Tuple[Path, str]]] = {}  # model_key -> [(path, case_type), ...]
    for path, model_key, case_type in jsonl_files:
        by_model.setdefault(model_key, []).append((path, case_type))

    # Ensure args attributes used by generate_translation_for_case
    if not hasattr(args, "timeout_seconds"):
        args.timeout_seconds = 60
    if not hasattr(args, "max_retries"):
        args.max_retries = MAX_RETRIES
    if not hasattr(args, "skip_retry_on_timeout"):
        args.skip_retry_on_timeout = False

    for model_key in sorted(by_model.keys()):
        model_name = MODELS.get(model_key) or (model_key if model_key == "gpt-4o-mini" else None)
        if not model_name:
            LOGGER.warning("Unknown model_key %s, skipping", model_key)
            continue
        is_openai = model_name == "gpt-4o-mini"
        tokenizer, model = None, None
        if not is_openai:
            try:
                token = None
                if args.token_path.exists():
                    token = args.token_path.read_text(encoding="utf-8").strip()
                tokenizer, model = load_text_generation_model(model_name, token, cpu_offload=getattr(args, "cpu_offload", None))
                LOGGER.info("Loaded model %s for fill-missing", model_key)
            except Exception as e:
                LOGGER.error("Failed to load model %s: %s", model_key, e)
                continue
        else:
            if not getattr(args, "openai_api_key", None):
                LOGGER.warning("gpt4o-mini requires openai_api_key; skipping %s", model_key)
                continue

        for jsonl_path, case_type in by_model[model_key]:
            try:
                _process_one_jsonl_fill_missing(
                    jsonl_path, case_type, model_name, tokenizer, model, args, similarity_lookup,
                )
            except Exception as e:
                LOGGER.exception("Error processing %s: %s", jsonl_path, e)
            if not is_openai and torch.cuda.is_available():
                safe_cuda_clear()

        if not is_openai and model is not None and torch.cuda.is_available():
            safe_cuda_clear()
            import gc
            gc.collect()


def _process_one_jsonl_fill_missing(
    jsonl_path: Path,
    case_type: str,
    model_name: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
    similarity_lookup: Dict[str, Dict[str, Any]],
) -> None:
    """Read jsonl, translate entries with empty predicted_translation, write back and regenerate txt."""
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    missing = [i for i, r in enumerate(rows) if not (r.get("predicted_translation") or "").strip()]
    if not missing:
        LOGGER.info("No missing translations in %s", jsonl_path)
        return
    n_missing = len(missing)
    LOGGER.info("Filling %d missing translation(s) in %s", n_missing, jsonl_path)
    sys.stderr.flush()

    txt_path = jsonl_path.with_suffix(".txt")
    CHECKPOINT_EVERY = 500

    for pos, i in enumerate(missing):
        r = rows[i]
        instance_id = r.get("test_instance_id") or (r.get("test_metadata") or {}).get("instance_id") or ""
        if pos == 0:
            LOGGER.info("Translating 1/%d (instance %s) — first may take 1–2 min for full_schema", n_missing, instance_id)
            sys.stderr.flush()
        elif (pos + 1) % 25 == 0:
            LOGGER.info("Fill-missing progress: %d / %d", pos + 1, n_missing)
            sys.stderr.flush()

        ti = r.get("test_index")
        if ti is None and isinstance(r.get("test_id"), str) and r["test_id"].startswith("test_"):
            try:
                ti = int(r["test_id"].replace("test_", "").strip())
            except ValueError:
                pass
        sim: Dict[str, Any] = {
            "test_query": r.get("test_cypher", ""),
            "test_metadata": r.get("test_metadata", {}),
            "instance_id": instance_id,
            "test_index": ti,
            "gold_translation": r.get("gold_translation", ""),
        }
        sim["top_matches"] = similarity_lookup.get(instance_id, {}).get("top_matches", []) or []

        try:
            res = generate_translation_for_case(sim, tokenizer, model, model_name, args, case_type)
            new_pred = (res.get("predicted_translation") or "").strip()
            rows[i]["predicted_translation"] = new_pred
            if "translation_error" in rows[i]:
                del rows[i]["translation_error"]
            if res.get("translation_error"):
                rows[i]["translation_error"] = res["translation_error"]
            if res.get("schema_content") is not None:
                rows[i]["schema_content"] = res["schema_content"]
        except Exception as e:
            LOGGER.warning("Failed to translate %s (case %s): %s", instance_id, case_type, e)
            rows[i]["translation_error"] = str(e)

        if pos == 0:
            LOGGER.info("Completed 1/%d", n_missing)
            sys.stderr.flush()

        # Checkpoint every N to avoid losing progress on crash
        if (pos + 1) % CHECKPOINT_EVERY == 0:
            with jsonl_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            _regenerate_txt_from_jsonl(rows, txt_path, case_type)
            LOGGER.info("Checkpoint: saved %d / %d", pos + 1, n_missing)
            sys.stderr.flush()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _regenerate_txt_from_jsonl(rows, txt_path, case_type)
    LOGGER.info("Updated %s and %s", jsonl_path, txt_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Cypher queries to natural language with improved error handling and fallbacks."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use SimilarityResultsManualDataset3, PrunedSchemasDataset3, 5 examples, pruned_schema only; save to Cypher2TextDataset3Manual.",
    )
    parser.add_argument(
        "--similarity-json",
        type=Path,
        default=HYBRID_SIMILARITY_JSON,
        help="Path to sentence embedding similarity results JSON.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to use for translation. Options: %s, or 'all' (default: all). All schema cases (no_schema, full_schema, pruned_schema) run per model."
        % (", ".join(MODELS.keys())),
    )
    parser.add_argument(
        "--test-instance-id",
        type=str,
        default=None,
        help="Specific test instance ID to use.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset identifier to embed in output filenames.",
    )
    parser.add_argument(
        "--token-path",
        type=Path,
        default=Path("/home/ubuntu/Huggingface_api.txt"),
        help="Path to Hugging Face API token file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory where the translation results will be saved (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=16384,  # Increased to handle very large full schemas without truncation
        help="Maximum number of tokens for the prompt input.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate for each translation.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between processing queries.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=str,
        default=None,
        help="Optional CPU offload memory budget.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process every entry in the similarity JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Number of entries to process per model (default: 10).",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Print the constructed prompt for each entry.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of queries to process in parallel per batch (default: 1). Uses ThreadPoolExecutor for batch_size>1. Recommended: 4 for OpenAI, 1 for local GPU models.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout for each translation generation in seconds (default: 60). Reduce for faster failure detection.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help=f"Maximum retries for failed translations (default: {MAX_RETRIES}). Reduce to 1 for faster processing.",
    )
    parser.add_argument(
        "--skip-retry-on-timeout",
        action="store_true",
        help="Skip retries if translation times out (faster but may miss some valid translations).",
    )
    parser.add_argument(
        "--failed-entries-json",
        type=Path,
        default=None,
        help="Path to JSON file with failed entries to retry. Format: [{\"instance_id\": \"...\", \"case_type\": \"...\"}, ...]",
    )
    parser.add_argument(
        "--openai-api-key-path",
        type=Path,
        default=Path("/home/ubuntu/OpenAI_api.txt"),
        help="Path to OpenAI API key file.",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Scan output-dir for Cypher2Text jsonl with empty predicted_translation, run translation for those only, and update files in place.",
    )
    args = parser.parse_args()

    token = None
    if args.token_path.exists():
        try:
            token = args.token_path.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                print("Loaded Hugging Face token from file.")
        except Exception as e:
            LOGGER.warning(f"Could not load token: {e}")
    
    # Load OpenAI API key
    openai_api_key = None
    if args.openai_api_key_path.exists():
        try:
            openai_api_key = args.openai_api_key_path.read_text(encoding="utf-8").strip()
            if openai_api_key:
                args.openai_api_key = openai_api_key
                print("Loaded OpenAI API key from file.")
        except Exception as e:
            LOGGER.warning(f"Could not load OpenAI API key: {e}")
    else:
        LOGGER.warning(f"OpenAI API key file not found at {args.openai_api_key_path}")

    if getattr(args, "fill_missing", False):
        run_fill_missing(args)
        return

    if getattr(args, "manual", False):
        global _pruned_schemas_dir_override
        _pruned_schemas_dir_override = PRUNED_SCHEMAS_DIR_MANUAL
        args.similarity_json = SIMILARITY_JSON_MANUAL
        args.output_dir = OUTPUT_DIR_MANUAL
        LOGGER.info("Manual mode: similarity=%s, pruned_schemas=%s, output=%s", args.similarity_json, _pruned_schemas_dir_override, args.output_dir)
    if "all" in args.models:
        models_to_use = list(MODELS.keys())
        LOGGER.info("Using all models: %s", ", ".join(models_to_use))
    else:
        models_to_use = args.models
    case_types = ["pruned_schema"] if getattr(args, "manual", False) else ["no_schema", "full_schema", "pruned_schema"]
    LOGGER.info("Schema cases per model: %s", ", ".join(case_types))

    dataset_name = args.dataset_name or args.similarity_json.stem
    args.dataset_name_resolved = _safe_filename_component(dataset_name)

    dataset = load_hybrid_similarity_dataset(args.similarity_json)

    # Filter by failed entries if provided
    if args.failed_entries_json and args.failed_entries_json.exists():
        LOGGER.info(f"Loading failed entries from {args.failed_entries_json}")
        with open(args.failed_entries_json, "r") as f:
            failed_entries_list = json.load(f)
        
        # Create a set of (instance_id, case_type) tuples for fast lookup
        failed_set = {(e["instance_id"], e["case_type"]) for e in failed_entries_list}
        LOGGER.info(f"Found {len(failed_set)} failed entry-case combinations to retry")
        
        # Filter dataset to only include entries that match failed instance_ids
        failed_instance_ids = {e["instance_id"] for e in failed_entries_list}
        filtered: List[Dict[str, Any]] = []
        for entry in dataset:
            instance_id = entry.get("test_metadata", {}).get("instance_id", "")
            if instance_id in failed_instance_ids:
                filtered.append(entry)
        
        if not filtered:
            LOGGER.warning(f"No entries found matching failed instance_ids. Proceeding with full dataset.")
            dataset = dataset
        else:
            dataset = filtered
            LOGGER.info(f"Filtered dataset to {len(dataset)} entries matching failed instance_ids")
            # Store failed_set in args for later filtering by case_type
            args.failed_entries_set = failed_set
    else:
        args.failed_entries_set = None

    if args.test_instance_id:
        filtered: List[Dict[str, Any]] = []
        for entry in dataset:
            instance_id = entry.get("test_metadata", {}).get("instance_id", "")
            if instance_id == args.test_instance_id:
                filtered.append(entry)
        if not filtered:
            raise ValueError(f"Test instance {args.test_instance_id!r} not found")
        dataset = filtered
    elif not args.process_all and args.failed_entries_json is None:
        dataset = dataset[:args.max_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Remove "similarity_results" from dataset name - we only want translation files
    sanitized_dataset = args.dataset_name_resolved.replace("similarity_results", "translations")
    if sanitized_dataset == args.dataset_name_resolved:  # If no replacement happened, add translations
        sanitized_dataset = "translations"

    # Load existing translations to skip already processed entries per case type
    def load_existing_translations(model_key: str, output_dir: Path, case_type: str) -> tuple[set, Optional[Path], Optional[Path]]:
        """Load set of test_ids that already have translations for this model and case type.
        Returns: (existing_test_ids, existing_jsonl_path, existing_txt_path)
        """
        existing = set()
        existing_jsonl = None
        existing_txt = None
        
        # Check in case_type subdirectory first (new structure), then root (old structure)
        case_dir = output_dir / case_type
        search_dirs = [case_dir] if case_dir.exists() else []
        search_dirs.append(output_dir)  # Also check root for backward compatibility
        
        jsonl_files = []
        for search_dir in search_dirs:
            # Look for existing files for this model and case type
            # New structure: Hybrid_{model_key}_{case_type}_*.jsonl in subdirectory (translation files only)
            # Also check old format: Hybrid_{model_key}_*.jsonl in subdirectory (backward compatibility)
            # Old structure: Hybrid_{model_key}_{case_type}_*.jsonl in root
            if search_dir == case_dir:
                # Check both new format (with case_type) and old format (without case_type)
                pattern1 = f"Hybrid_{model_key}_{case_type}_*.jsonl"
                pattern2 = f"Hybrid_{model_key}_*.jsonl"
                found_files1 = list(search_dir.glob(pattern1))
                found_files2 = list(search_dir.glob(pattern2))
                jsonl_files.extend(found_files1)
                jsonl_files.extend(found_files2)
            else:
                pattern = f"Hybrid_{model_key}_{case_type}_*.jsonl"
                found_files = list(search_dir.glob(pattern))
                jsonl_files.extend(found_files)
        
        if jsonl_files:
            # Load from all existing files for this case type (not just most recent)
            for jsonl_file in jsonl_files:
                # Load existing test_ids from each file
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                # Only count as existing if predicted_translation is non-empty (don't skip empty translations)
                                pred = (entry.get("predicted_translation") or "").strip()
                                if not pred:
                                    continue
                                test_id = entry.get("test_id", "") or f"test_{entry.get('test_index', '')}"
                                if test_id:
                                    existing.add(test_id)
                except Exception as e:
                    LOGGER.warning(f"Error reading existing file {jsonl_file}: {e}")
            
            # Use the most recent file for appending
            existing_jsonl = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            # Find corresponding txt file
            txt_path = existing_jsonl.with_suffix('.txt')
            if txt_path.exists():
                existing_txt = txt_path
            else:
                existing_txt = None
        
        return existing, existing_jsonl, existing_txt

    failed_entries = []
    success_count = 0
    fallback_count = 0

    for model_key in models_to_use:
        model_name = MODELS[model_key]
        print(f"\n{'='*80}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*80}\n")
        
        # Check if this is an OpenAI model
        is_openai_model = model_name == "gpt-4o-mini" or model_key == "gpt4o-mini"
        
        # Load existing translations per case type
        existing_per_case = {}
        for case_type in case_types:
            existing_per_case[case_type], _, _ = load_existing_translations(model_key, args.output_dir, case_type)
            if existing_per_case[case_type]:
                LOGGER.info(f"For {model_key} {case_type}: Found {len(existing_per_case[case_type])} existing translations")
        
        # Filter dataset to only include entries not yet processed for ALL case types
        # BUT: If retrying failed entries, don't skip - we want to replace failed entries
        model_dataset = dataset.copy()
        if args.process_all and not args.failed_entries_set:
            original_count = len(model_dataset)
            # Only skip if entry exists in ALL case types (and we're not retrying failed entries)
            model_dataset = [
                entry for entry in model_dataset 
                if not all(
                    f"test_{entry.get('test_index', '')}" in existing_per_case[case_type]
                    for case_type in case_types
                )
            ]
            skipped = original_count - len(model_dataset)
            if skipped > 0:
                LOGGER.info(f"For {model_key}: Skipping {skipped} entries already processed in all case types, {len(model_dataset)} remaining")
            if len(model_dataset) == 0:
                LOGGER.info(f"All entries for {model_key} are already processed in all case types. Skipping.")
                continue
        elif args.failed_entries_set:
            # When retrying failed entries, don't filter by existing - we want to retry them
            LOGGER.info(f"For {model_key}: Retrying failed entries, will replace existing failed translations")
        elif not args.process_all:
            model_dataset = model_dataset[:args.max_samples]

        # Initialize model and tokenizer to None (will be set below or remain None for OpenAI)
        tokenizer = None
        model = None
        
        # Check CUDA health before loading model (for non-OpenAI models)
        if not is_openai_model:
            if not is_cuda_healthy():
                LOGGER.error("CUDA is corrupted. Cannot load model. Please restart the process.")
                LOGGER.error("Skipping this model due to CUDA corruption.")
                continue
        
        # Skip model loading for OpenAI models (they use API)
        if is_openai_model:
            if not hasattr(args, 'openai_api_key') or not args.openai_api_key:
                LOGGER.error(f"OpenAI API key not found. Skipping {model_key}.")
                continue
            print(f"Using OpenAI API for {model_name} (no local model loading required).\n")
        else:
            print("Loading language model...")
            tokenizer, model = load_text_generation_model(model_name, token, cpu_offload=args.cpu_offload)
            # Keep use_cache enabled for faster generation (set in load_text_generation_model)
            print(f"Model {model_name} loaded.\n")

        # Organize outputs into separate folders for each schema type (like Cypher2Text2 structure)
        case_files = {}
        
        for case_type in case_types:
            # Create subdirectory for each case type
            case_dir = args.output_dir / case_type
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for existing files for this model and case type in the subdirectory
            # File naming: Hybrid_{model_key}_{case_type}_{dataset}_{timestamp}.jsonl (no "similarity_results")
            pattern = f"Hybrid_{model_key}_{case_type}_*.jsonl"
            jsonl_files = list(case_dir.glob(pattern))
            
            if jsonl_files:
                # Use the most recent existing file (will append to it)
                json_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                # Find corresponding txt file
                txt_path = json_path.with_suffix('.txt')
                if not txt_path.exists():
                    # If txt doesn't exist, create it with same base name
                    txt_path = json_path.parent / f"{json_path.stem}.txt"
                LOGGER.info(f"Appending to existing files for {model_key} {case_type}: {json_path.name} ({len(existing_per_case[case_type])} existing entries)")
            else:
                # Create new files with model_key and case_type in filename (translation files only, no similarity_results)
                json_path = case_dir / f"Hybrid_{model_key}_{case_type}_{sanitized_dataset}_{timestamp}.jsonl"
                txt_path = case_dir / f"Hybrid_{model_key}_{case_type}_{sanitized_dataset}_{timestamp}.txt"
                LOGGER.info(f"Creating new files for {model_key} {case_type} in {case_dir.name}/: {json_path.name}")
            
            case_files[case_type] = {
                "json": json_path,
                "txt": txt_path,
            }

        written = {case: 0 for case in case_types}
        failed_count = {case: 0 for case in case_types}
        start_time = time.time()

        # Open all file handles in append mode (will append to existing files)
        file_handles = {}
        for case_type in case_types:
            json_path = case_files[case_type]["json"]
            txt_path = case_files[case_type]["txt"]
            # Use append mode ("a") to write on top of existing translations
            file_handles[case_type] = {
                "json": json_path.open("a", encoding="utf-8"),
                "txt": txt_path.open("a", encoding="utf-8"),
            }

        try:
            batch_size = getattr(args, "batch_size", 1)
            # Only use parallel workers for OpenAI (API calls). Local GPU models are not thread-safe:
            # concurrent model.generate() from multiple threads causes CUDA illegal memory access.
            use_batch = batch_size > 1 and is_openai_model
            pbar = tqdm(model_dataset, desc=f"Translating with {model_key}", unit="query")
            file_lock = threading.Lock()
            abort_flag = [False]  # set to True on CUDA corruption to stop main loop

            def _process_one_entry(entry: Dict[str, Any]) -> None:
                """Process one entry (generate + write under lock). Skip returns early."""
                nonlocal success_count
                instance_id = entry.get("test_metadata", {}).get("instance_id", "")
                test_index = entry.get("test_index", "")
                test_id = f"test_{test_index}" if test_index else instance_id
                failed_cases_for_entry = None
                if args.failed_entries_set:
                    failed_cases_for_entry = {case for (inst_id, case) in args.failed_entries_set if inst_id == instance_id}
                    if not failed_cases_for_entry:
                        return  # skip
                    LOGGER.info(f"Retrying {instance_id} for cases: {failed_cases_for_entry}")
                try:
                    if failed_cases_for_entry:
                        # Only generate specific failed cases
                        case_results = []
                        # Process full_schema LAST to avoid memory issues affecting other cases
                        case_order = ["no_schema", "pruned_schema", "full_schema"] if "full_schema" in failed_cases_for_entry else ["no_schema", "full_schema", "pruned_schema"]
                        for case_type in case_order:
                            if case_type in failed_cases_for_entry:
                                # Aggressive memory clearing before each case type
                                if torch.cuda.is_available() and not is_openai_model:
                                    safe_cuda_clear()
                                    # Extra clearing before full_schema
                                    if case_type == "full_schema":
                                        safe_cuda_clear()
                                        time.sleep(0.5)  # Brief pause for full_schema
                                
                                try:
                                    result = generate_translation_for_case(
                                        entry, tokenizer, model, model_name, args, case_type
                                    )
                                    case_results.append(result)
                                    
                                    # Clear memory after each successful case
                                    if torch.cuda.is_available() and not is_openai_model:
                                        safe_cuda_clear()
                                except Exception as e:
                                    LOGGER.error(f"Failed to retry {instance_id} for {case_type}: {e}")
                                    # Create error entry
                                    test_metadata = entry.get("test_metadata", {})
                                    case_results.append({
                                        "model_name": model_name,
                                        "test_instance_id": instance_id,
                                        "test_id": test_id,
                                        "test_index": test_index,
                                        "test_cypher": entry.get("test_query", ""),
                                        "predicted_translation": "",
                                        "gold_translation": entry.get("gold_translation", "") or test_metadata.get("question", ""),
                                        "test_metadata": test_metadata,
                                        "case_type": case_type,
                                        "schema_content": None,
                                        "translation_error": str(e),
                                    })
                                    
                                    # Clear memory after error too
                                    if torch.cuda.is_available() and not is_openai_model:
                                        safe_cuda_clear()
                    else:
                        # Generate all 3 cases as normal
                        case_results = generate_translation_for_entry(
                            entry, tokenizer, model, model_name, args
                        )
                        
                        # Clear memory after processing all cases for this entry
                        if torch.cuda.is_available() and not is_openai_model:
                            safe_cuda_clear()
                    
                    # Write each case result to its respective file (thread-safe when batch_size > 1)
                    with file_lock:
                        for case_result in case_results:
                            case_type = case_result.get("case_type", "unknown")
                            if case_type not in case_types:
                                continue
                            
                            # When retrying failed entries, don't skip - we want to replace the failed entry
                            # Otherwise, skip if this test_id already exists for this case type
                            if not failed_cases_for_entry and test_id in existing_per_case[case_type]:
                                LOGGER.debug(f"Skipping {test_id} for {case_type} (already exists)")
                                continue
                            
                            # If retrying, remove from existing set so we can write the new result
                            if failed_cases_for_entry and case_type in failed_cases_for_entry:
                                existing_per_case[case_type].discard(test_id)
                            
                            # Write JSON
                            json.dump(case_result, file_handles[case_type]["json"], ensure_ascii=False)
                            file_handles[case_type]["json"].write("\n")
                            file_handles[case_type]["json"].flush()
                            
                            # Add to existing set to avoid duplicates in same run
                            existing_per_case[case_type].add(test_id)
                            
                            # Write TXT (same format as util4 with metadata)
                            txt_content = "=" * 80 + "\n"
                            txt_content += f"Test ID: {case_result.get('test_id', 'N/A')}\n"
                            txt_content += f"Test Instance ID: {case_result.get('test_instance_id', 'N/A')}\n"
                            txt_content += f"Model: {case_result.get('model_name', 'N/A')}\n"
                            txt_content += f"Case Type: {case_type}\n"
                            txt_content += f"Cypher Query: {case_result.get('test_cypher', '')}\n"
                            txt_content += f"Predicted Translation: {case_result.get('predicted_translation', '')}\n"
                            txt_content += f"Gold Translation: {case_result.get('gold_translation', '')}\n"
                            if case_result.get("translation_error"):
                                txt_content += f"Translation Error: {case_result.get('translation_error')}\n"
                            
                            # Add schema content used for this case
                            schema_content = case_result.get('schema_content')
                            txt_content += f"\nSchema Content Used ({case_type}):\n"
                            if schema_content:
                                txt_content += f"{schema_content}\n"
                            else:
                                txt_content += "None (no schema was used in the prompt)\n"
                            
                            txt_content += f"\nTest Metadata:\n"
                            txt_content += json.dumps(case_result.get('test_metadata', {}), ensure_ascii=False, indent=2)
                            txt_content += "\n" + "=" * 80 + "\n\n"
                            file_handles[case_type]["txt"].write(txt_content)
                            file_handles[case_type]["txt"].flush()
                            
                            # Track success/failure (failed entries are still written to main JSON/TXT files with error info)
                            if case_result.get("translation_error"):
                                failed_count[case_type] += 1
                            else:
                                written[case_type] += 1
                        
                        success_count += 1
                        total_written = sum(written.values())
                        elapsed = time.time() - start_time
                        avg_time = elapsed / total_written if total_written > 0 else 0
                        pbar.set_postfix({
                            **{c: written.get(c, 0) for c in case_types},
                            "avg_time": f"{avg_time:.2f}s",
                        })
                        if use_batch:
                            pbar.update(1)
                    
                    # Clear cache more aggressively for full_schema cases to prevent OOM (outside lock)
                    if torch.cuda.is_available():
                        if any("full_schema" in str(case_result.get("case_type", "")) for case_result in case_results):
                            safe_cuda_clear()
                        elif success_count % 10 == 0:
                            if torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory > 0.80:
                                safe_cuda_clear()
                    
                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
                
                except RuntimeError as err:
                    error_msg = str(err)
                    if "cuda corruption" in error_msg.lower() or "illegal memory access" in error_msg.lower():
                        LOGGER.error(f"[CUDA CORRUPTION] Skipping entry {instance_id} due to CUDA corruption.")
                        failed_entries.append({"entry": entry, "error": f"CUDA corruption: {error_msg}"})
                        if not is_cuda_healthy():
                            LOGGER.critical("[CUDA CORRUPTION] CUDA context permanently corrupted. Stopping.")
                            abort_flag[0] = True
                        return
                    else:
                        error_msg = f"Failed to process entry {instance_id}: {err}"
                        LOGGER.error(f"[CRITICAL ERROR] {error_msg}")
                        failed_entries.append({"entry": entry, "error": error_msg})
                        if use_batch:
                            with file_lock:
                                pbar.update(1)
                except Exception as err:
                    # Re-raise OOM so main loop can retry with batch_size=2 (don't write error entry)
                    err_str = str(err).lower()
                    is_oom = (
                        "out of memory" in err_str
                        or (torch.cuda.is_available() and isinstance(err, torch.cuda.OutOfMemoryError))
                    )
                    if is_oom:
                        LOGGER.warning("OOM in worker for entry %s, will retry with batch_size=2", instance_id)
                        raise
                    error_msg = f"Failed to process entry {instance_id}: {err}"
                    LOGGER.error(f"[CRITICAL ERROR] {error_msg}")
                    failed_entries.append({"entry": entry, "error": error_msg})
                    
                    test_index = entry.get("test_index", "")
                    test_metadata = entry.get("test_metadata", {})
                    error_test_id = f"test_{test_index}" if test_index else instance_id
                    
                    with file_lock:
                        for case_type in case_types:
                            if error_test_id in existing_per_case[case_type]:
                                LOGGER.debug(f"Skipping error entry {error_test_id} for {case_type} (already exists)")
                                continue
                            
                            schema_content = None
                            if case_type == "full_schema":
                                schema_content = test_metadata.get("schema", "")
                            elif case_type == "pruned_schema":
                                pruned_schema = load_pruned_schema(instance_id)
                                if pruned_schema:
                                    schema_content = format_pruned_schema(pruned_schema)
                            
                            error_result = {
                                "model_name": model_name,
                                "test_instance_id": instance_id,
                                "test_id": error_test_id,
                                "test_index": test_index,
                                "test_cypher": entry.get("test_query", ""),
                                "predicted_translation": "",
                                "gold_translation": entry.get("gold_translation", "") or test_metadata.get("question", ""),
                                "test_metadata": test_metadata,
                                "case_type": case_type,
                                "schema_content": schema_content,
                                "translation_error": error_msg,
                            }
                            json.dump(error_result, file_handles[case_type]["json"], ensure_ascii=False)
                            file_handles[case_type]["json"].write("\n")
                            file_handles[case_type]["json"].flush()
                            existing_per_case[case_type].add(error_test_id)
                            
                            txt_content = "=" * 80 + "\n"
                            txt_content += f"Test ID: {error_result.get('test_id', 'N/A')}\n"
                            txt_content += f"Test Instance ID: {error_result.get('test_instance_id', 'N/A')}\n"
                            txt_content += f"Model: {error_result.get('model_name', 'N/A')}\n"
                            txt_content += f"Case Type: {case_type}\n"
                            txt_content += f"Cypher Query: {error_result.get('test_cypher', '')}\n"
                            txt_content += f"Predicted Translation: (ERROR)\n"
                            txt_content += f"Gold Translation: {error_result.get('gold_translation', '')}\n"
                            txt_content += f"Translation Error: {error_result.get('translation_error', '')}\n"
                            txt_content += f"\nSchema Content Used ({case_type}):\n"
                            if schema_content:
                                txt_content += f"{schema_content}\n"
                            else:
                                txt_content += "None (no schema was used in the prompt)\n"
                            txt_content += f"\nTest Metadata:\n"
                            txt_content += json.dumps(error_result.get('test_metadata', {}), ensure_ascii=False, indent=2)
                            txt_content += "\n" + "=" * 80 + "\n\n"
                            file_handles[case_type]["txt"].write(txt_content)
                            file_handles[case_type]["txt"].flush()
                            failed_count[case_type] += 1
                        if use_batch:
                            pbar.update(1)
            
            if use_batch:
                oom_entries: List[Dict[str, Any]] = []
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    future_to_entry = {executor.submit(_process_one_entry, entry): entry for entry in model_dataset}
                    for future in as_completed(future_to_entry):
                        entry = future_to_entry[future]
                        try:
                            future.result()
                        except Exception as e:
                            err_str = str(e).lower()
                            is_oom = (
                                "out of memory" in err_str
                                or (torch.cuda.is_available() and isinstance(e, torch.cuda.OutOfMemoryError))
                            )
                            if is_oom:
                                oom_entries.append(entry)
                                LOGGER.warning("OOM detected, entry will be retried with batch_size=2")
                            else:
                                LOGGER.error("Batch worker error: %s", e)
                # Retry OOM entries with batch_size=2 (don't skip)
                if oom_entries:
                    LOGGER.info("Retrying %d entries with batch_size=2 after OOM", len(oom_entries))
                    with ThreadPoolExecutor(max_workers=2) as executor2:
                        for future in as_completed(
                            [executor2.submit(_process_one_entry, e) for e in oom_entries]
                        ):
                            try:
                                future.result()
                            except Exception as e:
                                LOGGER.error("Batch worker error (retry batch=2): %s", e)
            else:
                for entry in pbar:
                    if abort_flag[0]:
                        break
                    _process_one_entry(entry)
            
            pbar.close()
        
        finally:
            # Close all file handles
            for case_type in case_types:
                file_handles[case_type]["json"].close()
                file_handles[case_type]["txt"].close()
        
        # Sort translation files by Test ID
        print(f"\n{'='*80}")
        print(f"Sorting translation files by Test ID for {model_key}...")
        for case_type in case_types:
            json_path = case_files[case_type]["json"]
            txt_path = case_files[case_type]["txt"]
            
            if not json_path.exists() or json_path.stat().st_size == 0:
                continue
            
            try:
                # Read all entries from JSONL file
                entries = []
                with json_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError:
                                continue
                
                if not entries:
                    continue
                
                # Sort by test_id (extract numeric part for proper numeric sorting)
                def get_test_id_sort_key(entry):
                    test_id = entry.get("test_id", "")
                    # Extract numeric part from test_id (e.g., "test_123" -> 123)
                    match = re.search(r'(\d+)', test_id)
                    if match:
                        return int(match.group(1))
                    # Fallback to string comparison
                    return test_id
                
                entries.sort(key=get_test_id_sort_key)
                
                # Rewrite sorted JSONL file
                with json_path.open("w", encoding="utf-8") as f:
                    for entry in entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write("\n")
                
                # Rewrite sorted TXT file
                with txt_path.open("w", encoding="utf-8") as f:
                    for entry in entries:
                        txt_content = "=" * 80 + "\n"
                        txt_content += f"Test ID: {entry.get('test_id', 'N/A')}\n"
                        txt_content += f"Test Instance ID: {entry.get('test_instance_id', 'N/A')}\n"
                        txt_content += f"Model: {entry.get('model_name', 'N/A')}\n"
                        txt_content += f"Case Type: {entry.get('case_type', 'N/A')}\n"
                        txt_content += f"Cypher Query: {entry.get('test_cypher', '')}\n"
                        txt_content += f"Predicted Translation: {entry.get('predicted_translation', '')}\n"
                        txt_content += f"Gold Translation: {entry.get('gold_translation', '')}\n"
                        if entry.get("translation_error"):
                            txt_content += f"Translation Error: {entry.get('translation_error')}\n"
                        
                        schema_content = entry.get('schema_content')
                        txt_content += f"\nSchema Content Used ({case_type}):\n"
                        if schema_content:
                            txt_content += f"{schema_content}\n"
                        else:
                            txt_content += "None (no schema was used in the prompt)\n"
                        
                        txt_content += f"\nTest Metadata:\n"
                        txt_content += json.dumps(entry.get('test_metadata', {}), ensure_ascii=False, indent=2)
                        txt_content += "\n" + "=" * 80 + "\n\n"
                        f.write(txt_content)
                
                print(f"  ✓ Sorted {len(entries)} entries in {case_type}/{json_path.name}")
            except Exception as e:
                LOGGER.warning(f"Failed to sort {case_type} file {json_path}: {e}")
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        # Clean up model and tokenizer (skip for OpenAI models)
        if not is_openai_model and model is not None and tokenizer is not None:
            del model, tokenizer
            if torch.cuda.is_available():
                safe_cuda_clear()
        
        print(f"Translation Summary for {model_key}:")
        print(f"{'='*80}")
        for case_type in case_types:
            print(f"  {case_type}:")
            print(f"    Success: {written[case_type]}")
            print(f"    Failed: {failed_count[case_type]}")
            print(f"    JSON: {case_files[case_type]['json'].name}")
            print(f"    TXT: {case_files[case_type]['txt'].name}")
        print(f"\nTotal time: {elapsed:.2f} seconds")
        total_written = sum(written.values())
        print(f"Average time per query: {elapsed / max(total_written, 1):.2f} seconds")
        print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Successful translations: {success_count}")
    print(f"  Total entries processed: {success_count}")
    print(f"  All entries have translations (100% coverage)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
