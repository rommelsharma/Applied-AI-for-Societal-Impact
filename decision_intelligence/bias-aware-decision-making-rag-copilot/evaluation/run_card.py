"""Run card metadata for reproducible eval captures (Phase 4)."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_components.settings import BEDROCK_SETTINGS, RAG_SETTINGS
from shared_components.utilities.path_utils import get_prompts_dir


def _scenario_rel(project_root: Path, scenario_file: Path) -> Path:
    try:
        return scenario_file.relative_to(project_root)
    except ValueError:
        return scenario_file


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_head(project_root: Path) -> tuple[str | None, bool]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=str(project_root),
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=str(project_root),
        )
        is_dirty = bool(dirty.stdout.strip())
        return root.stdout.strip(), is_dirty
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None, False


def build_run_card(
    *,
    project_root: Path,
    scenario_file: Path,
    label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a single JSON-serialisable dict to store beside response payloads."""
    now_utc = datetime.now(timezone.utc)
    prompt_path = get_prompts_dir() / "bias_detection_system_prompt.txt"
    commit, dirty = _git_head(project_root)

    card: dict[str, Any] = {
        "label": label,
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "git_commit": commit,
        "git_dirty": dirty,
        "scenario_file": str(_scenario_rel(project_root, scenario_file)),
        "scenario_file_sha256": _file_sha256(scenario_file),
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": _file_sha256(prompt_path),
        "bedrock_region": BEDROCK_SETTINGS.region,
        "bedrock_chat_model_id": BEDROCK_SETTINGS.chat_model_id,
        "bedrock_embedding_model_id": BEDROCK_SETTINGS.embedding_model_id,
        "bedrock_embedding_dimensions": BEDROCK_SETTINGS.embedding_dimensions,
        "rag_corpus_name": RAG_SETTINGS.corpus_name,
        "rag_chunk_profile": RAG_SETTINGS.chunk_profile,
        "rag_top_k": RAG_SETTINGS.default_top_k,
        "rag_mmr_lambda": RAG_SETTINGS.mmr_lambda,
        "rag_reranker": RAG_SETTINGS.reranker,
        "rag_reranker_candidates": RAG_SETTINGS.reranker_candidates,
        "rag_embed_sentence_windows": RAG_SETTINGS.embed_sentence_windows,
        "rag_embed_sentence_radius": RAG_SETTINGS.embed_sentence_radius,
        "rag_embed_max_windows_per_chunk": RAG_SETTINGS.embed_max_windows_per_chunk,
        "rag_overlap_filter": RAG_SETTINGS.overlap_filter,
        "rag_overlap_chunk_radius": RAG_SETTINGS.overlap_chunk_radius,
        "rag_overlap_pool_multiplier": RAG_SETTINGS.overlap_candidate_pool_multiplier,
        "rag_overlap_mmr_pool_multiplier": RAG_SETTINGS.overlap_mmr_pool_multiplier,
        "rag_context_expand_neighbors": RAG_SETTINGS.context_expand_neighbors,
        "rag_expand_max_chars_per_side": RAG_SETTINGS.expand_max_chars_per_side,
    }
    if extra:
        card["extra"] = extra
    return card
