"""
Centralized utilities for resolving project paths.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute project root path."""

    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    return get_data_dir() / "raw"


def get_metadata_dir() -> Path:
    return get_data_dir() / "metadata"


def get_processed_dir() -> Path:
    return get_data_dir() / "processed"


def get_parsed_text_dir() -> Path:
    return get_processed_dir() / "parsed_text"


def get_chunks_dir() -> Path:
    return get_processed_dir() / "chunks"


def get_knowledge_dir() -> Path:
    return get_processed_dir() / "knowledge"


def get_vector_store_dir() -> Path:
    return get_processed_dir() / "vector_store"


def get_prompts_dir() -> Path:
    return get_project_root() / "prompts"


def get_evaluation_dir() -> Path:
    return get_project_root() / "evaluation"


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
