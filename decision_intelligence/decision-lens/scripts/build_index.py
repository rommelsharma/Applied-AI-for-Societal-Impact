#!/usr/bin/env python3
"""Build BM25 + FAISS indexes for the active corpus (v4 ``processed/index`` layout)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from data_pipeline.build_bm25_index import build_bm25
from data_pipeline.build_vector_index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--bm25-only", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Vector index metadata refresh only (see build_vector_index.py).",
    )
    args = parser.parse_args()

    if not args.skip_bm25:
        build_bm25(corpus=args.corpus)
    if args.bm25_only:
        return
    build_index(corpus=args.corpus, metadata_only=args.metadata_only)


if __name__ == "__main__":
    main()
