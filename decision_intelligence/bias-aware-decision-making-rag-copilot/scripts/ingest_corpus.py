#!/usr/bin/env python3
"""Offline corpus ingest: ``build_knowledge_base`` (+ synthesis) then optional index build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from data_pipeline.build_knowledge_base import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--build-index", action="store_true", help="Also run scripts/build_index.py after ingest.")
    args = parser.parse_args()

    run_pipeline(corpus=args.corpus, profile=args.profile, skip_parse=args.skip_parse)

    if args.build_index:
        from data_pipeline.build_bm25_index import build_bm25
        from data_pipeline.build_vector_index import build_index

        build_bm25(corpus=args.corpus)
        build_index(corpus=args.corpus)


if __name__ == "__main__":
    main()
