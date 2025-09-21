#!/usr/bin/env python3
from __future__ import annotations

"""Reindex CLI — JSONL 파일을 Elasticsearch로 일괄 인덱싱합니다.

간단한 예:
  PYTHONPATH=src uv run python scripts/maintenance/reindex.py data/documents_ko.jsonl --index test --batch-size 500
  # Or use the current configured data file:
  # PYTHONPATH=src uv run python scripts/maintenance/reindex.py $(python -c "from switch_config import get_current_documents_path; print(get_current_documents_path())") --index test --batch-size 500

이 스크립트는 `src/`를 sys.path에 추가해 로컬 패키지를 바로 임포트할 수 있게 합니다.
"""

import argparse
import os
import sys


def _add_src_to_path() -> None:
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def main(argv: list[str] | None = None) -> int:
    _add_src_to_path()
    parser = argparse.ArgumentParser(
        description="Reindex a JSONL file into Elasticsearch"
    )
    parser.add_argument("jsonl", help="Path to JSONL file")
    parser.add_argument("--index", "-i", help="Index name (overrides config)")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=500, help="Bulk batch size"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write to ES; simulate indexing"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-batch timing and ETA"
    )
    args = parser.parse_args(argv)

    # Import here so that src/ is already on sys.path
    from ir_core import api

    print(
        f"Reindexing {args.jsonl} -> index={args.index or 'configured'} batch={args.batch_size}"
    )
    api.index_documents_from_jsonl(
        args.jsonl,
        index_name=args.index,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
