#!/usr/bin/env python3
from __future__ import annotations

#!/usr/bin/env python3
"""
Reindex CLI — JSONL 파일을 Elasticsearch로 일괄 인덱싱합니다.
이제 생성된 주제 정보를 문서에 병합하는 기능이 포함됩니다.

사용법:
1. 먼저 주제 모델링을 실행하여 `doc_topics.jsonl` 파일을 생성합니다.
   PYTHONPATH=src poetry run python scripts/generate_topics.py

2. 주제 정보와 함께 재색인을 실행합니다.
   PYTHONPATH=src poetry run python scripts/reindex.py data/documents.jsonl --index test
"""

import argparse
import os
import sys
import json
import tempfile
from typing import Dict, Any

def _add_src_to_path() -> None:
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# 메인 함수 시작 전에 경로 추가
_add_src_to_path()
from ir_core import api
from ir_core.utils import read_jsonl

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reindex a JSONL file into Elasticsearch with topic information")
    parser.add_argument("jsonl", help="Path to documents.jsonl file")
    parser.add_argument("--index", "-i", help="Index name (overrides config)")
    parser.add_argument("--batch-size", "-b", type=int, default=500, help="Bulk batch size")
    parser.add_argument("--topics-path", default="data/doc_topics.jsonl", help="Path to the doc_topics.jsonl file")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to ES; simulate indexing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-batch timing and ETA")
    args = parser.parse_args(argv)

    # 1. 주제 정보 로드
    topic_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.topics_path):
        print(f"'{args.topics_path}'에서 주제 정보를 로드하는 중...")
        for topic_info in read_jsonl(args.topics_path):
            docid = topic_info.pop("docid", None)
            if docid:
                topic_map[docid] = topic_info
        print(f"{len(topic_map)}개의 문서에 대한 주제 정보를 로드했습니다.")
    else:
        # 2. 주제 정보와 함께 문서 색인 API 호출
        print(f"재색인 시작: {args.jsonl} -> index={args.index or 'configured'} batch={args.batch_size}")
        # api.index_documents_from_jsonl does not accept a 'topic_map' parameter, so merge
        # the topic information into the documents first and write to a temporary JSONL.
        jsonl_to_index = args.jsonl
    temp_path = None
    if topic_map:
        print("문서에 주제 정보를 병합하는 중...")
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", dir=".")
        temp_path = tmp.name
        try:
            for doc in read_jsonl(args.jsonl):
                # try common id keys to match the topic_map
                docid = None
                for k in ("docid", "id", "_id"):
                    if k in doc:
                        docid = doc.get(k)
                        break
                if docid is not None and str(docid) in topic_map:
                    # attach the topic info under 'topics' key
                    doc["topics"] = topic_map[str(docid)]
                json.dump(doc, tmp, ensure_ascii=False)
                tmp.write("\n")
        finally:
            tmp.close()
        jsonl_to_index = temp_path

    try:
        api.index_documents_from_jsonl(
            jsonl_to_index,
            index_name=args.index,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return 0
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
