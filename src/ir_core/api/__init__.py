"""High-level facade package for the project.

This module exposes a stable set of helper functions that scripts
and CLIs can import. Implementations live in subpackages.
"""
import time
from typing import Dict, Any, Optional, Callable, Iterable

# --- Local Project Imports ---
from ..config import settings
from ..embeddings.core import load_model, encode_texts, encode_query
from ..retrieval.core import sparse_retrieve, dense_retrieve, hybrid_retrieve
from ..evaluation.core import precision_at_k, mrr, mean_average_precision
from ..utils.core import read_jsonl, write_jsonl


# --- Public API Surface ---
# 이 __all__ 리스트는 `from ir_core.api import *` 사용 시
# 어떤 함수와 변수를 공개할지 정의합니다.
__all__ = [
    "settings",
    "load_model",
    "encode_texts",
    "encode_query",
    "sparse_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "precision_at_k",
    "mrr",
    "mean_average_precision",
    "read_jsonl",
    "write_jsonl",
    "index_documents_from_jsonl", # 색인 함수를 명시적으로 포함
]


def index_documents_from_jsonl(
    jsonl_path: str,
    index_name: Optional[str] = None,
    batch_size: int = 500,
    *,
    dry_run: bool = False,
    verbose: bool = False,
    dedupe: bool = False,
    topic_map: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    JSONL 파일의 문서를 bulk API를 사용하여 Elasticsearch에 색인합니다.
    이제 문서 ID를 키로 하는 주제 정보 맵을 받아 문서를 보강할 수 있습니다.

    Args:
        jsonl_path: 색인할 문서가 포함된 .jsonl 파일의 경로.
        index_name: 사용할 Elasticsearch 인덱스 이름. None이면 설정 파일의 값을 사용.
        batch_size: Elasticsearch bulk API로 한 번에 보낼 문서의 수.
        dry_run: True이면 Elasticsearch에 실제로 쓰지 않고 색인 과정을 시뮬레이션.
        verbose: True이면 배치별 처리 시간 등 상세 정보를 출력.
        dedupe: True이면 docid를 기준으로 중복된 문서를 건너뜀.
        topic_map: docid를 키로 하고 주제 정보를 값으로 갖는 딕셔너리.
                   문서에 주제 정보를 추가하는 데 사용됨.
    """
    # 임포트 시점의 사이드 이펙트를 피하기 위해 함수 내에서 infra 모듈을 임포트합니다.
    from .. import infra
    from elasticsearch.helpers import bulk

    es = infra.get_es()
    idx = index_name or settings.INDEX_NAME
    topic_map = topic_map or {}

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = None

    # tqdm이 설치되어 있으면 진행률 표시줄을 사용합니다.
    try:
        from tqdm import tqdm as _tqdm
        use_tqdm = True
    except ImportError:
        _tqdm = lambda x, **kwargs: x
        use_tqdm = False

    indexed = 0
    batch = []
    iterator = read_jsonl(jsonl_path)
    seen_docids = set() if dedupe else None

    # tqdm을 사용하여 진행률 표시
    progress_iterator = _tqdm(iterator, total=total, desc=f"Indexing -> {idx}", disable=not use_tqdm)

    start_time = time.time()

    for doc in progress_iterator:
        doc_id = doc.get("docid") or doc.get("id") or doc.get("_id")

        if not doc_id:
            continue

        # 중복 제거 로직
        if dedupe and seen_docids is not None:
            if doc_id in seen_docids:
                if verbose:
                    print(f"Skipping duplicate docid: {doc_id}")
                continue
            seen_docids.add(doc_id)

        # --- 주제 정보 병합 로직 ---
        if doc_id in topic_map:
            topic_info = topic_map[doc_id]
            doc.update({
                "topic_id": topic_info.get("topic_id"),
                "topic_keywords": topic_info.get("topic_keywords")
            })

        action = {"_index": idx, "_id": doc_id, "_source": doc}
        batch.append(action)

        if len(batch) >= batch_size:
            try:
                if not dry_run:
                    client_with_opts = es.options(request_timeout=30)
                    success, _ = bulk(client_with_opts, batch, raise_on_error=True)
                    indexed += success
                else:
                    indexed += len(batch)
                batch = []  # 성공적으로 처리 후 배치 초기화
            except Exception as exc:
                print(f"Bulk indexing failure (batch ending with id={doc_id}): {exc}", flush=True)

    # 마지막 남은 배치 처리
    if batch:
        try:
            if not dry_run:
                client_with_opts = es.options(request_timeout=30)
                success, _ = bulk(client_with_opts, batch, raise_on_error=True)
                indexed += success
            else:
                indexed += len(batch)
        except Exception as exc:
            print(f"Final bulk indexing failure: {exc}", flush=True)

    elapsed = time.time() - start_time
    print(f"'{idx}' 인덱스에 총 {indexed}개의 문서를 색인했습니다. (소요 시간: {elapsed:.2f}초)")

