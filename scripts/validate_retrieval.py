# scripts/validate_retrieval.py
"""
Hydra를 사용하여 RAG 검색 파이프라인을 검증하고 MAP 점수를 계산합니다.

이 스크립트는 최종 제출 파일을 생성하기 전에 다양한 검색 전략과
하이퍼파라미터(예: 리랭킹 alpha 값)를 튜닝하고 평가하는 주요 도구입니다.

Hydra 사용법 예시:
- 기본 설정으로 검증 실행:
  PYTHONPATH=src poetry run python scripts/validate_retrieval.py
- alpha 값을 0.3으로 변경하여 실행:
  PYTHONPATH=src poetry run python scripts/validate_retrieval.py params.retrieval.alpha=0.3
- rerank_k 값을 15로 변경하여 실행:
  PYTHONPATH=src poetry run python scripts/validate_retrieval.py params.retrieval.rerank_k=15
"""
import os
import sys
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# 프로젝트의 src 디렉토리를 경로에 추가합니다.
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

_add_src_to_path()
# 경로 추가 후, 필요한 모듈을 임포트합니다.
from ir_core.config import settings
from ir_core.generation import get_generator
from ir_core.orchestration.pipeline import RAGPipeline
from ir_core.utils import read_jsonl
from ir_core.evaluation.core import mean_average_precision


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run(cfg: DictConfig):
    """
    검증 데이터셋에 대한 검색 성능을 MAP 점수로 평가합니다.

    Args:
        cfg: Hydra에 의해 주입되는 DictConfig 객체.
    """
    print("--- 검증 실행 시작 ---")
    print("사용된 설정:\n" + OmegaConf.to_yaml(cfg))

    # Hydra 설정을 Pydantic 설정 객체에 반영합니다.
    # 이렇게 하면 파이프라인의 다른 부분들이 업데이트된 하이퍼파라미터를 사용할 수 있습니다.
    settings.ALPHA = cfg.params.retrieval.alpha
    settings.RERANK_K = cfg.params.retrieval.rerank_k
    settings.BM25_K = cfg.params.retrieval.bm25_k

    # 도구 호출 결정을 위해 생성기가 필요합니다.
    generator = get_generator()
    pipeline = RAGPipeline(generator)

    # 검증 데이터를 읽어옵니다.
    try:
        validation_data = list(read_jsonl(cfg.paths.validation))
        # Limit samples if max_samples > 0
        if cfg.params.retrieval.max_samples > 0:
            validation_data = validation_data[:cfg.params.retrieval.max_samples]
    except FileNotFoundError:
        print(f"오류: '{cfg.paths.validation}'에서 검증 파일을 찾을 수 없습니다.")
        return

    # 최종 MAP 계산을 위해 결과를 저장합니다.
    all_results = []

    debug_count = 0
    for item in tqdm(validation_data, desc="쿼리 검증 중"):
        messages = item.get("msg", [])
        ground_truth_id = item.get("ground_truth_doc_id")

        if not messages or not ground_truth_id:
            continue

        retrieval_output = pipeline.run_retrieval_only(messages)

        predicted_docs = []
        if retrieval_output:
            predicted_docs = retrieval_output[0].get("docs", [])

        predicted_ids = [doc["id"] for doc in predicted_docs]
        relevant_ids = [ground_truth_id]

        # Debug: print first 3 queries
        if debug_count < 3:
            print(f"\nDebug Query {debug_count+1}:")
            print(f"Original: {messages[-1]['content']}")
            print(f"Ground Truth ID: {ground_truth_id}")
            print(f"Predicted IDs: {predicted_ids}")
            print(f"Match: {ground_truth_id in predicted_ids}")
            debug_count += 1

        all_results.append((predicted_ids, relevant_ids))

    # 최종 MAP 점수를 계산합니다.
    map_score = mean_average_precision(all_results)

    print("\n--- 검증 완료 ---")
    print(f"총 검증된 쿼리 수: {len(all_results)}")
    print(f"MAP 점수: {map_score:.4f}")
    print("---------------------------")

if __name__ == "__main__":
    run()
