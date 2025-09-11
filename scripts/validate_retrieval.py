# scripts/validate_retrieval.py

import os
import sys
from tqdm import tqdm

# --- 새로운 임포트 (New Imports) ---
# Hydra와 OmegaConf는 설정 관리를 위해, wandb는 실험 추적을 위해 사용됩니다.
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
# ------------------------------------

# OmegaConf가 ${env:VAR_NAME} 구문을 해석할 수 있도록 'env' 리졸버를 등록합니다.
OmegaConf.register_new_resolver("env", os.getenv)

# Add the src directory to the Python path
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from ir_core.utils.wandb import generate_run_name

# --- Hydra 데코레이터 적용 (Applying the Hydra Decorator) ---
# @hydra.main은 이 스크립트의 진입점을 정의하며, 설정 파일의 경로를 지정합니다.
# 이제 이 스크립트는 'fire' 대신 Hydra를 통해 실행됩니다.
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    """
    Hydra 설정을 사용하여 검증 데이터셋에 대한 검색 파이프라인을 평가하고,
    그 결과를 WandB에 로깅합니다.

    Args:
        cfg (DictConfig): Hydra에 의해 관리되는 설정 객체.
                           conf/config.yaml 파일과 커맨드라인 오버라이드를 통해 채워집니다.
    """
    _add_src_to_path()

    # 필요한 모듈들을 지연 임포트합니다.
    from ir_core.config import settings
    from ir_core.generation import get_generator
    from ir_core.orchestration.pipeline import RAGPipeline
    from ir_core.utils import read_jsonl
    from ir_core.evaluation.core import mean_average_precision, average_precision

  # --- WandB 초기화 (WandB Initialization) ---
    # OmegaConf.set_struct를 사용하여 cfg 객체를 임시로 수정 가능하게 만듭니다.
    OmegaConf.set_struct(cfg, False)
    # 'validate' 유형에 맞는 실행 이름 접두사를 설정합니다.
    cfg.wandb.run_name_prefix = cfg.wandb.run_name.validate
    run_name = generate_run_name(cfg)
    OmegaConf.set_struct(cfg, True) # 다시 읽기 전용으로 변경

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        job_type="validation" # 작업 유형을 'validation'으로 지정
    )
    print("--- 검증 실행 시작 (Starting Validation Run) ---")
    print(f"사용할 검증 파일: {cfg.data.validation_path}")
    print(f"적용된 설정:\n{OmegaConf.to_yaml(cfg)}")

    # --- 설정 오버라이드 (Overriding Settings) ---
    # Hydra 설정을 기존의 전역 settings 객체에 반영합니다.
    # 이는 파이프라인의 다른 부분들이 최신 파라미터를 사용하도록 보장합니다.
    settings.ALPHA = cfg.model.alpha
    settings.RERANK_K = cfg.model.rerank_k
    print(f"Alpha 값을 {settings.ALPHA}(으)로 오버라이드합니다.")
    print(f"Rerank_k 값을 {settings.RERANK_K}(으)로 오버라이드합니다.")

    # RAG 파이프라인을 초기화합니다.
    generator = get_generator(cfg)
    pipeline = RAGPipeline(generator)

    # 검증 데이터를 읽어옵니다.
    try:
        validation_data = list(read_jsonl(cfg.data.validation_path))
    except FileNotFoundError:
        print(f"오류: '{cfg.data.validation_path}'에서 검증 파일을 찾을 수 없습니다.")
        return

    # --- 샘플 제한 로직 (Sample Limiting Logic) ---
    # cfg.limit 값이 설정된 경우, 데이터셋을 해당 크기만큼만 사용합니다.
    if cfg.limit:
        print(f"데이터셋을 {cfg.limit}개의 샘플로 제한합니다.")
        validation_data = validation_data[:cfg.limit]

    # 최종 MAP 점수 계산 및 WandB 테이블 로깅을 위한 결과 저장
    all_results_for_map = []
    wandb_table_data = []

    for item in tqdm(validation_data, desc="Validating Queries"):
        query = item.get("msg", [{}])[0].get("content")
        ground_truth_id = item.get("ground_truth_doc_id")

        if not query or not ground_truth_id:
            continue

        # 검색 파이프라인만 실행하여 예측된 문서 ID 목록을 가져옵니다.
        retrieval_output = pipeline.run_retrieval_only(query)

        predicted_docs = retrieval_output[0].get("docs", []) if retrieval_output else []
        predicted_ids = [doc["id"] for doc in predicted_docs]
        relevant_ids = [ground_truth_id]

        all_results_for_map.append((predicted_ids, relevant_ids))

        # --- WandB 테이블 데이터 준비 (Preparing WandB Table Data) ---
        ap_score = average_precision(predicted_ids, relevant_ids)
        # 테이블에는 쿼리, 정답 ID, 예측 ID 목록, AP 점수를 기록합니다.
        wandb_table_data.append([
            query,
            ground_truth_id,
            predicted_ids[:settings.RERANK_K], # 상위 K개만 표시
            f"{ap_score:.4f}"
        ])

    # 최종 MAP 점수를 계산합니다.
    map_score = mean_average_precision(all_results_for_map)

    # --- WandB에 결과 로깅 (Logging Results to WandB) ---
    # 1. 상세 결과 테이블을 생성하고 로깅합니다.
    results_table = wandb.Table(
        columns=["Query", "Ground Truth ID", "Predicted IDs", "AP Score"]
    )
    for row in wandb_table_data:
        results_table.add_data(*row)
    wandb.log({"Validation Results": results_table})

    # 2. 최종 MAP 점수를 로깅합니다.
    wandb.log({"map_score": map_score})

    print("\n--- 검증 완료 (Validation Complete) ---")
    print(f"검증된 쿼리 수: {len(all_results_for_map)}")
    print(f"MAP Score: {map_score:.4f}")
    print("---------------------------")
    print(f"WandB 실행 URL: {wandb.run.url}")
    wandb.finish()


if __name__ == "__main__":
    run()