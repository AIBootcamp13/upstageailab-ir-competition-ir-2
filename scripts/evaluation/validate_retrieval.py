# scripts/validate_retrieval.py

import os
import sys
from tqdm import tqdm
from ir_core.orchestration.rewriter import QueryRewriter

import hydra
from hydra.core.hydra_config import HydraConfig
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
    from ir_core.analysis.core import RetrievalAnalyzer
    from ir_core.utils.wandb_logger import WandbAnalysisLogger

    # --- WandB 초기화 (WandB Initialization) ---
    # OmegaConf.set_struct를 사용하여 cfg 객체를 임시로 수정 가능하게 만듭니다.
    OmegaConf.set_struct(cfg, False)
    # 'validate' 유형에 맞는 실행 이름 접두사를 설정합니다.
    cfg.wandb.run_name_prefix = cfg.wandb.run_name.validate
    run_name = generate_run_name(cfg)
    OmegaConf.set_struct(cfg, True)  # 다시 읽기 전용으로 변경

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        job_type="validation",  # 작업 유형을 'validation'으로 지정
    )

    # 설정 파일 경로 로깅 (Log Config File Path)
    if hasattr(cfg.wandb, "log_config_path") and cfg.wandb.log_config_path:
        try:
            # Hydra가 저장한 설정 파일 경로를 가져옵니다
            config_path = HydraConfig.get().runtime.output_dir
            config_file = os.path.join(config_path, ".hydra", "config.yaml")
            if os.path.exists(config_file):
                print(f"Merged Config File: {config_file}")
                wandb.log({"config_file_path": config_file})
            else:
                print(f"Config file not found at expected location: {config_file}")
        except Exception as e:
            print(f"Could not determine config file path: {e}")

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
    # 설정에서 지정된 경로의 도구 설명 프롬프트를 읽어옵니다.
    try:
        with open(cfg.prompts.tool_description, "r", encoding="utf-8") as f:
            tool_desc = f.read()
    except FileNotFoundError:
        print(
            f"오류: '{cfg.prompts.tool_description}'에서 도구 설명 파일을 찾을 수 없습니다."
        )
        return

    # 1. QueryRewriter 인스턴스를 생성합니다.
    query_rewriter = QueryRewriter(
        model_name=cfg.pipeline.rewriter_model,
        prompt_template_path=cfg.prompts.rephrase_query,
    )

    # 2. RAG 파이프라인을 초기화할 때 rewriter 인스턴스를 전달합니다.
    generator = get_generator(cfg)
    pipeline = RAGPipeline(
        generator=generator,
        query_rewriter=query_rewriter,
        tool_prompt_description=tool_desc,
    )

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
        validation_data = validation_data[: cfg.limit]

    # === ANALYSIS FRAMEWORK INTEGRATION ===
    # The new analysis framework will handle all metrics collection and logging

    # === NEW ANALYSIS FRAMEWORK INTEGRATION ===
    # The new analysis framework handles all metrics collection and analysis internally

    # Prepare data for the new analysis framework
    queries_data = []
    retrieval_results_data = []

    for item in tqdm(validation_data, desc="Validating Queries"):
        query = item.get("msg", [{}])[0].get("content")
        ground_truth_id = item.get("ground_truth_doc_id")

        if not query or not ground_truth_id:
            continue

        queries_data.append(
            {"msg": [{"content": query}], "ground_truth_doc_id": ground_truth_id}
        )

        # Get retrieval results for this query
        retrieval_output = pipeline.run_retrieval_only(query)
        retrieval_results_data.append(
            retrieval_output[0] if retrieval_output else {"docs": []}
        )

    # Initialize the new analysis framework
    analyzer = RetrievalAnalyzer(cfg)
    wandb_logger = WandbAnalysisLogger()

    # Perform comprehensive analysis
    analysis_result = analyzer.analyze_batch(
        queries=queries_data, retrieval_results=retrieval_results_data
    )

    # Log results using the enhanced Wandb logger
    wandb_logger.log_analysis_result(result=analysis_result)

    # Update run name with analysis results
    if wandb.run is not None:
        original_name = wandb.run.name
        updated_name = f"{original_name}-MAP_{analysis_result.map_score:.3f}"
        wandb.run.name = updated_name
        print(f"WandB 실행 이름 업데이트: {original_name} -> {updated_name}")

    print("\n--- 검증 완료 (Validation Complete) ---")
    print(f"검증된 쿼리 수: {analysis_result.total_queries}")
    print(f"MAP Score: {analysis_result.map_score:.4f}")
    print(f"Retrieval Success Rate: {analysis_result.retrieval_success_rate:.1%}")
    print(f"Rewrite Rate: {analysis_result.rewrite_rate:.1%}")
    print("---------------------------")
    if analysis_result.recommendations:
        print("📋 Recommendations:")
        for rec in analysis_result.recommendations:
            print(f"  • {rec}")
    print("---------------------------")
    if wandb.run is not None:
        print(f"WandB 실행 URL: {wandb.run.url}")
    wandb.finish()


if __name__ == "__main__":
    run()
