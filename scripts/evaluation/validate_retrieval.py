# scripts/validate_retrieval.py

import os
import sys
import logging
from typing import cast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ir_core.orchestration.rewriter_openai import QueryRewriter

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

# ------------------------------------

# Suppress httpx INFO logs to prevent BrokenPipeError with tqdm in multi-threaded environment
logging.getLogger("httpx").setLevel(logging.WARNING)

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
@hydra.main(config_path="../../conf", config_name="settings", version_base=None)
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
        config=cast(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)),
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
    from ir_core.generation import get_query_rewriter

    query_rewriter = get_query_rewriter(cfg)    # 2. RAG 파이프라인을 초기화할 때 rewriter 인스턴스를 전달합니다.
    generator = get_generator(cfg)
    pipeline = RAGPipeline(
        generator=generator,
        model_name=cfg.pipeline.tool_calling_model,  # Pass model_name for query enhancement
        query_rewriter=query_rewriter,
        tool_prompt_description=tool_desc,
        tool_calling_model=cfg.pipeline.tool_calling_model,
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

    # Debug mode: Log prompts and answers for first few queries
    debug_mode = getattr(cfg, 'debug', False)
    if debug_mode:
        debug_limit = getattr(cfg, 'debug_limit', 3)
        print(f"🐛 Debug mode enabled - will log prompts and answers for first {debug_limit} queries")

    # === ANALYSIS FRAMEWORK INTEGRATION ===
    # The new analysis framework will handle all metrics collection and logging

    # === NEW ANALYSIS FRAMEWORK INTEGRATION ===
    # The new analysis framework handles all metrics collection and analysis internally

    # Prepare data for the new analysis framework
    queries_data = []
    retrieval_results_data = []

    # Use parallel processing for retrieval if enabled
    if cfg.analysis.enable_parallel and len(validation_data) > 1:
        max_workers = cfg.analysis.max_workers
        if max_workers is None:
            # Auto-determine workers: use min(sample_size, reasonable_max)
            max_workers = min(len(validation_data), 4)
        print(f"🔄 Processing {len(validation_data)} queries using {max_workers} parallel workers...")

        def process_single_query(item, idx=0):
            """Process a single query for parallel execution."""
            debug_limit = getattr(cfg, 'debug_limit', 3)  # Define in function scope
            query = item.get("msg", [{}])[0].get("content")
            ground_truth_id = item.get("ground_truth_doc_id")

            if not query or not ground_truth_id:
                return None, None

            query_data = {"msg": [{"content": query}], "ground_truth_doc_id": ground_truth_id}

            # Get retrieval results for this query
            try:
                retrieval_output = pipeline.run_retrieval_only(query)
                if retrieval_output and isinstance(retrieval_output, list) and len(retrieval_output) > 0:
                    retrieval_result = retrieval_output[0]
                    if not isinstance(retrieval_result, dict):
                        print(f"Warning: Expected dict, got {type(retrieval_result)} for query '{query}'")
                        retrieval_result = {"docs": []}
                else:
                    retrieval_result = {"docs": []}

                # Debug mode: Log full pipeline for first few queries
                if debug_mode and idx < debug_limit:
                    print(f"\n🐛 DEBUG Query {idx + 1}: {query}")
                    try:
                        full_answer = pipeline.run(query)
                        print(f"🐛 DEBUG Answer: {full_answer[:200]}..." if len(full_answer) > 200 else f"🐛 DEBUG Answer: {full_answer}")

                        # Log retrieved context
                        docs = retrieval_result.get("docs", [])
                        if docs:
                            print(f"🐛 DEBUG Retrieved {len(docs)} documents:")
                            for i, doc in enumerate(docs[:3]):  # Show first 3 docs
                                content_preview = doc.get("content", "")[:100]
                                score = doc.get("score", 0)
                                print(f"  Doc {i+1} (score: {score:.3f}): {content_preview}...")
                        else:
                            print("🐛 DEBUG No documents retrieved")
                    except Exception as e:
                        print(f"🐛 DEBUG Error in full pipeline: {e}")

                return query_data, retrieval_result
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                return query_data, {"docs": []}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_query, item, idx) for idx, item in enumerate(validation_data)]

            for future in tqdm(as_completed(futures), total=len(validation_data), desc="Validating Queries"):
                query_data, retrieval_result = future.result()
                if query_data is not None:
                    queries_data.append(query_data)
                    retrieval_results_data.append(retrieval_result)
    else:
        # Sequential processing (original logic)
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

    # Phase 4: Enhanced Error Analysis Output
    print("\n--- Phase 4: Enhanced Error Analysis ---")

    if analysis_result.query_understanding_failures:
        print("🔍 Query Understanding Failures:")
        for error_type, count in analysis_result.query_understanding_failures.items():
            if count > 0:
                print(f"  • {error_type}: {count} queries")

    if analysis_result.retrieval_failures:
        print("📊 Retrieval Failures:")
        for error_type, count in analysis_result.retrieval_failures.items():
            if count > 0:
                print(f"  • {error_type}: {count} queries")

    if analysis_result.system_failures:
        print("⚠️  System Failures:")
        for error_type, count in analysis_result.system_failures.items():
            if count > 0:
                print(f"  • {error_type}: {count} queries")

    if analysis_result.domain_error_rates:
        print("🌍 Domain-Specific Error Rates:")
        for domain, rate in analysis_result.domain_error_rates.items():
            print(f"  • {domain}: {rate:.1%} error rate")

    if analysis_result.error_patterns.get("query_length_correlation"):
        corr = analysis_result.error_patterns["query_length_correlation"]
        print(f"📈 Query Length vs Success Correlation: {corr:.3f}")

    print("---------------------------")
    if analysis_result.recommendations:
        print("📋 Recommendations:")
        for rec in analysis_result.recommendations:
            print(f"  • {rec}")
    print("---------------------------")
    if analysis_result.error_recommendations:
        print("🔧 Enhanced Error Analysis Recommendations:")
        for rec in analysis_result.error_recommendations:
            print(f"  • {rec}")
    print("---------------------------")

    if wandb.run is not None:
        print(f"WandB 실행 URL: {wandb.run.url}")
    wandb.finish()


if __name__ == "__main__":
    run()
