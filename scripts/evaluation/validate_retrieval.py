# scripts/validate_retrieval.py

import os
import sys
import json
import logging
from typing import cast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ir_core.orchestration.rewriter_openai import QueryRewriter
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

# ------------------------------------

# Suppress httpx INFO logs to prevent BrokenPipeError with tqdm in multi-threaded environment
logging.getLogger("httpx").setLevel(logging.WARNING)

# OmegaConf가 ${env:VAR_NAME} 구문을 해석할 수 있도록 'env' 리졸버를 등록합니다.
try:
    OmegaConf.register_new_resolver("env", os.getenv)
except ValueError:
    # Resolver already registered, continue
    pass


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

    # Configure logging
    import logging
    from rich.logging import RichHandler

    # Get the root logger
    root_logger = logging.getLogger()
    # Clear any existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a logger for this script
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the lowest level for the logger

    # Prevent messages from being passed to the root logger's handlers
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s')

    # File handler (writes DEBUG level and above to a file)
    log_file = "outputs/logs/evaluation.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Rich console handler (writes INFO level and above to the console)
    rich_handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
    rich_handler.setLevel(logging.INFO)

    # Add both handlers to our specific logger
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)



    # Log startup information
    logger.info("=== Starting Validation Run ===")
    logger.info(f"Validation file: {cfg.data.validation_path}")
    logger.info(f"Debug mode: {getattr(cfg, 'debug', False)}")
    logger.info(f"Debug limit: {getattr(cfg, 'debug_limit', 'unlimited')}")

    # Handle output file generation
    generate_output = getattr(cfg, 'generate_output', False)
    if generate_output:
        # Get default output path from data config
        default_output_path = getattr(cfg.data, 'output_path', 'outputs/submission.jsonl')

        # Check if custom output file is specified
        if hasattr(cfg, 'custom_output_file') and cfg.custom_output_file:
            custom_file = cfg.custom_output_file
            # If custom file doesn't have directory, put it in outputs/
            if '/' not in custom_file and '\\' not in custom_file:
                custom_file = f"outputs/{custom_file}"
            output_path = custom_file
            logger.info(f"Using custom output file: {output_path}")
        else:
            output_path = default_output_path
            logger.info(f"Using default output file: {output_path}")

        # Ask for confirmation
        import questionary
        confirmed = questionary.confirm(
            f"Generate output file: {output_path}?",
            default=True
        ).ask()

        if not confirmed:
            new_path = questionary.text(
                "Enter new output file path:",
                default=output_path
            ).ask()
            if new_path:
                output_path = new_path
            else:
                generate_output = False
                logger.info("Output file generation cancelled")

        if generate_output:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output_path):
                os.remove(output_path)
                logger.info(f"Removed existing output file: {output_path}")
    else:
        output_path = None
        logger.info("Output file generation disabled")

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
                logger.info(f"Merged Config File: {config_file}")
                wandb.log({"config_file_path": config_file})
            else:
                logger.warning(f"Config file not found at expected location: {config_file}")
        except Exception as e:
            logger.error(f"Could not determine config file path: {e}")

    logger.info("--- 검증 실행 시작 (Starting Validation Run) ---")
    logger.info(f"사용할 검증 파일: {cfg.data.validation_path}")
    logger.debug(f"적용된 설정:\n{OmegaConf.to_yaml(cfg)}")

    # --- 설정 오버라이드 (Overriding Settings) ---
    # Hydra 설정을 기존의 전역 settings 객체에 반영합니다.
    # 이는 파이프라인의 다른 부분들이 최신 파라미터를 사용하도록 보장합니다.
    settings.ALPHA = cfg.model.alpha
    settings.RERANK_K = cfg.model.rerank_k
    logger.info(f"Alpha 값을 {settings.ALPHA}(으)로 오버라이드합니다.")
    logger.info(f"Rerank_k 값을 {settings.RERANK_K}(으)로 오버라이드합니다.")

    # RAG 파이프라인을 초기화합니다.
    # 설정에서 지정된 경로의 도구 설명 프롬프트를 읽어옵니다.
    try:
        with open(cfg.prompts.tool_description, "r", encoding="utf-8") as f:
            tool_desc = f.read()
    except FileNotFoundError:
        logger.error(
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
        logger.error(f"오류: '{cfg.data.validation_path}'에서 검증 파일을 찾을 수 없습니다.")
        return

    # --- 샘플 제한 로직 (Sample Limiting Logic) ---
    # cfg.limit 값이 설정된 경우, 데이터셋을 해당 크기만큼만 사용합니다.
    if cfg.limit:
        logger.info(f"데이터셋을 {cfg.limit}개의 샘플로 제한합니다.")
        validation_data = validation_data[: cfg.limit]

    # Debug mode: Limit processing to debug_limit queries for faster debugging
    debug_mode = getattr(cfg, 'debug', False)
    if debug_mode:
        debug_limit = getattr(cfg, 'debug_limit', 3)
        logger.info(f"🐛 Debug mode enabled - processing only first {debug_limit} queries for fast debugging")
        validation_data = validation_data[:debug_limit]

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
        logger.info(f"🔄 Processing {len(validation_data)} queries using {max_workers} parallel workers...")

        # def process_single_query(item, idx=0):
        #     """Process a single query for parallel execution."""
        #     debug_limit = getattr(cfg, 'debug_limit', 3)  # Define in function scope
        #     query = item.get("msg", [{}])[0].get("content")
        #     ground_truth_id = item.get("ground_truth_doc_id")

        # Pass the SINGLE pipeline instance to the worker function
        def process_single_query(item_and_idx):
            """Process a single query for parallel execution."""
            item, idx = item_and_idx
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
                        logger.warning(f"Expected dict, got {type(retrieval_result)} for query '{query}'")
                        retrieval_result = {"docs": []}
                    elif not isinstance(retrieval_result.get("docs"), list):
                        logger.warning(f"Expected docs to be list, got {type(retrieval_result.get('docs'))} for query '{query}'")
                        retrieval_result["docs"] = []
                else:
                    retrieval_result = {"docs": []}

                # Debug mode: Log full pipeline for all queries (since we're limiting processing)
                if debug_mode:
                    query_text = Text(f"Query {idx + 1}: {query}", style="bold yellow")

                    try:
                        full_answer = pipeline.run(query)
                        answer_text = Text(f"Answer: {full_answer[:200]}{'...' if len(full_answer) > 200 else ''}", style="green")

                        debug_content = f"{query_text}\n{answer_text}"

                        # Log retrieved context
                        docs = retrieval_result.get("docs", [])
                        if docs and isinstance(docs, list):
                            docs_text = Text(f"Retrieved {len(docs)} documents:", style="bold blue")
                            debug_content += f"\n{docs_text}"
                            for i, doc in enumerate(docs[:3]):  # Show first 3 docs
                                content_preview = doc.get("content", "")[:100]
                                score = doc.get("score", 0)
                                doc_text = Text(f"  Doc {i+1} (score: {score:.3f}): {content_preview}...", style="dim white")
                                debug_content += f"\n{doc_text}"
                        elif docs and isinstance(docs, str):
                            error_docs_text = Text(f"Retrieval error: {docs[:100]}{'...' if len(docs) > 100 else ''}", style="red")
                            debug_content += f"\n{error_docs_text}"
                        else:
                            no_docs_text = Text("No documents retrieved", style="red")
                            debug_content += f"\n{no_docs_text}"

                        panel = Panel.fit(
                            debug_content,
                            title=f"[bold magenta]🐛 DEBUG Query {idx + 1}[/bold magenta]",
                            border_style="magenta"
                        )
                        rprint(panel)
                    except Exception as e:
                        error_text = Text(f"Error in full pipeline: {e}", style="red")
                        panel = Panel.fit(
                            f"{query_text}\n{error_text}",
                            title=f"[bold magenta]🐛 DEBUG Query {idx + 1}[/bold magenta]",
                            border_style="red"
                        )
                        rprint(panel)

                return query_data, retrieval_result
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                return query_data, {"docs": []}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_query, (item, idx)) for idx, item in enumerate(validation_data)]

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
            if retrieval_output and isinstance(retrieval_output, list) and len(retrieval_output) > 0:
                retrieval_result = retrieval_output[0]
                if isinstance(retrieval_result, dict) and isinstance(retrieval_result.get("docs"), list):
                    retrieval_results_data.append(retrieval_result)
                else:
                    retrieval_results_data.append({"docs": []})
            else:
                retrieval_results_data.append({"docs": []})

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
        logger.info(f"WandB 실행 이름 업데이트: {original_name} -> {updated_name}")

    logger.info("--- 검증 완료 (Validation Complete) ---")
    logger.info(f"검증된 쿼리 수: {analysis_result.total_queries}")
    logger.info(f"MAP Score: {analysis_result.map_score:.4f}")
    logger.info(f"Retrieval Success Rate: {analysis_result.retrieval_success_rate:.1%}")
    logger.info(f"Rewrite Rate: {analysis_result.rewrite_rate:.1%}")

    # Phase 4: Enhanced Error Analysis Output
    logger.info("--- Phase 4: Enhanced Error Analysis ---")

    if analysis_result.query_understanding_failures:
        logger.info("🔍 Query Understanding Failures:")
        for error_type, count in analysis_result.query_understanding_failures.items():
            if count > 0:
                logger.info(f"  • {error_type}: {count} queries")

    if analysis_result.retrieval_failures:
        logger.info("📊 Retrieval Failures:")
        for error_type, count in analysis_result.retrieval_failures.items():
            if count > 0:
                logger.info(f"  • {error_type}: {count} queries")

    if analysis_result.system_failures:
        logger.info("⚠️  System Failures:")
        for error_type, count in analysis_result.system_failures.items():
            if count > 0:
                logger.info(f"  • {error_type}: {count} queries")

    if analysis_result.domain_error_rates:
        logger.info("🌍 Domain-Specific Error Rates:")
        for domain, rate in analysis_result.domain_error_rates.items():
            logger.info(f"  • {domain}: {rate:.1%} error rate")

    if analysis_result.error_patterns.get("query_length_correlation"):
        corr = analysis_result.error_patterns["query_length_correlation"]
        logger.info(f"📈 Query Length vs Success Correlation: {corr:.3f}")

    logger.info("---------------------------")
    if analysis_result.recommendations:
        logger.info("📋 Recommendations:")
        for rec in analysis_result.recommendations:
            logger.info(f"  • {rec}")
    logger.info("---------------------------")
    if analysis_result.error_recommendations:
        logger.info("🔧 Enhanced Error Analysis Recommendations:")
        for rec in analysis_result.error_recommendations:
            logger.info(f"  • {rec}")
    logger.info("---------------------------")

    if wandb.run is not None:
        logger.info(f"WandB 실행 URL: {wandb.run.url}")

    # Generate output file if requested
    if generate_output and output_path:
        logger.info(f"📝 Generating output file: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (query_data, retrieval_result) in enumerate(zip(queries_data, retrieval_results_data)):
                    # Create submission entry
                    submission_entry = {
                        "id": i,
                        "question": query_data["msg"][0]["content"],
                        "retrieved_docs": []
                    }

                    # Add retrieved documents
                    docs = retrieval_result.get("docs", [])
                    if docs and isinstance(docs, list):
                        for doc in docs:
                            if isinstance(doc, dict):
                                submission_entry["retrieved_docs"].append({
                                    "doc_id": doc.get("doc_id", ""),
                                    "score": doc.get("score", 0.0)
                                })

                    # Write JSONL entry
                    f.write(json.dumps(submission_entry, ensure_ascii=False) + '\n')

            logger.info(f"✅ Successfully generated output file: {output_path}")
            logger.info(f"📊 Generated {len(queries_data)} entries")

        except Exception as e:
            logger.error(f"❌ Failed to generate output file: {e}")

    wandb.finish()


if __name__ == "__main__":
    run()
