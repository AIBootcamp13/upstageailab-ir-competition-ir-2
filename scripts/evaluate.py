# scripts/evaluate.py
"""
Hydra를 사용하여 RAG 파이프라인의 평가 프로세스를 실행합니다.

이 스크립트는 평가 파일(예: data/eval.jsonl)을 반복하면서
각 쿼리에 대한 검색 파이프라인을 실행하고, 대회 제출 형식에 맞는
submission 파일을 생성합니다.

Hydra 사용법 예시:
- 기본 설정으로 실행:
  PYTHONPATH=src poetry run python scripts/evaluate.py
- top-k 값 변경하여 실행:
  PYTHONPATH=src poetry run python scripts/evaluate.py params.submission.topk=5
- 평가 파일 경로 변경하여 실행:
  PYTHONPATH=src poetry run python scripts/evaluate.py paths.evaluation=data/validation.jsonl
"""
import os
import sys
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# 프로젝트의 src 디렉토리를 경로에 추가하여 사용자 정의 모듈을 임포트할 수 있게 합니다.
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

_add_src_to_path()
# 경로 추가 후, 필요한 모듈을 임포트합니다.
from ir_core.orchestration.pipeline import RAGPipeline
from ir_core.generation import get_generator
from ir_core.utils import read_jsonl
import json


# Hydra 메인 데코레이터를 사용하여 설정을 관리합니다.
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run(cfg: DictConfig):
    """
    RAG 파이프라인에 대한 평가를 실행합니다.

    Args:
        cfg: Hydra에 의해 주입되는 DictConfig 객체. conf/config.yaml 파일과
             커맨드 라인 오버라이드로부터 설정을 포함합니다.
    """
    print("--- 평가 실행 시작 ---")
    print("사용된 설정:\n" + OmegaConf.to_yaml(cfg))

    # 1. RAG 파이프라인을 초기화합니다.
    try:
        generator = get_generator()
        pipeline = RAGPipeline(generator)
        print("RAG 파이프라인이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"RAG 파이p라인 초기화 실패: {e}")
        return

    # 출력 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    output_dir = os.path.dirname(cfg.paths.submission)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 기존 제출 파일이 있다면 덮어쓰기 위해 삭제
    if os.path.exists(cfg.paths.submission):
        os.remove(cfg.paths.submission)

    # 2. 평가 파일의 각 항목을 처리합니다.
    eval_items = list(read_jsonl(cfg.paths.evaluation))
    # Limit samples if max_samples > 0
    if cfg.params.submission.max_samples > 0:
        eval_items = eval_items[:cfg.params.submission.max_samples]

    print(f"{len(eval_items)}개의 평가 쿼리를 처리합니다...")
    for item in tqdm(eval_items, desc="쿼리 평가 중"):
        query = item.get('msg', [{'content': ''}])[-1].get('content', '')
        eval_id = item.get('eval_id')

        if not query or eval_id is None:
            continue

        # 3. 파이프라인의 검색 전용 메소드를 사용하여 도구 결과를 가져옵니다.
        retrieval_out = pipeline.run_retrieval_only(query)

        standalone_query = query
        docs = []
        if retrieval_out:
            entry = retrieval_out[0]
            standalone_query = entry.get('standalone_query', query)
            docs = entry.get('docs', [])

        # 제출을 위한 top-k ID 추출
        topk = cfg.params.submission.topk
        topk_ids = [d.get('id') for d in docs[:topk]]

        # 검색된 문서를 컨텍스트로 사용하여 답변 생성
        context_texts = [d.get('content', '') for d in docs[:topk]]
        try:
            answer_text = pipeline.generator.generate(query=standalone_query, context_docs=context_texts)
        except Exception as e:
            print(f"경고: eval_id {eval_id}에 대한 답변 생성 실패: {e}")
            answer_text = ""

        # 참조 문서 목록 생성
        references = [{"score": d.get('score'), "content": d.get('content', '')} for d in docs[:topk]]

        record = {
            "eval_id": eval_id,
            "standalone_query": standalone_query,
            "topk": topk_ids,
            "answer": answer_text,
            "references": references,
        }

        # 결과를 즉시 JSONL 파일에 추가합니다.
        with open(cfg.paths.submission, 'a', encoding='utf-8') as outf:
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n--- 평가 완료 ---")
    print(f"제출 파일이 다음 경로에 저장되었습니다: {cfg.paths.submission}")

if __name__ == '__main__':
    # .env 파일 로드를 위해 dotenv 라이브러리를 사용하려고 시도합니다.
    # Hydra가 자동으로 .env를 로드하지만, 명시적으로 호출하면 더 안전합니다.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("경고: python-dotenv가 설치되지 않았습니다. .env 파일이 로드되지 않을 수 있습니다.")

    # OpenAI API 키가 설정되었는지 확인합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    run()
