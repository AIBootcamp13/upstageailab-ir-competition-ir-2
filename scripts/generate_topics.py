# scripts/generate_topics.py
"""
BERTopic 모델을 사용하여 documents.jsonl의 모든 문서에 대한 주제(토픽)를 생성합니다.

이 스크립트는 문서 모음을 읽고, 내용에 기반하여 의미론적 주제를 식별한 다음,
각 문서 ID를 해당 주제 번호 및 관련 키워드에 매핑하는 출력 파일을 생성합니다.
이 정보는 reindex.py 스크립트에서 Elasticsearch 인덱스를 보강하는 데 사용됩니다.

실행 방법:
PYTHONPATH=src poetry run python scripts/generate_topics.py
"""
import os
import sys
import json
from tqdm import tqdm

# 프로젝트의 src 디렉토리를 경로에 추가합니다.
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

_add_src_to_path()
from ir_core.utils import read_jsonl

# BERTopic 라이브러리를 안전하게 임포트합니다.
try:
    from bertopic import BERTopic
except ImportError:
    print("오류: 'bertopic' 라이브러리가 설치되지 않았습니다.")
    print("해결책: poetry add bertopic")
    sys.exit(1)

def run(
    input_path: str = "data/documents.jsonl",
    output_path: str = "data/doc_topics.jsonl"
):
    """
    문서 모음에 대한 주제 모델링을 실행하고 결과를 저장합니다.

    Args:
        input_path: 입력 documents.jsonl 파일 경로.
        output_path: 각 문서의 주제 정보를 저장할 출력 .jsonl 파일 경로.
    """
    print("--- 주제 모델링 시작 ---")
    print(f"입력 문서: {input_path}")

    # 1. 문서 로드
    print("문서 내용을 메모리로 로드하는 중...")
    docs = list(read_jsonl(input_path))
    contents = [doc.get("content", "") for doc in docs]
    doc_ids = [doc.get("docid") for doc in docs]
    print(f"{len(contents)}개의 문서를 로드했습니다.")

    # 2. BERTopic 모델 초기화 및 학습
    # 한국어에 적합한 임베딩 모델을 사용합니다.
    # verbose=True로 설정하여 진행 상황을 확인합니다.
    print("BERTopic 모델을 초기화하고 학습을 시작합니다. 이 과정은 시간이 걸릴 수 있습니다...")
    topic_model = BERTopic(
        embedding_model="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        verbose=True,
        language="multilingual" # 한국어 지원을 위해 'multilingual' 설정
    )

    # 모델 학습 및 각 문서에 대한 주제 할당
    topics, _ = topic_model.fit_transform(contents)
    print("모델 학습 및 주제 할당이 완료되었습니다.")

    # 3. 결과 저장
    print(f"주제 정보를 '{output_path}'에 저장하는 중...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, topic_id in tqdm(zip(doc_ids, topics), total=len(doc_ids), desc="결과 저장"):
            if not doc_id:
                continue

            # 각 주제에 대한 키워드 추출
            # BERTopic은 get_topic을 사용하여 주제의 대표 단어를 반환합니다.
            topic_keywords = topic_model.get_topic(topic_id)
            # 키워드를 공백으로 구분된 문자열로 변환합니다. (예: "단어1 단어2 단어3")
            keywords_str = " ".join([word for word, _ in topic_keywords]) if topic_keywords else ""

            record = {
                "docid": doc_id,
                "topic_id": topic_id,
                "topic_keywords": keywords_str
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("\n--- 주제 모델링 완료 ---")
    # 생성된 상위 주제 몇 개를 출력하여 결과를 확인합니다.
    print("생성된 상위 5개 주제:")
    print(topic_model.get_topic_info().head(6))


if __name__ == "__main__":
    run()
