# **🚀 프로젝트 사용 가이드**

이 문서는 RAG(Retrieval-Augmented Generation) 프로젝트의 주요 기능을 사용하고, 성능을 튜닝하며, 대회를 위한 제출물을 생성하는 전체 워크플로우를 안내합니다.

## **목차**

1. [초기 설정](#1-초기-설정)
2. [핵심 워크플로우](#2-핵심-워크플로우)
3. [모델 성능 튜닝 및 검증](#3-모델-성능-튜닝-및-검증)
4. [고급 실험 가이드](#4-고급-실험-가이드)

## **1. 초기 설정**

프로젝트를 실행하기 위해 필요한 초기 설정 단계입니다.

### **1.1. 의존성 설치**

Poetry를 사용하여 필요한 모든 라이브러리를 설치합니다.

poetry install

### **1.2. 환경 변수 설정**

프로젝트 루트 디렉토리에 있는 .env.example 파일을 복사하여 .env 파일을 생성합니다. 그 후, 파일 내의 OPENAI_API_KEY 값을 자신의 키로 채워넣습니다.

cp .env.example .env
# nano .env 또는 다른 편집기를 사용하여 API 키를 입력하세요.

### **1.3. 로컬 인프라 실행**

Elasticsearch와 Redis를 로컬 환경에서 실행합니다. 이 스크립트는 필요한 경우 서비스를 자동으로 다운로드하고 설정합니다.

./scripts/run-local.sh start

**참고:** status 명령어로 서비스 상태를 확인할 수 있습니다. (./scripts/run-local.sh status)

### **1.4. 데이터 색인 (Indexing)**

대회에서 제공된 documents.jsonl 파일을 Elasticsearch에 색인합니다. 이 작업은 검색을 위해 필수적이며, 한 번만 수행하면 됩니다.

PYTHONPATH=src poetry run python scripts/reindex.py data/documents.jsonl --index test

## **2. 핵심 워크플로우**

프로젝트의 주요 기능을 실행하는 방법입니다.

### **2.1. 단일 질문으로 RAG 파이프라인 테스트**

run_rag.py 스크립트를 사용하여 단일 질문에 대한 전체 RAG 파이프라인의 동작을 테스트할 수 있습니다. 이 스크립트는 LLM의 도구 호출, 문서 검색, 최종 답변 생성 과정을 모두 포함합니다.

# OpenAI 모델 사용 (기본값)
PYTHONPATH=src poetry run python scripts/run_rag.py "광합성이란 무엇인가요?"

# 로컬 Ollama 모델 사용 (Ollama 서버가 실행 중이어야 함)
PYTHONPATH=src poetry run python scripts/run_rag.py "광합성이란 무엇인가요?" --generator_type ollama

### **2.2. 대회 제출 파일 생성**

evaluate.py 스크립트는 공식 평가 데이터(eval.jsonl)를 사용하여 각 질문에 대한 검색 결과를 생성하고, 이를 대회 제출 형식인 submission.csv 파일로 저장합니다.

PYTHONPATH=src poetry run python scripts/evaluate.py

실행이 완료되면 outputs/ 디렉토리에서 제출 파일을 확인할 수 있습니다.

## **3. 모델 성능 튜닝 및 검증**

최적의 성능을 내는 모델을 찾기 위한 실험 및 검증 과정입니다. 이 과정은 공식 평가 데이터를 사용하기 전에 수행하여 과적합(overfitting)을 방지하는 데 필수적입니다.

### **3.1. 검증 데이터셋 생성**

create_validation_set.py 스크립트를 사용하여 원본 문서에서 질문을 자동으로 생성, 자신만의 검증 데이터셋(validation.jsonl)을 만듭니다.

**왜 필요한가요?**

공식 eval.jsonl에 모델을 반복적으로 테스트하면, 자신도 모르게 해당 데이터셋에만 잘 동작하는 편향된 모델을 만들게 될 위험이 있습니다. 독립적인 검증 데이터셋을 사용하면 일반화 성능을 객관적으로 측정할 수 있습니다.

# 50개의 질문-문서 쌍으로 구성된 검증 데이터셋 생성
PYTHONPATH=src poetry run python scripts/create_validation_set.py --sample_size 50

data/validation.jsonl 파일이 생성됩니다.

### **3.2. 검색 성능 검증 및 하이퍼파라미터 튜닝**

validate_retrieval.py 스크립트는 validation.jsonl 파일을 사용하여 현재 모델의 검색 성능을 **MAP(Mean Average Precision)** 점수로 측정합니다. 이 스크립트를 사용하여 다양한 하이퍼파라미터를 테스트하고 최적의 조합을 찾을 수 있습니다.

사용 예시: alpha 값 튜닝
alpha는 BM25(키워드 검색)와 시맨틱 검색(의미 기반 검색)의 가중치를 조절하는 중요한 파라미터입니다.
# 기준 성능 측정 (alpha 기본값 사용)
PYTHONPATH=src poetry run python scripts/validate_retrieval.py

# 시맨틱 검색에 더 큰 가중치를 부여하여 테스트
PYTHONPATH=src poetry run python scripts/validate_retrieval.py --alpha 0.3

# 키워드 검색에 더 큰 가중치를 부여하여 테스트
PYTHONPATH=src poetry run python scripts/validate_retrieval.py --alpha 0.7

각각의 MAP 점수를 비교하여 가장 높은 점수를 내는 alpha 값을 찾은 후, 그 값을 최종 모델에 적용할 수 있습니다.

## **4. 고급 실험 가이드**

프로젝트의 성능을 극대화하기 위한 추가적인 실험 방법입니다.

### **4.1. 임베딩 모델 변경**

hybrid_retrieve의 성능은 임베딩 모델에 크게 좌우됩니다. .env 파일에서 EMBEDDING_MODEL 값을 변경하여 다른 모델을 테스트할 수 있습니다.

# .env 파일 예시
# 다른 한국어 특화 모델로 변경하여 테스트
EMBEDDING_MODEL=Upstage/solar-1-mini-embedding-ko

**중요:** 임베딩 모델을 변경한 후에는 **반드시 데이터를 재색인**해야 합니다. (scripts/reindex.py 실행)

### **4.2. 프롬프트 수정**

LLM의 동작(도구 호출, 답변 생성)은 프롬프트에 매우 민감합니다. prompts/ 디렉토리의 scientific_qa_v1.jinja2 또는 conversational_v1.jinja2 파일을 수정하여 LLM의 응답을 개선할 수 있습니다.

* **실험:** 도구 호출 설명을 더 명확하게 하거나, 답변 스타일을 변경하는 등의 수정을 시도해볼 수 있습니다.
* **검증:** 프롬프트를 수정한 후에는 validate_retrieval.py를 실행하여 MAP 점수가 향상되었는지 반드시 확인해야 합니다.