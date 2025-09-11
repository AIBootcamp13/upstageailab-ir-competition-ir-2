# 🚀 프로젝트 사용 가이드 (Hydra 버전)

이 문서는 Hydra 구성 프레임워크를 사용하여 RAG(Retrieval-Augmented Generation) 프로젝트의 주요 기능을 사용하고, 성능을 튜닝하며, 제출물을 생성하는 전체 워크플로우를 안내합니다.

## 목차
- 초기 설정
- 핵심 워크플로우 (Hydra 사용)
- 모델 성능 튜닝 및 검증
- 고급 실험 가이드

---

## 1. 초기 설정
프로젝트를 실행하기 위해 필요한 초기 설정 단계입니다.

### 1.1 의존성 설치
Poetry를 사용하여 필요한 모든 라이브러리를 설치합니다. `hydra-core`가 포함되어 있는지 확인하세요.

```bash
poetry install
```

### 1.2 환경 변수 설정
프로젝트 루트의 `.env.example`을 복사하여 `.env`를 생성한 뒤 `OPENAI_API_KEY` 등을 설정합니다. Hydra는 이 파일을 자동으로 인식합니다.

```bash
cp .env.example .env
# 편집기 예시:
# nano .env
```

### 1.3 로컬 인프라 실행
Elasticsearch와 Redis를 로컬에서 실행합니다.

```bash
./scripts/run-local.sh start
```

### 1.4 데이터 색인 (Indexing)
대회에서 제공된 `documents.jsonl` 파일을 Elasticsearch에 색인합니다.

```bash
PYTHONPATH=src poetry run python scripts/reindex.py data/documents.jsonl --index test
```

---

## 2. 핵심 워크플로우 (Hydra 사용)
Hydra는 `conf/` 디렉토리의 설정 파일로 프로젝트 동작을 제어하며, 커맨드라인에서 설정을 쉽게 덮어쓸 수 있습니다.

### 2.1 대회 제출 파일 생성
`evaluate.py` 스크립트는 `conf/config.yaml`에 정의된 기본 설정을 사용해 제출 파일을 생성합니다.

```bash
# 기본 설정으로 실행
PYTHONPATH=src poetry run python scripts/evaluate.py
```

실행 후 `outputs/submission.jsonl` 파일이 생성됩니다.

커맨드라인에서 설정 변경 예시:

```bash
# 제출 파일에 포함할 문서 수를 5개로 변경
PYTHONPATH=src poetry run python scripts/evaluate.py params.submission.topk=5

# 사용할 평가 파일을 검증 데이터셋으로 변경
PYTHONPATH=src poetry run python scripts/evaluate.py paths.evaluation=data/validation.jsonl
```

---

## 3. 모델 성능 튜닝 및 검증
`validate_retrieval.py` 스크립트로 하이퍼파라미터 튜닝 및 MAP 점수 기반 성능 측정을 수행합니다.

### 3.1 검증 데이터셋 생성 (필요시)

```bash
PYTHONPATH=src poetry run python scripts/create_validation_set.py --sample_size 50
```

이후 `data/validation.jsonl` 파일이 생성됩니다.

### 3.2 검색 성능 검증 및 하이퍼파라미터 튜닝
기본 파라미터로 MAP 점수를 확인합니다.

```bash
PYTHONPATH=src poetry run python scripts/validate_retrieval.py
```

alpha 값( BM25와 시맨틱 가중치 조절) 튜닝 예시:

```bash
# alpha=0.2 (시맨틱 가중치 증가)
PYTHONPATH=src poetry run python scripts/validate_retrieval.py params.retrieval.alpha=0.2

# alpha=0.5 (균형)
PYTHONPATH=src poetry run python scripts/validate_retrieval.py params.retrieval.alpha=0.5

# alpha=0.8 (BM25 가중치 증가)
PYTHONPATH=src poetry run python scripts/validate_retrieval.py params.retrieval.alpha=0.8
```

가장 높은 MAP 점수를 주는 값을 찾아 `conf/params/base.yaml`의 기본값으로 설정하세요.

---

## 4. 고급 실험 가이드

### 4.1 임베딩 모델 변경
`.env`에서 `EMBEDDING_MODEL` 값을 변경하여 다른 임베딩을 테스트할 수 있습니다. 변경 후에는 데이터 재색인이 필요합니다.

```bash
# .env 예시
EMBEDDING_MODEL=Upstage/solar-1-mini-embedding-ko
```

임베딩 모델 변경 뒤 반드시 재색인 수행:

```bash
PYTHONPATH=src poetry run python scripts/reindex.py data/documents.jsonl --index test
```

### 4.2 프롬프트 수정
`prompts/` 디렉토리의 파일을 수정하여 LLM 동작을 개선할 수 있습니다. 프롬프트 변경 후 `validate_retrieval.py`로 MAP 점수 향상 여부를 확인하세요.

---

끝.