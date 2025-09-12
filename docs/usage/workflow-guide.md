# **🚀 프로젝트 워크플로우 가이드**

이 가이드는 RAG 프로젝트의 초기 설정부터 고급 실험 실행, 제출 파일 생성까지의 전체 워크플로우를 안내합니다. 이 프로젝트는 [Hydra](https://hydra.cc/)를 사용하여 모든 설정을 관리합니다.

## **1. 초기 설정**

먼저 로컬 개발 환경을 준비합니다.

### **1.1. 의존성 설치**

이 명령어는 `pyproject.toml`에 정의된 모든 필수 라이브러리를 설치합니다.

```bash
poetry install
```

### **1.2. 환경 변수 설정**

예제 `.env` 파일을 복사합니다. `OPENAI_API_KEY`를 추가하기 위해 반드시 이 파일을 수정해야 합니다.

```bash
cp .env.example .env
# nano 또는 vim과 같은 텍스트 편집기를 사용하여 .env 파일을 수정하세요
nano .env
```

### **1.3. 로컬 인프라 실행**

이 스크립트는 로컬 환경에 Elasticsearch와 Redis를 다운로드하고 실행합니다.

```bash
./scripts/run-local.sh start
```

`./scripts/run-local.sh status` 명령어로 서비스 상태를 확인할 수 있습니다.

-----

## **2. 데이터 색인**

실험을 실행하기 전에, 제공된 `documents.jsonl` 파일을 Elasticsearch에 색인해야 합니다. 이 명령어는 `conf/config.yaml`에서 설정을 읽어옵니다.

```bash
PYTHONPATH=src poetry run python scripts/reindex.py
```

-----

## **3. 핵심 워크플로우: 실험 실행**

이 프로젝트는 실험 중심으로 설계되었습니다. 핵심 스크립트인 `validate_retrieval.py`는 로컬 검증 데이터셋으로 검색 파이프라인을 실행하고 결과를 Weights & Biases에 기록합니다.

### **3.1. 기본 검증 실행**

`conf/` 디렉토리에 정의된 기본 파라미터로 검증을 실행하려면 다음 명령어를 사용하세요.

```bash
PYTHONPATH=src poetry run python scripts/validate_retrieval.py
```

### **3.2. 파라미터 변경하기**

Hydra의 가장 큰 장점은 커맨드라인에서 설정을 쉽게 변경할 수 있다는 점입니다.

**예시: `alpha` 값을 변경하고 데이터셋 일부만 사용하기**

```bash
# 이번 실행에만 model.alpha 값을 0.7로 설정
# limit=50을 추가하여 빠른 테스트 실행
PYTHONPATH=src poetry run python scripts/validate_retrieval.py model.alpha=0.7 limit=50
```

### **3.3. 고급 실험: 프롬프트 튜닝 (Multi-Run)**

여러 실험을 병렬로 실행할 수 있습니다. 예를 들어, `conf/experiment/prompt_tuning.yaml`에 정의된 여러 도구 호출 프롬프트를 한 번에 테스트하려면 다음 명령어를 사용하세요.

```bash
PYTHONPATH=src poetry run python scripts/validate_retrieval.py --multirun experiment=prompt_tuning limit=50
```

이 명령어는 각기 다른 프롬프트를 사용하는 3개의 개별 실행을 시작하고, 결과를 WandB에 기록하여 쉽게 비교할 수 있도록 합니다.

-----

## **4. 제출 파일 생성**

실험을 통해 최적의 파라미터를 찾았다면, `evaluate.py` 스크립트를 실행하여 공식 제출 파일을 생성합니다.

```bash
# 검증 단계에서 찾은 최적의 파라미터를 사용
PYTHONPATH=src poetry run python scripts/evaluate.py model.alpha=0.7
```

이 명령어는 `outputs/` 디렉토리에 `submission.jsonl` 파일을 생성하고, 해당 파일을 WandB 아티팩트로도 기록합니다.

### **4.1. 제출 파일 후처리**

실험 결과를 분석하거나 LLM으로 추가 처리하기 위해 제출 파일을 후처리할 수 있는 유틸리티 스크립트들이 제공됩니다.

#### **제출 파일 내용 트리밍**

`trim_submission.py` 스크립트는 제출 파일의 내용을 지정된 길이로 트리밍하여 LLM 분석에 적합하게 만듭니다.

```bash
# 기본 최대 길이(500자)로 트리밍
python trim_submission.py outputs/submission.csv outputs/submission_trimmed.csv

# 사용자 지정 최대 길이(300자)로 트리밍
python trim_submission.py outputs/submission.csv outputs/submission_trimmed.csv 300
```

**파라미터 설명:**
- `input_file`: 트리밍할 원본 제출 CSV 파일 경로
- `output_file`: 트리밍된 결과를 저장할 CSV 파일 경로
- `max_length` (선택): 최대 내용 길이 (기본값: 500자)

#### **제출 파일을 평가 로그로 변환**

`transform_submission.py` 스크립트는 제출 파일을 구조화된 JSON 형식의 평가 로그로 변환합니다.

```bash
# 제출 파일을 평가 로그로 변환
python transform_submission.py data/eval.jsonl outputs/submission.csv outputs/evaluation_logs.jsonl
```

**파라미터 설명:**
- `eval_file`: 원본 평가 쿼리가 포함된 JSONL 파일 경로
- `submission_file`: 변환할 제출 CSV 파일 경로
- `output_file`: 변환된 평가 로그를 저장할 JSONL 파일 경로

이 스크립트는 다음을 수행합니다:
- 평가 파일에서 원본 쿼리를 추출
- 제출 파일의 검색 결과를 구조화된 형식으로 변환
- 평가 분석에 적합한 JSONL 형식으로 저장

-----

## **6. 고급 사용법**

### **6.1. 새로운 검증 데이터셋 생성**

`create_validation_set.py` 스크립트 또한 Hydra로 설정을 관리할 수 있습니다.

```bash
# 100개의 샘플로 새로운 검증 데이터셋 생성
PYTHONPATH=src poetry run python scripts/create_validation_set.py create_validation_set.sample_size=100
```

### **6.3. 고성능 분석: 병렬 처리**

프로젝트의 분석 프레임워크는 대규모 데이터셋 처리 시 자동으로 병렬 처리를 지원합니다. 자세한 내용은 [병렬 처리 가이드](parallel-processing-guide.md)를 참조하세요.

#### **기본 사용법**

```bash
# 기본 설정으로 분석 실행 (자동 병렬 처리)
PYTHONPATH=src poetry run python scripts/validate_retrieval.py
```

#### **병렬 처리 설정**

```yaml
# conf/config.yaml에 추가
analysis:
  max_workers: 8          # 최대 워커 수
  enable_parallel: true   # 병렬 처리 활성화
```

#### **커맨드라인에서 설정 변경**

```bash
# 병렬 처리 비활성화
PYTHONPATH=src poetry run python scripts/validate_retrieval.py analysis.enable_parallel=false

# 최대 워커 수 지정
PYTHONPATH=src poetry run python scripts/validate_retrieval.py analysis.max_workers=4
```

#### **성능 모니터링**

```bash
# 처리 시간 모니터링
time PYTHONPATH=src poetry run python scripts/validate_retrieval.py limit=100

# 메모리 사용량 확인
PYTHONPATH=src poetry run python scripts/validate_retrieval.py | grep "🔄"
```

## Multirun 3 run example
```bash
poetry run python scripts/validate_retrieval.py --multirun experiment=prompt_tuning limit=50 prompts.tool_description='prompts/tool_desc_baseline.txt,prompts/tool_desc_balanced_v1.txt,prompts/tool_desc_recall_v1.txt'
```


