```markdown
# 구성 전환기 문서

## 개요

`switch_config.py` 스크립트는 다양한 RAG 시스템 구성 간 전환을 편리하게 제공합니다. 여러 임베딩 프로바이더와 다양한 언어 및 사용 사례에 최적화된 구성을 지원합니다.

## 지원되는 구성

### 한국어 구성 (768D)
- **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **차원**: 768
- **인덱스**: `documents_ko_with_embeddings_new`
- **데이터**: 한국어 문서 (`data/documents_ko.jsonl`)
- **번역**: 비활성화
- **사용 사례**: 한국어 전용 콘텐츠, 한국어 쿼리에 최적 성능

### 영어 구성 (384D)
- **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (384D 투영)
- **차원**: 384
- **인덱스**: `documents_en_with_embeddings_new`
- **데이터**: 이중언어 문서 (`data/documents_bilingual.jsonl`)
- **번역**: 활성화
- **사용 사례**: 번역 지원이 있는 영어 콘텐츠

### 이중언어 구성 (768D)
- **임베딩 모델**: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- **차원**: 768
- **인덱스**: `documents_bilingual_with_embeddings_new`
- **데이터**: 이중언어 문서 (`data/documents_bilingual.jsonl`)
- **번역**: 비활성화
- **사용 사례**: 번역 없는 한국어/영어 혼합 콘텐츠

### Solar API 구성 (4096D)
- **임베딩 프로바이더**: Upstage Solar API
- **모델**: `solar-embedding-1-large`
- **차원**: 4096
- **인덱스**: `documents_solar_with_embeddings_new`
- **데이터**: 이중언어 문서 (`data/documents_bilingual.jsonl`)
- **번역**: 비활성화
- **사용 사례**: 두 언어 모두에 대한 고품질 임베딩, API 키 필요

## 사용법

### 명령줄 인터페이스

```bash
# 한국어 구성으로 전환
PYTHONPATH=src poetry run python switch_config.py korean

# 영어 구성으로 전환
PYTHONPATH=src poetry run python switch_config.py english

# 이중언어 구성으로 전환
PYTHONPATH=src poetry run python switch_config.py bilingual

# Solar API 구성으로 전환
PYTHONPATH=src poetry run python switch_config.py solar

# 현재 구성 표시
PYTHONPATH=src poetry run python switch_config.py show
```

### 구성 세부사항

각 구성은 `conf/settings.yaml`에서 다음 설정을 업데이트합니다:

- `EMBEDDING_PROVIDER`: 프로바이더 유형 (`huggingface` 또는 `solar`)
- `EMBEDDING_MODEL`: 모델 이름 또는 식별자
- `EMBEDDING_DIMENSION`: 벡터 차원
- `INDEX_NAME`: Elasticsearch 인덱스 이름
- `ALPHA`: BM25/Dense 검색 균형 (0.4)
- `BM25_K`: BM25 결과 수 (200)
- `RERANK_K`: Dense 재순위 결과 수 (10)
- 번역 설정
- 데이터 구성 참조

## 전제 조건

### HuggingFace 구성용 (한국어/영어/이중언어)

추가 설정 불필요 - 로컬 모델 사용.

### Solar API 구성용

1. **API 키 설정**: `UPSTAGE_API_KEY` 환경 변수 설정:
   ```bash
   # .env 파일에서
   UPSTAGE_API_KEY=your_upstage_api_key_here

   # 또는 셸에서 내보내기
   export UPSTAGE_API_KEY=your_key
   ```

2. **인덱스 생성**: 올바른 차원으로 Solar 인덱스 생성:
   ```bash
   PYTHONPATH=src poetry run python scripts/maintenance/create_index.py \
     --index documents_solar_with_embeddings_new \
     --dimension 4096
   ```

3. **데이터 인덱싱**: Solar 임베딩으로 문서 인덱싱:
   ```bash
   PYTHONPATH=src poetry run python scripts/maintenance/reindex.py \
     data/documents_bilingual.jsonl \
     --index documents_solar_with_embeddings_new
   ```

## 구성 검증

### 현재 구성 확인

```bash
PYTHONPATH=src poetry run python switch_config.py show
```

출력 포함 사항:
- 임베딩 프로바이더 및 모델
- 벡터 차원
- 인덱스 이름
- 번역 상태
- 데이터 구성
- API 키 상태 (Solar용)

### API 키 확인 (Solar 전용)

`show` 명령은 Solar API 키가 올바르게 구성되었는지 표시합니다:
- `Solar API Key Set: ✅ Yes` - 사용 준비 완료
- `Solar API Key Set: ❌ No` - 먼저 UPSTAGE_API_KEY 설정

## 다른 스크립트와의 통합

### 자동 구성 감지

대부분의 스크립트는 `conf/settings.yaml`의 현재 구성을 자동으로 사용합니다. 추가 매개변수 불필요.

### 수동 프로바이더 오버라이드

구성 변경 없이 다른 프로바이더 테스트용:

```python
from ir_core.embeddings import get_embedding_provider

# 특정 프로바이더 강제 사용
hf_provider = get_embedding_provider('huggingface')
solar_provider = get_embedding_provider('solar')
```

## 문제 해결

### 일반적인 문제

1. **"UPSTAGE_API_KEY environment variable not set"**
   - 해결책: `.env` 파일에 `UPSTAGE_API_KEY=your_key` 추가

2. **"Index does not exist"**
   - 해결책: 먼저 인덱스 생성, 그 다음 데이터 재인덱싱
   - 위의 전제 조건 섹션 참조

3. **"Configuration not applied"**
   - 해결책: `conf/settings.yaml`이 업데이트되었는지 확인
   - 실행 중인 서비스 재시작

4. **"Module not found" 오류**
   - 해결책: `PYTHONPATH=src`가 설정되었는지 확인

### 구성 복구

알려진 정상 구성으로 재설정:

```bash
# 한국어 구성으로 재설정
PYTHONPATH=src poetry run python switch_config.py korean

# 설정 확인
PYTHONPATH=src poetry run python switch_config.py show
```

## 성능 고려사항

### 임베딩 차원 영향

- **384D (영어)**: 가장 빠름, 낮은 메모리 사용량
- **768D (한국어/이중언어)**: 성능과 품질의 균형
- **4096D (Solar)**: 최고 품질, 느림, 높은 메모리 사용량

### 프로바이더 성능

- **HuggingFace**: 로컬 추론, GPU 가속 가능
- **Solar API**: 클라우드 추론, 일관된 품질, 인터넷 필요

## 고급 사용법

### 사용자 정의 구성

개발/테스트용으로 다음을 통해 사용자 정의 구성을 생성할 수 있습니다:

1. `switch_to_korean()` 패턴을 따라 새 함수 추가
2. 새 명령을 포함하도록 `main()` 함수 업데이트
3. `switch_config.py show`로 테스트

### 환경별 설정

스크립트는 환경 변수와 `.env` 파일 설정을 준수합니다. 프로덕션 배포의 경우 다음을 확인하세요:

- API 키가 적절히 보안됨
- 네트워크 타임아웃이 구성됨
- 대체 구성이 사용 가능함

## 예제

### 완전한 Solar 설정 워크플로

```bash
# 1. API 키 설정
echo "UPSTAGE_API_KEY=your_key_here" >> .env

# 2. 구성 전환
PYTHONPATH=src poetry run python switch_config.py solar

# 3. 설정 확인
PYTHONPATH=src poetry run python switch_config.py show

# 4. 인덱스 생성
PYTHONPATH=src poetry run python scripts/maintenance/create_index.py \
  --index documents_solar_with_embeddings_new \
  --dimension 4096

# 5. 데이터 인덱싱
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py \
  data/documents_bilingual.jsonl \
  --index documents_solar_with_embeddings_new

# 6. 임베딩 테스트
PYTHONPATH=src poetry run python -c "
from ir_core.embeddings import encode_query
emb = encode_query('테스트 쿼리')
print(f'임베딩 형태: {emb.shape}')  # (4096,)이어야 함
"
```

### 실험용 구성 전환

```bash
# 다양한 구성 테스트
for config in ['korean', 'english', 'bilingual', 'solar']:
    echo "$config 구성 테스트중..."
    PYTHONPATH=src poetry run python switch_config.py $config
    PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py debug=true debug_limit=5
done
```
