# Solar API 통합 가이드

## 개요

RAG 시스템은 이제 고품질 4096차원 임베딩 생성을 위한 Upstage Solar API를 지원합니다. Solar는 한국어와 영어 콘텐츠 모두에 대해 최첨단 임베딩 성능을 제공합니다.

## 아키텍처

### 프로바이더 패턴

시스템은 임베딩 생성을 위해 프로바이더 패턴을 사용합니다:

```
BaseEmbeddingProvider (추상)
├── HuggingFaceEmbeddingProvider (로컬 모델)
└── SolarEmbeddingProvider (Upstage API)
```

### 자동 프로바이더 선택

시스템은 자동으로 적절한 프로바이더를 선택합니다:

- **Solar**: `UPSTAGE_API_KEY` 환경 변수가 설정된 경우
- **HuggingFace**: API 키가 없을 때의 기본 대체

## 설정

### 1. API 키 구성

환경에서 Upstage API 키를 설정하세요:

```bash
# 옵션 1: .env 파일 (권장)
echo "UPSTAGE_API_KEY=your_upstage_api_key_here" >> .env

# 옵션 2: 환경 변수
export UPSTAGE_API_KEY=your_key

# 옵션 3: 시스템 전체
echo 'export UPSTAGE_API_KEY=your_key' >> ~/.bashrc
```

### 2. Solar 구성으로 전환

```bash
PYTHONPATH=src poetry run python switch_config.py solar
```

### 3. 구성 확인

```bash
PYTHONPATH=src poetry run python switch_config.py show
```

다음을 확인하세요:
- `Embedding Provider: solar`
- `Embedding Dimension: 4096`
- `Solar API Key Set: ✅ Yes`

## 사용법

### 기본 임베딩 생성

```python
from ir_core.embeddings import encode_texts, encode_query

# 여러 텍스트 인코딩
texts = [
    "양자역학이란 무엇인가?",
    "태양광 패널은 어떻게 작동하는가?",
    "머신러닝을 설명하세요"
]
embeddings = encode_texts(texts)
print(f"Shape: {embeddings.shape}")  # (3, 4096)

# 단일 쿼리 인코딩
query_emb = encode_query("인공지능이란 무엇인가?")
print(f"Shape: {query_emb.shape}")  # (4096,)
```

### 직접 프로바이더 사용

```python
from ir_core.embeddings import get_embedding_provider

# Solar 프로바이더 직접 가져오기
provider = get_embedding_provider('solar')

# 차원 확인
print(f"Dimensions: {provider.dimension}")  # 4096

# 사용자 정의 매개변수로 인코딩
embeddings = provider.encode_texts(
    texts=["안녕하세요"],
    model="solar-embedding-1-large"  # 선택적 모델 오버라이드
)
```

## API 세부사항

### 엔드포인트
- **URL**: `https://api.upstage.ai/v1/solar/embeddings`
- **메서드**: POST
- **인증**: Bearer 토큰 (`UPSTAGE_API_KEY`)

### 요청 형식

```json
{
  "model": "solar-embedding-1-large",
  "input": ["text1", "text2", "..."]
}
```

### 응답 형식

```json
{
  "data": [
    {
      "embedding": [0.123, 0.456, ...],
      "index": 0
    },
    {
      "embedding": [0.789, 0.012, ...],
      "index": 1
    }
  ]
}
```

### 오류 처리

프로바이더는 일반적인 API 오류를 처리합니다:

- **401 Unauthorized**: 잘못된 API 키
- **429 Too Many Requests**: 속도 제한 초과
- **500 Internal Error**: 서버 오류
- **Timeout**: 네트워크 문제

오류는 설명적인 메시지와 함께 `RuntimeError`로 발생합니다.

## 성능 특성

### 차원: 4096
- **품질**: 최고 임베딩 품질
- **메모리**: 저장/인덱싱을 위한 높은 메모리 사용량
- **속도**: 낮은 차원 임베딩보다 느림
- **정확도**: 최고의 의미 이해

### 배치 처리
- **최적 배치 크기**: 요청당 10-50개 텍스트
- **속도 제한**: API 속도 제한 준수
- **병렬화**: 내장 배치 지원

### 네트워크 고려사항
- **지연시간**: 요청당 ~200-500ms
- **신뢰성**: 클라우드 기반, 높은 가동시간
- **비용**: 사용량 기반 가격

## 통합 예제

### RAG 파이프라인 통합

```python
from ir_core.embeddings import encode_query
from ir_core.retrieval import retrieve_documents

# Solar로 쿼리 인코딩
query = "재생에너지의 장점은 무엇인가?"
query_embedding = encode_query(query)

# 문서 검색
hits = retrieve_documents(
    query_embedding=query_embedding,
    index_name="documents_solar_with_embeddings_new",
    top_k=10
)

print(f"Retrieved {len(hits)} documents")
```

### 평가 스크립트 사용

```bash
# Solar 임베딩으로 평가 실행
PYTHONPATH=src poetry run python switch_config.py solar
PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py \
  --config-dir conf \
  pipeline=qwen-full \
  model.alpha=0.4 \
  limit=10
```

### 테스트 통합

```python
# Solar API 연결 테스트
from ir_core.embeddings import get_embedding_provider

try:
    provider = get_embedding_provider('solar')
    test_emb = provider.encode_query("테스트")
    print("✅ Solar API가 올바르게 작동합니다")
    print(f"임베딩 차원: {len(test_emb)}")
except Exception as e:
    print(f"❌ Solar API 오류: {e}")
```

## 구성 옵션

### 환경 변수

```bash
# 필수
UPSTAGE_API_KEY=your_key_here

# 선택적 오버라이드
SOLAR_BASE_URL=https://api.upstage.ai/v1/solar  # 사용자 정의 엔드포인트
SOLAR_MODEL=solar-embedding-1-large            # 모델 이름
```

### 설정 구성

Solar 프로바이더는 `conf/settings.yaml`의 다음 설정을 준수합니다:

```yaml
UPSTAGE_API_KEY: ${env:UPSTAGE_API_KEY}  # Upstage API 키
SOLAR_BASE_URL: https://api.upstage.ai/v1/solar
SOLAR_MODEL: solar-embedding-1-large
EMBEDDING_DIMENSION: 4096  # Solar용 고정값
```

## 문제 해결

### 일반적인 문제

1. **"UPSTAGE_API_KEY environment variable not set"**
   ```bash
   # 설정 여부 확인
   echo $UPSTAGE_API_KEY

   # 현재 세션에서 설정
   export UPSTAGE_API_KEY=your_key

   # 또는 .env에 추가
   echo "UPSTAGE_API_KEY=your_key" >> .env
   ```

2. **"Solar API request failed"**
   - API 키 유효성 확인
   - 인터넷 연결 확인
   - API 할당량/제한 확인

3. **"Invalid Solar API response format"**
   - API가 변경되었을 수 있음 - Upstage 문서 확인
   - 모델 이름이 올바른지 확인

4. **타임아웃 오류**
   - 프로바이더에서 타임아웃 증가 (현재 30초)
   - 네트워크 안정성 확인

### 디버깅

디버그 로깅 활성화:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 디버그 출력으로 테스트
from ir_core.embeddings import encode_query
emb = encode_query("테스트 쿼리")
```

### 대체 구성

Solar API가 실패하면 시스템은 HuggingFace로 대체할 수 있습니다:

```python
from ir_core.embeddings import get_embedding_provider

try:
    # 먼저 Solar 시도
    provider = get_embedding_provider('solar')
    emb = provider.encode_query("테스트")
except Exception as e:
    print(f"Solar 실패: {e}")
    # HuggingFace로 대체
    provider = get_embedding_provider('huggingface')
    emb = provider.encode_query("테스트")
```

## 모범 사례

### 비용 최적화

1. **배치 요청**: 단일 API 호출로 여러 텍스트 전송
2. **결과 캐싱**: 반복 쿼리에 Redis 캐싱 사용
3. **선택적 사용**: 중요한 쿼리에는 Solar, 대량 처리에는 HuggingFace 사용

### 성능 최적화

1. **비동기 처리**: 높은 처리량을 위한 비동기 구현 고려
2. **연결 풀링**: HTTP 연결 재사용
3. **오류 재시도**: 일시적
오류에 대한 지수 백오프 구현

### 모니터링

1. **API 사용량**: 요청 수와 비용 모니터링
2. **지연시간**: 응답 시간 추적
3. **오류율**: API 신뢰성 모니터링

## 마이그레이션 가이드

### HuggingFace에서 Solar로

1. **현재 설정 백업**:
   ```bash
   cp conf/settings.yaml conf/settings.yaml.backup
   ```

2. **API 키 설정**:
   ```bash
   echo "UPSTAGE_API_KEY=your_key" >> .env
   ```

3. **구성 전환**:
   ```bash
   PYTHONPATH=src poetry run python switch_config.py solar
   ```

4. **새 인덱스 생성**:
   ```bash
   PYTHONPATH=src poetry run python scripts/maintenance/create_index.py \
     --index documents_solar_with_embeddings_new \
     --dimension 4096
   ```

5. **데이터 재인덱싱**:
   ```bash
   PYTHONPATH=src poetry run python scripts/maintenance/reindex.py \
     data/documents_bilingual.jsonl \
     --index documents_solar_with_embeddings_new
   ```

6. **통합 테스트**:
   ```bash
   PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py \
     debug=true debug_limit=3
   ```

### 롤백 계획

HuggingFace로 롤백하려면:

```bash
# 이전 구성으로 다시 전환
PYTHONPATH=src poetry run python switch_config.py korean  # 또는 english/bilingual

# 필요시 백업 복원
cp conf/settings.yaml.backup conf/settings.yaml
```

## API 참조

### SolarEmbeddingProvider

```python
class SolarEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str = None, base_url: str = None)
    def encode_texts(self, texts: List[str], **kwargs) -> np.ndarray
    def encode_query(self, query: str, **kwargs) -> np.ndarray
    @property
    def dimension(self) -> int  # 4096 반환
```

### 매개변수

- **api_key**: Upstage API 키 (UPSTAGE_API_KEY 환경 변수에서)
- **base_url**: API 기본 URL (기본값: Upstage 엔드포인트)
- **model**: 모델 이름 (기본값: solar-embedding-1-large)
- **timeout**: 요청 타임아웃(초) (기본값: 30)

## 지원

Solar API 통합 문제의 경우:

1. [Upstage API 문서](https://developers.upstage.ai/) 확인
2. API 키와 할당량 확인
3. 최소 예제 코드로 테스트
4. 네트워크 연결 확인
5. 구체적인 지침을 위해 오류 메시지 검토
