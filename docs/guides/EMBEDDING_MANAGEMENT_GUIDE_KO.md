# 임베딩 관리 가이드 (Embedding Management Guide)

## 개요

이 문서는 RAG (Retrieval-Augmented Generation) 시스템에서 발생할 수 있는 임베딩 관련 문제를 방지하고 해결하기 위한 포괄적인 가이드를 제공합니다. 특히 HyDE (Hypothetical Document Embeddings) 검색 실패와 관련된 문제를 중점적으로 다룹니다.

## 목차

1. [임베딩 문제의 종류](#임베딩-문제의-종류)
2. [HyDE 검색 실패의 원인](#hyde-검색-실패의-원인)
3. [임베딩 재생성 시점](#임베딩-재생성-시점)
4. [임베딩 관리 워크플로우](#임베딩-관리-워크플로우)
5. [설정 관리](#설정-관리)
6. [검증 및 테스트](#검증-및-테스트)
7. [베스트 프랙티스](#베스트-프랙티스)
8. [문제 해결 가이드](#문제-해결-가이드)

## 임베딩 문제의 종류

### 1. 차원 불일치 (Dimension Mismatch)
- **증상**: `BadRequestError(400, 'search_phase_execution_exception', 'runtime error')`
- **원인**: 쿼리 임베딩(384D)과 인덱스 임베딩(768D)의 차원이 일치하지 않음
- **영향**: 모든 밀집 검색(dense retrieval) 실패

### 2. 누락된 임베딩 (Missing Embeddings)
- **증상**: `IllegalArgumentException: Dense vector value missing for a field`
- **원인**: 문서가 임베딩 없이 인덱싱됨
- **영향**: cosineSimilarity 스크립트 실행 실패

### 3. 모델 불일치 (Model Mismatch)
- **증상**: 검색 결과의 질이 낮거나 무관한 결과 반환
- **원인**: 다른 임베딩 모델로 생성된 벡터 사용
- **영향**: 검색 정확도 저하

## HyDE 검색 실패의 원인

### 주요 실패 시나리오

#### 시나리오 1: 잘못된 인덱싱
```bash
# ❌ 잘못된 순서
1. 문서를 임베딩 없이 인덱싱
2. HyDE 검색 시도
3. cosineSimilarity 함수 실패
```

#### 시나리오 2: 설정 변경 후 재인덱싱 누락
```bash
# ❌ 설정 변경 후 임베딩 재생성 누락
1. switch_config.py korean 실행
2. 임베딩 재생성(recompute.py) 누락
3. 차원 불일치로 검색 실패
```

#### 시나리오 3: 모델 변경 후 재인덱싱 누락
```bash
# ❌ 모델 변경 후 재인덱싱 누락
1. 설정에서 EMBEDDING_MODEL 변경
2. recompute.py 실행 누락
3. 모델 불일치로 검색 실패
```

## 임베딩 재생성 시점

### 필수 재생성 케이스

| 시나리오 | 재생성 필요 | 이유 | 명령어 |
|----------|-------------|------|--------|
| **모델 변경** | ✅ 항상 | 다른 벡터 공간 사용 | `recompute.py --model X` |
| **차원 변경** | ✅ 항상 | 벡터 길이 불일치 | `recompute.py --model X` |
| **새 문서 추가** | ✅ 항상 | 새 문서에 임베딩 필요 | `recompute.py` |
| **인덱스 손상** | ✅ 항상 | 데이터 무결성 문제 | 인덱스 삭제 후 `recompute.py` |
| **설정 전환** | ✅ 항상 | 다른 모델/차원 사용 | `switch_config.py` + `recompute.py` |

### 선택적 재생성 케이스

| 시나리오 | 재생성 필요 | 이유 |
|----------|-------------|------|
| **코드 변경** | ❌ 드물게 | 임베딩 로직 변경시에만 |
| **설정 변경** | ❌ 드물게 | 임베딩 생성에 영향 주는 경우만 |
| **성능 튜닝** | ❌ 선택적 | 개선 목적일 경우 |

## 임베딩 관리 워크플로우

### 1. 설정 전환 워크플로우

```bash
# 한국어 설정으로 전환
PYTHONPATH=src poetry run python switch_config.py korean

# 임베딩 재생성 (필수!)
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type korean \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS
```

### 2. 새 문서 추가 워크플로우

```bash
# 1. 새 문서를 데이터 파일에 추가
echo '{"docid": "new_doc", "content": "새 문서 내용", "src": "new_source"}' >> data/documents_ko.jsonl

# 2. 전체 데이터셋에 대한 임베딩 재생성
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type korean \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS
```

### 3. Solar API 전환 워크플로우

```bash
# Solar API 설정으로 전환
PYTHONPATH=src poetry run python switch_config.py solar

# 4096D 임베딩 재생성
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type bilingual \
  --model solar-embedding-1-large
```

## 설정 관리

### 지원되는 구성 조합

| 구성 | 임베딩 모델 | 차원 | 인덱스 이름 | 데이터 파일 |
|------|-------------|------|-------------|------------|
| **한국어** | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 768D | `documents_ko_with_embeddings_new` | `data/documents_ko.jsonl` |
| **영어** | `sentence-transformers/all-MiniLM-L6-v2` | 384D | `documents_en_with_embeddings_new` | `data/documents_bilingual.jsonl` |
| **이중언어** | `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | 768D | `documents_bilingual_with_embeddings_new` | `data/documents_bilingual.jsonl` |
| **Solar** | `solar-embedding-1-large` | 4096D | `documents_solar_with_embeddings_new` | `data/documents_bilingual.jsonl` |

### 설정 검증

```bash
# 현재 설정 확인
PYTHONPATH=src poetry run python switch_config.py show

# 출력 예시:
# 📋 Current Configuration:
#    Embedding Provider: huggingface
#    Embedding Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
#    Embedding Dimension: 768
#    Index Name: documents_ko_with_embeddings_new
```

## 검증 및 테스트

### 임베딩 검증 스크립트

```bash
# 임베딩 생성 및 검색 검증
PYTHONPATH=src poetry run python -c "
from ir_core.embeddings.core import encode_query
from ir_core.retrieval.core import dense_retrieve
import numpy as np

# 임베딩 생성 테스트
query_emb = encode_query('양자역학이란 무엇인가?')
print(f'임베딩 형태: {query_emb.shape}')

# 검색 테스트
results = dense_retrieve(query_emb, size=5)
print(f'검색 성공: {len(results)}개 결과 반환')

# 결과 검증
if results:
    first_result = results[0]
    print(f'첫 번째 결과 점수: {first_result.get(\"_score\", \"N/A\")}')
    print('✅ 임베딩 시스템 정상 작동')
else:
    print('❌ 검색 결과 없음 - 임베딩 문제 가능성')
"
```

### 인덱스 상태 확인

```bash
# Elasticsearch 인덱스 상태 확인
curl -X GET "localhost:9200/_cat/indices?v"

# 특정 인덱스 매핑 확인
curl -X GET "localhost:9200/documents_ko_with_embeddings_new/_mapping?pretty"

# 임베딩 필드 존재 확인
curl -X GET "localhost:9200/documents_ko_with_embeddings_new/_search?pretty" -H 'Content-Type: application/json' -d '{
  "query": {"exists": {"field": "embeddings"}},
  "size": 1,
  "_source": ["docid", "embeddings"]
}'
```

### HyDE 기능 테스트

```bash
# HyDE 검색 테스트
PYTHONPATH=src poetry run python -c "
from ir_core.query_enhancement.hyde import HyDE

hyde = HyDE()
query = '양자역학의 기본 원리'

# HyDE 검색 실행
results = hyde.retrieve_with_hyde(query, top_k=3)

print(f'HyDE 검색 결과: {len(results)}개')
for i, result in enumerate(results, 1):
    print(f'{i}. {result[\"content\"][:100]}... (점수: {result[\"score\"]:.3f})')
"
```

## 베스트 프랙티스

### 1. 변경 전 검증
```bash
# ✅ 변경 전 현재 상태 확인
PYTHONPATH=src poetry run python switch_config.py show
curl -X GET "localhost:9200/_cat/indices?v"
```

### 2. 변경 후 검증
```bash
# ✅ 변경 후 임베딩 검증
PYTHONPATH=src poetry run python -c "
from ir_core.embeddings.core import encode_query
from ir_core.retrieval.core import dense_retrieve
query_emb = encode_query('test')
results = dense_retrieve(query_emb, size=1)
print(f'✅ 검증 완료: {len(results)}개 결과')
"
```

### 3. 백업 전략
```bash
# ✅ 중요 변경 전 백업
# 인덱스 스냅샷 생성 (Elasticsearch)
curl -X PUT "localhost:9200/_snapshot/my_backup/snapshot_$(date +%Y%m%d_%H%M%S)" -H 'Content-Type: application/json' -d '{
  "indices": "documents_*",
  "ignore_unavailable": true
}'
```

### 4. 모니터링
```bash
# ✅ 정기적 상태 확인
# 크론잡이나 CI/CD에 추가
PYTHONPATH=src poetry run python -c "
# 임베딩 시스템 상태 확인
from ir_core.embeddings.core import encode_query
from ir_core.retrieval.core import dense_retrieve

try:
    emb = encode_query('health check')
    results = dense_retrieve(emb, size=1)
    print('✅ 임베딩 시스템 정상')
except Exception as e:
    print(f'❌ 임베딩 시스템 오류: {e}')
"
```

## 문제 해결 가이드

### 문제 1: "runtime error" in dense_retrieve

**증상:**
```
BadRequestError(400, 'search_phase_execution_exception', 'runtime error')
```

**해결 방법:**
```bash
# 1. 인덱스에 임베딩이 있는지 확인
curl -X GET "localhost:9200/{INDEX_NAME}/_search?pretty" -H 'Content-Type: application/json' -d '{
  "query": {"exists": {"field": "embeddings"}},
  "size": 1
}'

# 2. 임베딩이 없으면 재생성
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type korean \
  --model snunlp/KR-SBERT-V40K-klueNLI-augSTS
```

### 문제 2: 차원 불일치

**증상:**
```
ScriptException: runtime error
Caused by: IllegalArgumentException: Dense vector value missing
```

**해결 방법:**
```bash
# 1. 현재 설정 확인
PYTHONPATH=src poetry run python switch_config.py show

# 2. 올바른 모델로 재생성
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type {current_type} \
  --model {correct_model}
```

### 문제 3: OmegaConf 환경변수 보간 오류

**증상:**
```
omegaconf.errors.UnsupportedInterpolationType: Unsupported interpolation type env
```

**해결 방법:**
```yaml
# ❌ 잘못된 방식
UPSTAGE_API_KEY: ${env:UPSTAGE_API_KEY}

# ✅ 올바른 방식
UPSTAGE_API_KEY: ""  # 환경변수에서 직접 읽음
```

### 문제 4: 인덱스 손상

**증상:**
```
Index corruption or missing embeddings
```

**해결 방법:**
```bash
# 1. 손상된 인덱스 삭제
curl -X DELETE "localhost:9200/{INDEX_NAME}"

# 2. 새로 생성
PYTHONPATH=src poetry run python scripts/maintenance/recompute.py \
  --index-type {type} \
  --model {model}
```

## 결론

임베딩 관리는 RAG 시스템의 핵심 요소입니다. 이 가이드에 따라 체계적으로 임베딩을 관리하면 HyDE 검색 실패와 같은 문제를 예방할 수 있습니다.

### 핵심 원칙:
1. **설정 변경 시 항상 임베딩 재생성**
2. **변경 전후 검증 실시**
3. **백업 전략 수립**
4. **모니터링 시스템 구축**

### 빠른 참조:
- **설정 확인**: `switch_config.py show`
- **임베딩 재생성**: `recompute.py --index-type X --model Y`
- **검증**: 위의 검증 스크립트 사용
- **문제 해결**: 이 가이드의 문제 해결 섹션 참조

이 가이드를 팀 문서로 활용하여 일관된 임베딩 관리 프로세스를 유지하시기 바랍니다.