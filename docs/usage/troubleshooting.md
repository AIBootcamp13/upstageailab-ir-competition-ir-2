# 트러블슈팅 가이드

이 문서는 Information Retrieval 프로젝트에서 발생할 수 있는 일반적인 문제와 해결 방법을 설명합니다.

## 자주 발생하는 문제

### ConnectionRefusedError 발생 시

Elasticsearch나 Redis에 연결할 수 없는 경우:

```bash
# 서비스 상태 확인
curl -X GET "localhost:9200/_cluster/health"
redis-cli ping

# 서비스 재시작
./scripts/execution/run-local.sh stop
./scripts/execution/run-local.sh start
```

### index_not_found_exception 발생 시

Elasticsearch 인덱스가 존재하지 않는 경우:

```bash
# 인덱스 생성 및 문서 인덱싱
PYTHONPATH=src poetry run python -c "
from ir_core import api
api.index_documents_from_jsonl('data/documents.jsonl', index_name='test')
"
```

또는 제공된 CLI 사용:

```bash
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents.jsonl --index test
```

### 메모리 부족 시

Elasticsearch 메모리 부족 오류:

```bash
# Elasticsearch 힙 메모리 조정
export ES_JAVA_OPTS="-Xms1g -Xmx2g"
./scripts/execution/run-local.sh start
```

### 포트 충돌

포트가 이미 사용 중인 경우:

```bash
# 포트 사용 확인
sudo netstat -tlnp | grep :9200  # Elasticsearch
sudo netstat -tlnp | grep :6379  # Redis

# 프로세스 종료
sudo kill -9 <PID>
```

### 빌드 도구 누락 (Redis)

Redis 빌드 시 make/gcc 누락:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

## 로그 확인

```bash
# Elasticsearch 로그
tail -f elasticsearch-*/logs/elasticsearch.log

# Redis 로그
tail -f redis-*/logs/redis-server.log

# 스크립트 실행 로그 (있는 경우)
tail -f logs/script-execution.log
```

## 추가 문제 해결

### 권한 문제

```bash
# 실행 권한 부여
chmod +x scripts/execution/run-local.sh
chmod +x scripts/infra/start-*.sh

# 데이터 디렉터리 권한 설정
mkdir -p data logs
chmod 755 data logs
```

### 시스템 요구사항 확인

```bash
# Java 버전 확인 (Elasticsearch용)
java -version

# Python 환경 확인
poetry --version
python --version

# 필수 도구 확인
which curl tar make gcc
```

### 네트워크 문제

방화벽이나 네트워크 설정으로 인한 연결 문제:

```bash
# 로컬 호스트 확인
curl http://127.0.0.1:9200
redis-cli -h 127.0.0.1 ping
```

## 고급 문제 해결

### Elasticsearch 클러스터 상태

```bash
# 클러스터 건강 상태
curl -X GET "localhost:9200/_cluster/health?pretty"

# 노드 정보
curl -X GET "localhost:9200/_nodes?pretty"
```

### Redis 정보

```bash
# Redis 서버 정보
redis-cli info server

# 연결 테스트
redis-cli ping
```

### 메모리 모니터링

```bash
# 시스템 메모리 사용량
free -h

# 프로세스 메모리 사용량
ps aux --sort=-%mem | head
```

문제가 지속되면 로그 파일을 확인하고, 필요한 경우 GitHub 이슈를 생성하세요.
