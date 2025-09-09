```markdown
# Elasticsearch와 Redis를 위한 Docker 없는 해결책

이 문서는 Docker 없이 Linux 호스트에서 Elasticsearch와 Redis를 실행하기 위한 `scripts/` 내의 작은 스크립트들을 설명합니다. 이는 로컬 개발 전용입니다.

## Elasticsearch

스크립트: `scripts/start-elasticsearch.sh [version]` (기본값: 8.9.0)

동작:
- `elasticsearch-<version>`이 없으면 공식 Linux tarball을 프로젝트 루트에 다운로드합니다.
- 압축을 해제하고 단일 노드 개발 모드를 위한 최소한의 `config/elasticsearch.yml`을 추가하며 보안을 비활성화합니다(로컬 개발 전용).
- 배포판의 `logs/` 폴더 아래에 로그를 남기며 백그라운드에서 Elasticsearch를 시작합니다.

참고사항:
- 편의를 위해 스크립트는 `xpack.security`를 비활성화합니다. 프로덕션에서는 사용하지 마세요.
- 다른 버전을 선호한다면 첫 번째 인수로 전달하세요.

예시:

```bash
./scripts/start-elasticsearch.sh 8.9.0
curl http://127.0.0.1:9200
```

## Redis


스크립트: `scripts/start-redis.sh [version]` (기본값: 7.2.0)

동작:
- 공식 Redis 소스 tarball을 다운로드하고 로컬에서 빌드합니다(`make`와 C 툴체인 필요).
- 프로젝트 로컬 `data/` 디렉터리를 사용하여 백그라운드에서 `redis-server`를 시작하고 `logs/` 아래에 로그를 작성합니다.

참고사항:
- Redis 빌드에는 Linux에 `build-essential` 또는 동등한 패키지가 설치되어 있어야 합니다.

예시:

```bash
./scripts/start-redis.sh 7.2.0
redis-cli -p 6379 ping
```

## 보안 및 정리
- 이 스크립트들은 개발 편의를 위한 것입니다. 프로덕션급 보안이나 복원력을 구성하지 않습니다.
- 서버를 중지하려면 스크립트에서 출력된 PID를 찾아 `kill <pid>`를 실행하세요.
- 다운로드된 배포판은 저장소 루트 아래에 배치됩니다(`.gitignore`는 `elasticsearch-*` 패턴을 무시합니다). 저장소 트리에 큰 tarball을 저장하는 것을 피하려면 사용 후 압축 해제된 폴더를 삭제하거나 임시 위치에서 스크립트를 실행하세요.

## 대안
- 나중에 Docker에 액세스할 수 있게 되면, 프로덕션과의 일치성을 위해 공식 Docker 이미지를 사용하는 것이 여전히 권장됩니다.
- 로컬 설치가 불가능한 경우 원격 관리 서비스(Elastic Cloud, 관리형 Redis)를 사용하세요.

## 추가 유틸리티 스크립트

### 서비스 관리

스크립트: `scripts/manage-services.sh [install|status|uninstall]`

동작:
- **install**: systemd 사용자 서비스로 Elasticsearch와 Redis를 설치합니다.
- **status**: 서비스 상태를 확인합니다.
- **uninstall**: 설치된 서비스를 제거합니다.

예시:

```bash
# 서비스 설치
./scripts/manage-services.sh install

# 상태 확인
./scripts/manage-services.sh status

# 서비스 제거
./scripts/manage-services.sh uninstall
```

### 정리 스크립트

스크립트: `scripts/cleanup-distros.sh`

동작:
- 다운로드된 모든 Elasticsearch 및 Redis 배포판을 정리합니다.
- 임시 파일과 로그 파일을 제거합니다.

예시:

```bash
./scripts/cleanup-distros.sh
```

### 통합 스모크 테스트

스크립트: `scripts/smoke-test.sh`

동작:
- 서비스를 시작하고 연결을 테스트한 후 자동으로 중지합니다.
- 전체 파이프라인의 기본 기능을 검증합니다.

참고: `scripts/smoke-test.sh`의 정리(cleanup) 단계는 이제 PID 파일을 확인하고
프로세스가 실제로 실행 중인지 확인한 뒤에 신호를 보냅니다. 따라서 서비스가
자발적으로 종료된 경우 "No such process"와 같은 불필요한 메시지가 표시되지
않습니다.

예시:

```bash
./scripts/smoke-test.sh
```

Flags:
- `--no-install`: do not invoke package managers; prefer local distros or existing binaries.
- `--no-cleanup`: skip stopping services after the test.

## 문제 해결

### 일반적인 문제

**포트 충돌**
```bash
# 포트 사용 확인
sudo netstat -tlnp | grep :9200  # Elasticsearch
sudo netstat -tlnp | grep :6379  # Redis

# 프로세스 종료
sudo kill -9 <PID>
```

**권한 문제**
```bash
# 실행 권한 부여
chmod +x scripts/*.sh

# 데이터 디렉터리 권한 설정
mkdir -p data logs
chmod 755 data logs
```

**빌드 도구 누락 (Redis)**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

### 로그 확인

```bash
# Elasticsearch 로그
tail -f elasticsearch-*/logs/elasticsearch.log

# Redis 로그
tail -f redis-*/logs/redis-server.log

# 스크립트 실행 로그
tail -f logs/script-execution.log
```

## 성능 튜닝

### Elasticsearch 설정

```yaml
# config/elasticsearch.yml 추가 설정
cluster.name: "dev-cluster"
node.name: "dev-node"
path.data: "./data"
path.logs: "./logs"
network.host: "127.0.0.1"
http.port: 9200
discovery.type: "single-node"

# 메모리 설정 (환경변수)
ES_JAVA_OPTS="-Xms1g -Xmx2g"
```

### Redis 설정

```conf
# redis.conf 추가 설정
maxmemory 256mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## 모니터링

### 기본 상태 확인

```bash
# Elasticsearch 클러스터 상태
curl -X GET "localhost:9200/_cluster/health?pretty"

# Redis 정보
redis-cli info server

# 시스템 리소스 사용량
htop
```

### 자동화된 모니터링

```bash
# 서비스 상태 모니터링 스크립트
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    curl -s http://localhost:9200/_cluster/health | jq '.status'
    redis-cli ping
    echo "---"
    sleep 30
done
```

이러한 도구들을 통해 Docker 없이도 효율적인 로컬 개발 환경을 구축할 수 있습니다.
```