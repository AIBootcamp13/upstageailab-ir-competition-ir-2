## 테스트 실행 가이드

아래는 이 저장소에서 사용 가능한 테스트들을 실행하는 방법과 자주 쓰는 옵션들입니다. 이 프로젝트는 `poetry`를 사용하여 의존성과 가상환경을 관리하므로, 모든 명령은 `poetry run`을 앞에 붙여 실행하세요.

### 전체 테스트 실행

프로젝트 루트에서 전체 테스트를 실행하려면:

```bash
poetry run pytest -q
```

설명: `-q`(quiet)는 pytest 출력을 간결하게 만들어 줍니다.

### 특정 테스트 파일 또는 테스트 케이스 실행

특정 파일만 실행하려면 경로를 지정하세요:

```bash
poetry run pytest tests/test_retrieval.py -q
```

파일 내 특정 테스트 함수만 실행하려면 `::` 표기를 사용합니다:

```bash
poetry run pytest tests/test_reindex_batching.py::test_batching_counts -q
```

### 병렬로 실행 (pytest-xdist가 설치된 경우)

의존성에 `pytest-xdist`가 포함되어 있으면 워커 수를 지정해 병렬 실행할 수 있습니다:

```bash
poetry run pytest -n auto
```

### 실패한 테스트만 다시 실행

pytest는 실패한 테스트 정보를 캐시합니다. 실패한 테스트만 재실행하려면:

```bash
poetry run pytest --lf -q
```

### 디버깅 모드로 실행

테스트 중단점 디버깅이 필요하면 `-s`(stdout) 또는 `--pdb`를 사용합니다:

```bash
poetry run pytest -q -s
poetry run pytest -q --pdb
```

### 특정 마커(예: 통합 테스트)만 실행

테스트에 마커가 사용된 경우 해당 마커만 골라 실행할 수 있습니다:

```bash
export RUN_INTEGRATION=1
poetry run pytest -m integration -q
```

### 통합 테스트 (Integration tests)

통합 테스트는 Elasticsearch와 Redis 같은 외부 서비스를 필요로 하며, `integration` pytest 마커로 표시되어 있습니다. 기본적으로는 긴 실행 시간이나 외부 사이드 이펙트를 피하기 위해 건너뛰도록 되어 있습니다. 로컬에서 통합 테스트를 실행하려면 다음 항목을 참고하세요.

- 통합 테스트 실행(마커 기반):

```bash
# 마커로 걸러서 실행
poetry run pytest -m integration -s

# 또는 키워드로 단건 실행
RUN_INTEGRATION=1 poetry run pytest -k integration -q -s
```

- 로컬 서비스를 미리 띄우고 특정 통합 테스트만 실행하려면:

```bash
./scripts/run-local.sh start
export RUN_INTEGRATION=1
poetry run pytest tests/test_integration_pipeline.py::test_full_retrieval_pipeline -s
```

- Elasticsearch를 로컬에서 실행할 때 호스트 커널 설정이 필요할 수 있습니다(예: Docker/호스트 설치 시):

```bash
sudo sysctl -w vm.max_map_count=262144
```

- 로컬 헬퍼 스크립트 설명:
	- `scripts/run-local.sh start` : 저장소 내에 포함된 Elasticsearch/Redis 바이너리를 내려받아 실행합니다(비루트, 재현 가능한 로컬 개발용).
	- `scripts/start-elasticsearch.sh`, `scripts/start-redis.sh` : 개별 서비스 시작 헬퍼입니다.

- 참고: 통합 테스트는 외부 서비스에 의존하므로 CI에서 실행할 때 별도 워크플로/태그로 관리하는 것이 안전합니다.

### 로컬 재색인/스모크 테스트 실행 예시

문서 색인 관련 스크립트(예: `scripts/reindex.py`)를 테스트하려면, 테스트에서 사용하는 것과 동일한 방식으로 `poetry run`을 사용하세요:

```bash
poetry run python scripts/reindex.py data/documents.jsonl --index test
```

### 테스트 문제 해결 팁
- 가상환경 확인: `poetry shell`로 가상환경에 진입하거나 `poetry run`으로 실행하세요.
- 의존성 설치/갱신: `poetry install` 또는 `poetry update`를 사용하세요.
- 로그/출력 보기: 실패하는 테스트에서 `-s` 옵션을 추가하면 stdout/stderr를 확인할 수 있습니다.
- 외부 서비스: 일부 통합 테스트는 Elasticsearch나 Redis 같은 서비스가 필요합니다. 관련 스크립트(`scripts/start-elasticsearch.sh`, `scripts/start-redis.sh`)를 사용해 서비스를 기동하세요.

### CI/자동화 권장사항
- 테스트는 `poetry run pytest -q`로 실행되도록 CI를 구성하세요.
- 통합 테스트는 별도 태그로 관리하여 CI에서 선택적으로 실행하세요.

---

필요하시면 이 문서에 프로젝트 특화된 테스트 실행 예시(예: 환경변수, 테스트 데이터 준비, 자주 발생하는 실패와 해결법)를 추가해 드리겠습니다.
