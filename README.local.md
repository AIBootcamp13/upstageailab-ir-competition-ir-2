로컬 개발 실행 지침
================================

이 프로젝트는 시스템 전체에 설치하지 않고 Elasticsearch와 Redis를 로컬에서 실행할 수 있는 경량 헬퍼를 포함합니다. 헬퍼는 공식 tarball을 다운로드하고 저장소 디렉터리 아래에서 바이너리를 실행합니다. 이는 시스템 패키지를 설치할 수 없거나 설치하고 싶지 않은 재현 가능한 개발 및 CI에 유용합니다.

빠른 시작
-----------

저장소 루트에서:

```bash
# 로컬 ES와 Redis 시작 (필요시 다운로드)
scripts/run-local.sh start

# 상태 확인
scripts/run-local.sh status

# 중지
```

시스템 패키지를 사용하는 것을 선호한다면, 패키지 매니저를 사용하여 Elasticsearch와 Redis를 설치한 다음 시스템 바이너리를 선호하는 `scripts/smoke-test.sh`를 실행하세요. `scripts/smoke-test.sh`는 서비스가 없을 때 서비스를 시작하려고 시도합니다(현재 시스템 바이너리를 선호하거나, 비루트 로컬 실행을 위해 `scripts/run-local.sh start`를 실행할 수 있습니다).

요구사항
------------
- curl
- Elasticsearch를 위한 PATH의 Java 11+
- 소스에서 Redis 빌드를 위한 make & gcc (첫 번째 실행 시에만)
- 호스트에서 루트로 한 번 실행해야 할 수 있습니다:


Integration tests
-----------------

Integration tests require Elasticsearch and Redis and are marked with the `integration` pytest marker.
They are skipped by default to avoid unexpected long-running or side-effectful test runs. To run them:

Run a single integration test (one-off):

```bash
RUN_INTEGRATION=1 poetry run pytest -k integration -q -s
```

Or run all integration tests explicitly with pytest's marker selection:

```bash
# Use pytest -m integration to run only integration-marked tests
poetry run pytest -m integration -s
```

If you prefer to keep services running yourself for fast iteration, start the local services first:

```bash
./scripts/run-local.sh start
export RUN_INTEGRATION=1
poetry run pytest tests/test_integration_pipeline.py::test_full_retrieval_pipeline -s
```

```bash
sudo sysctl -w vm.max_map_count=262144
```

참고사항
-----
- 스크립트는 의도적으로 최소화되고 비루트입니다. 데이터는 저장소 내의 `elasticsearch-<version>/data`와 `redis-<version>/data` 아래에 저장됩니다.
- 프로덕션이나 시스템 전체 설치의 경우, 패키지 매니저 설치와 systemd/init 스크립트를 선호하세요.

