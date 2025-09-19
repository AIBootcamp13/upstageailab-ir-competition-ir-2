# 스모크 테스트: 임베딩 + 하이브리드 검색

이 작은 스모크 테스트는 transformers 기반 임베딩 래퍼와 하이브리드 검색 재순위화기를 실행합니다. 합성 BM25 후보 목록을 사용하고 `hybrid_retrieve`를 직접 호출하기 때문에 실행 중인 Elasticsearch 클러스터가 필요하지 않습니다.

파일
- `scripts/smoke_test.py` — 독립적인 테스트 스크립트.

전제조건
- Python 3.10
- 최소 런타임 의존성 설치 (`information_retrieval_rag` 폴더 내에서 권장):

```bash
poetry install
# GPU 지원이 필요한 경우 호환되는 torch wheel을 수동으로 설치
# 플랫폼에 맞는 정확한 torch wheel 명령은 README.md를 참조하세요.
```

빠른 실행

`information_retrieval_rag` 디렉터리에서 실행:

```bash
# 스모크 테스트 실행
python scripts/smoke_test.py
```

래퍼 플래그
- `scripts/smoke-test.sh --no-install`은 시스템 패키지 설치 시도를 건너뛰고(apt/yum 없음) 로컬 배포판이나 기존 바이너리만 사용합니다. 스크립트가 패키지 매니저를 호출하지 않기를 원하는 머신에서 유용합니다.
- `scripts/smoke-test.sh --no-cleanup`은 검사를 실행하지만 시작된 서비스를 실행 상태로 둡니다(마지막에 중지하는 것을 건너뜀).

수행 작업
- `information_retrieval_rag/src/ir_core/config.py`에서 구성된 임베딩 모델을 사용하여 작은 예시 문장 세트를 인코딩합니다.
- 합성 BM25 히트 세트를 구축하고 `sparse_retrieve`를 몽키패치하여 해당 히트를 반환합니다.
- `hybrid_retrieve`를 호출하여 코사인 유사도로 후보를 재순위화하고 상위 결과를 출력합니다.

참고사항 및 관련 워크플로우
- 스모크 테스트 자체는 Elasticsearch나 Redis를 시작하지 않습니다. 저장소에는 중복되는 책임을 가진 헬퍼 스크립트들이 포함되어 있어 혼란을 야기했습니다:
  - `scripts/start-elasticsearch.sh`와 `scripts/start-redis.sh`는 시스템 패키지를 다운로드하거나 사용하여 서비스를 시작할 수 있는 헬퍼입니다.
  - `scripts/smoke-test.sh`는 서비스를 시작하려고 시도하는(현재 시스템 바이너리를 선호) 편의 래퍼이며 짧은 검사를 실행합니다.
  - 재현 가능한 비루트 워크플로우를 위해 저장소에는 이제 저장소 내에서 로컬 배포판 바이너리를 다운로드하고 실행하는 `scripts/run-local.sh`가 포함되어 있습니다(sudo 없는 개발에 권장).

  개선된 정리
  - `scripts/smoke-test.sh` 정리 단계는 이제 PID 파일을 확인하고 `kill`을 호출하기 전에 프로세스가 여전히 실행 중인지 확인합니다. 이는 서비스가 정리 전에 자체적으로 종료될 때 시끄러운 "No such process" 메시지를 방지합니다.

- 환경에 따른 권장 플로우:
  - 루트 액세스가 있고 시스템 서비스를 원하는 경우: 배포판 패키지를 통해 설치하고(`docs/docker-less.md` 참조) `scripts/smoke-test.sh`를 실행하세요.
  - 루트가 없거나 재현 가능한 로컬 실행을 선호하는 경우: `scripts/run-local.sh start`를 사용한 다음 `python scripts/smoke_test.py`를 실행하세요.

- `scripts/run-local.sh`의 첫 번째 실행은 선택된 버전을 다운로드하고 Redis를 빌드할 수 있습니다(`make`와 `gcc` 필요). 여전히 한 번 실행해야 합니다:

```bash
sudo sysctl -w vm.max_map_count=262144
```

테스트가 `ir_core` 임포트에 실패하면, `information_retrieval_rag/src`가 Python 경로에 있도록 `information_retrieval_rag` 디렉터리에서 실행하거나 다음을 실행하세요:

```bash
python -c "import sys; sys.path.insert(0,'information_retrieval_rag/src'); import scripts.smoke_test;"
```
