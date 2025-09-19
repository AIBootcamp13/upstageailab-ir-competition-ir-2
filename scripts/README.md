# Scripts Directory

This directory contains all executable scripts for the RAG system, organized into logical submodules for better maintainability and discoverability.

## Directory Structure

### Core Execution Scripts
- **`execution/`** - Main pipeline execution and query processing
  - `run_rag.py` - Full RAG pipeline with Hydra configuration
  - `run_query.py` - CLI tool for hybrid retrieval queries
  - `run-local.sh` - Local Elasticsearch/Redis management

### Evaluation & Validation
- **`evaluation/`** - Model evaluation and validation scripts
  - `evaluate.py` - Official dataset evaluation with WandB logging
  - `validate_retrieval.py` - Retrieval performance validation
  - `validate_domain_classification.py` - Domain classification accuracy
  - `smoke_test.py` - System health verification
  - `benchmark_enhancement.py` - Query enhancement benchmarking

### Data Processing
- **`data/`** - Dataset processing and analysis tools
  - `analyze_data.py` - Dataset statistics and analysis
  - `profile_documents.py` - Document profiling and metadata
  - `create_validation_set.py` - Validation dataset generation
  - `transform_submission.py` - Submission file formatting
  - `extract_scientific_terms.py` - Scientific term extraction

### System Maintenance
- **`maintenance/`** - Index management and system maintenance
  - `reindex.py` - Bulk reindexing to Elasticsearch
  - `swap_alias.py` - Atomic alias swapping
  - `index_orchestrator.py` - Complex indexing operations
  - `recompute.py` - Cache and embedding recomputation

### Infrastructure Management
- **`infra/`** - Infrastructure setup and management
  - `start-elasticsearch.sh` - Local Elasticsearch setup
  - `start-redis.sh` - Local Redis setup
  - `cleanup-distros.sh` - Distribution cleanup

### Specialized Submodules

#### Debugging & Troubleshooting
- **`debugging/`** - Performance analysis and debugging tools
  - `debug_performance.py` - RAG performance debugging

#### Indexing & Configuration
- **`indexing/`** - Document indexing and configuration management
  - `index_with_embeddings.py` - Document indexing with embeddings
  - `switch_config.py` - Korean/English configuration switching

#### Testing & Validation
- **`testing/`** - Component testing and validation
  - `test_polyglot_optimized.py` - Polyglot-Ko embedding tests
  - `test_techniques.py` - Retrieval/generation technique tests

#### CLI Tools
- **`cli/`** - Command-line interfaces
  - `cli_menu.py` - Interactive CLI menu system

#### Integration Testing
- **`integration/`** - Cross-component integration tests
  - `test_huggingface_integration.py` - HuggingFace model integration
  - `test_qwen_integration.py` - Qwen2 model integration

#### Translation & Localization
- **`translation/`** - Language translation utilities
  - `translate_validation.py` - Validation dataset translation

#### Validation & Visualization
- **`validation/`** - Result validation and visualization
  - `visualize_submissions.py` - Submission result visualization

## Usage

### Running Scripts

All scripts should be run from the project root directory using Poetry for proper dependency management:

```bash
# General pattern
PYTHONPATH=src poetry run python scripts/<submodule>/<script>.py [args]

# Examples
PYTHONPATH=src poetry run python scripts/evaluation/evaluate.py
PYTHONPATH=src poetry run python scripts/indexing/switch_config.py korean
PYTHONPATH=src poetry run python scripts/cli/cli_menu.py
```

### Script Discovery

Use the script listing tool to see all available scripts with descriptions:

```bash
PYTHONPATH=src poetry run python scripts/list_scripts.py
```

## Development Guidelines

### Adding New Scripts

1. **Choose appropriate submodule** based on script purpose
2. **Create descriptive filename** following existing patterns
3. **Add comprehensive docstring** with usage examples
4. **Update `list_scripts.py`** with script description
5. **Test script execution** from project root

### Script Organization Principles

- **Single Responsibility**: Each script should have one clear purpose
- **Consistent Naming**: Use descriptive, action-oriented names
- **Proper Documentation**: Include usage examples and parameter descriptions
- **Error Handling**: Implement robust error handling and user feedback
- **Configuration**: Use Hydra/OmegaConf for complex configurations

## Maintenance

- Regularly review and update script descriptions in `list_scripts.py`
- Remove deprecated scripts and update references
- Keep documentation synchronized with code changes
- Test scripts after dependency updates
- Ensure `PYTHONPATH=src` or run via Poetry for proper imports.
- Shell scripts are for Linux environments; check dependencies (e.g., `make` for Redis).
- Run scripts from the project root for correct relative paths.

## 한국어 문서

이 디렉토리는 정보 검색 RAG 프로젝트의 다양한 작업을 위한 실행 가능한 스크립트를 포함합니다. 스크립트는 유지보수성과 검색성을 위해 카테고리별 하위 폴더로 구성됩니다.

### 하위 폴더

#### `execution/`
시스템 실행을 위한 핵심 런타임 스크립트.
- `run_rag.py`: Hydra 구성으로 전체 RAG 파이프라인 실행.
- `run_query.py`: 하이브리드 검색 쿼리 실행을 위한 CLI 도구.
- `run-local.sh`: 로컬 Elasticsearch 및 Redis 인스턴스 관리 (시작/중지/상태).

#### `evaluation/`
시스템 테스트, 검증 및 벤치마킹을 위한 스크립트.
- `evaluate.py`: 공식 데이터셋에 대한 평가 실행 및 WandB에 로그.
- `validate_retrieval.py`: 구성 가능한 매개변수로 검색 성능 검증.
- `validate_domain_classification.py`: 도메인 분류 정확도 확인.
- `smoke_test.py`: 시스템을 위한 Python 기반 스모크 테스트.
- `smoke-test.sh`: 서비스 관리와 함께 스모크 테스트를 위한 셸 래퍼.

#### `data/`
데이터 분석, 처리 및 변환을 위한 스크립트.
- `analyze_data.py`: 문서 데이터셋 통계 분석 (예: 토큰 수).
- `check_duplicates.py`: 데이터셋의 중복 항목 감지.
- `create_validation_set.py`: LLM 프롬프트를 사용하여 검증 데이터셋 생성.
- `transform_submission.py`: 제출 파일 형식화.
- `trim_submission.py`: 제출 데이터 트리밍 및 정리.

#### `infra/`
인프라 설정 및 관리 스크립트.
- `start-elasticsearch.sh`: 로컬 Elasticsearch 다운로드 및 시작 (Linux tarball).
- `start-redis.sh`: 로컬 Redis 다운로드, 빌드 및 시작.
- `cleanup-distros.sh`: 다운로드된 배포판 정리.

#### `maintenance/`
유지보수, 데모 및 기타 작업을 위한 유틸리티.
- `reindex.py`: JSONL 파일을 Elasticsearch로 벌크 재인덱싱하는 CLI.
- `swap_alias.py`: Elasticsearch 별칭을 인덱스 간에 원자적으로 교환.
- `parallel_example.py`: 병렬 처리를 위한 예제 스크립트.
- `demo_ollama_integration.py`: Ollama 모델 통합 데모.

#### `integration/`
외부 종속성 또는 실제 모델이 필요한 시스템 및 통합 테스트.
- `test_huggingface_integration.py`: 검색 및 생성을 위한 HuggingFace 모델 통합 테스트.
- `test_qwen_integration.py`: RAG 파이프라인과 Qwen2 모델 통합 테스트.
- `test_report_generator.py`: 시스템 평가를 위한 테스트 보고서 생성 및 검증.
- `test_visualizer.py`: 테스트 결과 및 시스템 성능을 위한 시각화 생성.

### 사용법 노트
- 대부분의 Python 스크립트는 구성에 Hydra 사용 (`conf/` 디렉토리 참조).
- 적절한 임포트를 위해 `PYTHONPATH=src` 설정 또는 Poetry를 통해 실행.
- 셸 스크립트는 Linux 환경용; 종속성 확인 (예: Redis용 `make`).
- 올바른 상대 경로를 위해 프로젝트 루트에서 스크립트 실행.

### 로컬 서비스 관리 스크립트 설명
프로젝트에는 로컬 Elasticsearch 및 Redis 서비스를 관리하는 세 가지 관련 스크립트가 있습니다. 이들은 위치가 다르기 때문에 혼동을 일으킬 수 있지만, 각기 다른 용도로 설계되었습니다.

1. **`run-local.sh`** (위치: `scripts/execution/`):
   - **용도**: 로컬 개발을 위한 주요 스크립트. Elasticsearch와 Redis를 함께 시작, 중지 또는 상태 확인.
   - **사용법**:
     - 시작: `./scripts/execution/run-local.sh start`
     - 중지: `./scripts/execution/run-local.sh stop`
     - 상태: `./scripts/execution/run-local.sh status`
     - 도움말: `./scripts/execution/run-local.sh help`
   - **특징**: 두 서비스를 동시에 관리. 다운로드가 필요한 경우 자동으로 수행. PID 파일을 사용하여 프로세스 추적.
   - **권장**: 로컬 개발 시 이 스크립트를 사용하세요.

2. **`start-elasticsearch.sh`** (위치: `scripts/infra/`):
   - **용도**: Elasticsearch만 개별적으로 시작.
   - **사용법**: `./scripts/infra/start-elasticsearch.sh [--foreground] [version]`
   - **특징**: 버전 지정 가능, 포그라운드 모드 지원. 이미 실행 중이면 건너뜀.

3. **`start-redis.sh`** (위치: `scripts/infra/`):
   - **용도**: Redis만 개별적으로 시작.
   - **사용법**: `./scripts/infra/start-redis.sh [--foreground] [version]`
   - **특징**: 소스에서 빌드 필요 시 자동 빌드. 이미 실행 중이면 건너뜀.

4. **`cleanup-distros.sh`** (위치: `scripts/infra/`):
   - **용도**: 다운로드된 Elasticsearch 및 Redis 배포판 정리.
   - **사용법**: `./scripts/infra/cleanup-distros.sh`

**참고**: 스크립트 위치가 다르기 때문에 통합을 고려할 수 있지만, 현재는 기능별로 분리되어 있습니다. `run-local.sh`를 주로 사용하고, 개별 제어가 필요할 때 다른 스크립트를 사용하세요.

## Contributing
- Add new scripts to the appropriate subfolder.
- Update this README and script docstrings with usage examples.
- Test scripts before committing changes.
