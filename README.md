
<div align="center">


[![CI](https://github.com/AIBootcamp13/upstageailab-ir-competition-upstageailab-information-retrieval_2/actions/workflows/ci.yml/badge.svg)](https://github.com/AIBootcamp13/upstageailab-ir-competition-upstageailab-information-retrieval_2/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.9.0-yellow.svg)](https://elastic.co)
[![Redis](https://img.shields.io/badge/Redis-Latest-red.svg)](https://redis.io)
[![Kibana](https://img.shields.io/badge/Kibana-8.9.0-pink.svg)](https://elastic.co/kibana)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Management-green.svg)](https://python-poetry.org)

</div>


# **Information Retrieval for Scientific Documents Q&A**



과학적 상식 검색을 위한 모듈형 RAG 파이프라인

## 👥 팀 소개
<table>
    <tr>
        <td align="center"><img src="https://avatars.githubusercontent.com/u/156163982?v=4" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/hskimh1982.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/Wchoi189/document-classifier/blob/dev-hydra/docs/images/team/AI13_%EC%B5%9C%EC%9A%A9%EB%B9%84.png?raw=true" width="180" height="180"/>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/YeonkyungKang.png" width="180" height="180"/></td>
        <td align="center"><img src="https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-2/blob/main/docs/assets/images/team/jungjaehoon.jpg" width="180" height="180"/></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/SuWuKIM">AI13_이상원</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_김효석</a></td>
        <td align="center"><a href="https://github.com/Wchoi189">AI13_최용비</a></td>
        <td align="center"><a href="https://github.com/YeonkyungKang">AI13_강연경</a></td>
        <td align="center"><a href="https://github.com/YOUR_GITHUB">AI13_정재훈</a></td>
    </tr>
    <tr>
        <td align="center">검색 알고리즘 최적화</td>
        <td align="center">툴 연동, 평가 검증</td>
        <td align="center">베이스라인, readme 작성</td>
        <td align="center">모델 최적화, 프롬트 엔지니어링</td>
        <td align="center">API 개발, Steamlit UI</td>
    </tr>
 </table>


---

## 📋 목차

- [개요](#-개요)
- [대회 정보](#-대회-정보)
- [프로젝트 구조](#-프로젝트-구조)
- [아키텍처](#️-아키텍처)
- [설치 및 실행](#-설치-및-실행)
- [사용법](#-사용법)
- [결과](#-결과)
<!-- - [현재 작업](#-현재-작업) -->

---

## 🚀 빠른 시작

이 프로젝트는 대화형 CLI 메뉴를 제공하여 일반적인 작업을 쉽게 실행할 수 있습니다. 모든 명령은 메뉴를 통해 실행할 수 있습니다.

**대화형 CLI 메뉴 실행**

**중요:** PYTHONPATH 환경 변수 설정이 필요합니다.

```bash
# 프로젝트 루트에서 실행
poetry run python cli_menu.py
```

메뉴를 통해 다음 작업들을 수행할 수 있습니다:
- 초기 설정 및 인프라 구축
- 데이터 관리 및 색인
- 실험 및 검증 실행
- 평가 및 제출 파일 생성
- **번역 기능** (한국어-영어 변환, 캐시 관리)
- 유틸리티 기능

> 자세한 메뉴 설명은 아래의 **사용법** 섹션을 참고하세요.
> 실험 및 고급 사용법에 대한 더 자세한 안내는 **[워크플로우 가이드](docs/usage/workflow-guide.md)**를 참고하세요.
> 병렬 처리에 대한 자세한 정보는 **[병렬 처리 가이드](docs/usage/parallel-processing-guide.md)**를 참고하세요.

---

## 🎯 개요

### 환경 요구사항

| 구분 | 사양 |
| OS | Ubuntu 20.04 (권장) |
| Python | 3.10 |
| 의존성 관리 | Poetry |
| 필수 도구 | curl, tar, make, gcc |

### 주요 기능

* ✅ Elasticsearch + Redis 기반 인덱싱 및 캐싱
* ✅ 모듈형 RAG 파이프라인
* ✅ **병렬 처리 지원**: 대규모 분석 시 자동 성능 최적화
* ✅ 임베딩, 검색, 평가 유틸리티 제공
* ✅ Docker 없는 로컬 개발 환경 지원
* ✅ Ollama 통합: 비용 효율적인 로컬 AI 모델
* ✅ Kibana :인데이터 시각화 및 분석

## 🏆 대회 정보

### 📊 개요

과학적 상식 검색 작업을 위한 Information Retrieval 시스템 구축

### 📅 일정

* **시작 날짜**: 2025-09-08
* **최종 제출**: 2025-09-18 (완료)

### 📈 평가 지표

* **주요 지표**: MAP (Mean Average Precision)
* **데이터셋**: 인덱싱용 4,272개 문서, 평가용 220개 쿼리

## 📁 프로젝트 구조
```
📦 프로젝트 루트
├── 📄 README.md
├── 📄 pyproject.toml
├── 📄 poetry.lock
├── 📄 poetry.toml
├── 📄 pytest.ini
├── 📂 conf/                    # 설정 파일들
├──  data/                    # 데이터셋 및 실험 결과
├── 📂 deprecated/              # deprecated 코드
├── 📂 docs/                    # 문서 및 가이드
├── 📂 models/                  # 모델 파일들
├── 📂 notebooks/               # Jupyter 노트북
├── 📂 outputs/                 # 출력 파일들
├── 📂 scripts/                 # 스크립트들
├── 📂 src/                     # 소스 코드
├── 📂 tests/                   # 테스트 코드
├── 📂 tools/                   # 도구들
├── 📂 wandb/                   # Weights & Biases 로그
├── 📂 elasticsearch-8.9.0/     # Elasticsearch 바이너리
├── 📂 redis-7.2.0/            # Redis 바이너리
├── 📂 __pycache__/            # Python 캐시
└── 기타 설정 파일들
```
#### **🔧 주요 컴포넌트**

| 모듈 | 기능 | 주요 함수 |
|------|------|-----------|
| **api** | 메인 인터페이스 | `index_documents_from_jsonl()` |
| **embeddings** | 임베딩 처리 | `encode_texts()`, `encode_query()`, `load_model()` |
| **retrieval** | 검색 엔진 | `sparse_retrieve()`, `dense_retrieve()`, `hybrid_retrieve()` |
| **generation** | 텍스트 생성 | `get_generator()`, `get_query_rewriter()` |
| **orchestration** | 파이프라인 조율 | `Pipeline`, `QueryRewriter` |
| **tools** | 도구 스키마 | `ScientificSearchArgs`, `dispatcher()` |
| **analysis** | 분석 및 시각화 | `QueryAnalyzer`, `RetrievalAnalyzer`, `Visualizer` |
| **config** | 설정 관리 | `Settings` |
| **infra** | 인프라 관리 | `get_es()`, `count_docs_with_embeddings()` |
| **utils** | 유틸리티 | `read_jsonl()`, `write_jsonl()`, `configure_logging()` |
| **evaluation** | 평가 메트릭 | `precision_at_k()`, `mrr()` |


## 🏗️ 아키텍처

#### **시스펨 플로우**

```mermaid
---
config:
  theme: "base"
  themeVariables:
    background: "#ffffff"
    primaryColor: "#4CAF50"
    primaryTextColor: "#000000"
    primaryBorderColor: "#2E7D32"
    lineColor: "#424242"
    secondaryColor: "#FFC107"
    tertiaryColor: "#FF5722"
---
flowchart TD
    A[👤 User Query] --> B[🔌 API Layer]
    B --> C[🧠 Encode Query]
    C --> D{🔍 Search Strategy}

    D -->|Sparse| E[📝 BM25 Search]
    D -->|Dense| F[🎯 Vector Search]

    E --> G[🔀 Hybrid Reranking]
    F --> G

    G --> H{⚡ Cache Check}
    H -->|Hit| I[📊 Return Results]
    H -->|Miss| J[💾 Store & Return]

    I --> K[📈 Evaluation]
    J --> K

    style A fill:#e1f5fe
    style K fill:#f3e5f5
    style G fill:#fff3e0

```

* 데이터 플로우 (요약)
  1. 사용자 쿼리 → API 수신
  2. 쿼리 임베딩 생성(임베딩 엔진)
  3. 검색 전략 선택(BM25 / Vector / Hybrid)
  4. 검색 결과를 재랭킹 및 캐시 확인(Redis)
  5. 결과 반환 및 평가 저장

> 자세한 아키텍처 다이어그램은 [시스템 개요](docs/assets/diagrams/system-overview.md), [RAG 아키텍처](docs/assets/diagrams/rag-architecture.md), [시퀀스 플로우](docs/assets/diagrams/system-sequence-flow.md)를 참고하세요.

## 🚀 설치 및 실행

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/AIBootcamp13/upstageailab-ir-competition-upstageailab-information-retrieval_2.git
cd upstageailab-ir-competition-upstageailab-information-retrieval_2
```

### 2️⃣ 의존성 설치

```bash
# Poetry를 사용한 의존성 설치
poetry install
```

### 3️⃣ 서비스 시작

로컬 Elasticsearch와 Redis를 시작하려면 대화형 CLI 메뉴의 **Setup & Infrastructure** 카테고리를 사용하세요:

```bash
poetry run python cli_menu.py
```

메뉴에서 다음 옵션들을 선택할 수 있습니다:
- Start Local Services: 로컬 서비스 시작
- Check Service Status: 서비스 상태 확인
- Stop Local Services: 서비스 중지

### 4️⃣ 초기 데이터 인덱싱

데이터 색인은 대화형 CLI 메뉴의 **Data Management** 카테고리를 통해 수행합니다:

```bash
poetry run python cli_menu.py
```

**Reindex Documents** 옵션을 선택하여 Elasticsearch에 문서를 색인할 수 있습니다.

환경 관련 팁:

* 항상 `poetry run` 또는 `poetry shell`로 가상환경을 활성화하세요. 에디터가 가상환경을 사용하지 않으면 pydantic/tqdm 등이 "탐지되지 않음"으로 표시될 수 있습니다.
* VSCode 사용 시, 왼쪽 하단 또는 Command Palette에서 Poetry 가상환경을 선택해 인터프리터를 맞추면 편리합니다.

#### 재인덱싱(재구축) 사용법

데이터 재색인은 CLI 메뉴의 **Data Management** 카테고리에서 **Reindex Documents** 옵션을 통해 쉽게 수행할 수 있습니다. 수동으로 실행해야 하는 경우:

```bash
PYTHONPATH=src poetry run python scripts/maintenance/reindex.py data/documents.jsonl --index documents_ko_with_embeddings_new --batch-size 500
```

팁:
* 배치 사이즈(`--batch-size`)를 늘리면 네트워크 왕복 횟수가 줄어들어 전체 속도가 빨라질 수 있지만, 메모리/ES 부하를 고려하세요.
* ES가 로컬에 없거나 테스트용으로 동작하지 않는 경우 `--index`를 임의의 값으로 지정해도 에러가 발생할 수 있습니다.
* 에디터에서 `elasticsearch`나 `tqdm` 같은 라이브러리가 "해결되지 않음"으로 보이면 VSCode의 Python 인터프리터를 Poetry venv로 설정하세요.

#### 테스트 데이터 인프런스 (대회 제출용)

제출 파일 생성은 CLI 메뉴의 **Evaluation & Submission** 카테고리를 통해 수행합니다:

```bash
poetry run python cli_menu.py
```

다양한 모델 옵션(OpenAI, Qwen2, Llama 등)을 선택하여 제출 파일을 생성할 수 있습니다.

---

## 📖 사용법

#### 대화형 CLI 메뉴

프로젝트의 모든 주요 작업은 대화형 CLI 메뉴를 통해 수행할 수 있습니다. 메뉴는 다음과 같은 카테고리로 구성되어 있습니다:

> **📝 참고**: 새로운 주요 기능이 추가될 때마다 CLI 메뉴 옵션을 해당 카테고리에 업데이트하세요:
> - **Setup & Infrastructure**: 인프라 설정 및 관리
> - **Data Management**: 데이터 처리 및 색인
> - **Experiments & Validation**: 모델 검증 및 테스트
> - **Evaluation & Submission**: 제출 파일 생성 및 평가
> - **Translation**: 번역 기능 및 캐시 관리
> - **Utilities**: 테스트, 모니터링 및 헬퍼 도구

#### 1. Setup & Infrastructure
프로젝트 인프라 설정 및 관리 기능을 제공합니다. 의존성 설치, 환경 설정, 로컬 서비스(Elasticsearch, Redis) 시작/중지 등의 작업을 수행할 수 있습니다.

<p align="center">
  <img src="docs/assets/images/04_refactor_visualizations/cli-menu/cli-menu-setup-infrastructure-option.png" alt="Setup & Infrastructure Menu" width="1000">
</p>

#### 2. Data Management
문서 데이터의 색인, 분석, 중복 검사 등의 데이터 관리 작업을 지원합니다. Elasticsearch에 문서를 효율적으로 색인하고 데이터 품질을 검증할 수 있습니다.

<p align="center">
  <img src="docs/assets/images/04_refactor_visualizations/cli-menu/cli-menu-data-management-options.png" alt="Data Management Menu" width="1000">
</p>

#### 3. Experiments & Validation
검색 성능 검증 및 실험을 위한 다양한 옵션을 제공합니다. 기본 검증부터 Ollama 모델을 활용한 고급 실험까지 지원하며 병렬 처리도 가능합니다.

<p align="center">
  <img src="docs/assets/images/04_refactor_visualizations/cli-menu/cli-menu-experiments-validation.png" alt="Experiments & Validation Menu" width="1000">
</p>

#### 4. Evaluation & Submission
대회 제출용 파일 생성 및 평가 작업을 수행합니다. 다양한 모델(OpenAI, Qwen2, Llama 등)을 선택하여 최적의 제출 파일을 생성할 수 있습니다.

<p align="center">
  <img src="docs/assets/images/04_refactor_visualizations/cli-menu/cli-menu-evaluation-submissions.png" alt="Evaluation & Submission Menu" width="1000">
</p>

#### 5. Translation
번역 관련 기능을 제공합니다. 한국어 질의를 영어로 번역하고, 문서 번역, 캐시 관리 등의 기능을 지원합니다.

**주요 기능:**
- **Translate Validation Data**: 검증 데이터를 한국어에서 영어로 번역
- **Translate Documents (Ollama/Google)**: Ollama 또는 Google Translate를 사용한 문서 번역
- **Validate with Translation**: 번역과 함께 검증 파이프라인 실행
- **Cache Management**: Redis 캐시 조회, 정리, 모니터링

#### 6. Utilities
시스템 상태 확인, 스크립트 목록 조회, Streamlit UI 실행 등의 유틸리티 기능을 제공합니다. 프로젝트 유지보수 및 모니터링에 유용합니다.

<p align="center">
  <img src="docs/assets/images/04_refactor_visualizations/cli-menu/cli-menu-utilities-streamlit-menu.png" alt="Utilities Menu" width="1000">
</p>

### 🔧 Configuration Management

프로젝트의 임베딩 모델, 검색 인덱스, API 설정 등을 쉽게 전환할 수 있는 구성 관리 도구입니다.

#### Quick Configuration Switch

```bash
# 한국어 설정으로 전환 (768D 임베딩)
PYTHONPATH=src poetry run python switch_config.py korean

# 영어 설정으로 전환 (768D 임베딩)
PYTHONPATH=src poetry run python switch_config.py english

# 다국어 설정으로 전환 (768D 임베딩)
PYTHONPATH=src poetry run python switch_config.py bilingual

# Solar API 설정으로 전환 (4096D 임베딩)
PYTHONPATH=src poetry run python switch_config.py solar

# 현재 설정 확인
PYTHONPATH=src poetry run python switch_config.py show
```

#### Available Configurations

| Configuration | Embedding Model | Dimensions | Index | Data File |
|---------------|----------------|------------|-------|-----------|
| `korean` | KR-SBERT-V40K-klueNLI-augSTS | 768D | documents_ko_with_embeddings_new | documents_ko.jsonl |
| `english` | KR-SBERT-V40K-klueNLI-augSTS | 768D | documents_en_with_embeddings_new | documents_bilingual.jsonl |
| `bilingual` | KR-SBERT-V40K-klueNLI-augSTS | 768D | documents_bilingual_with_embeddings_new | documents_bilingual.jsonl |
| `solar` | solar-embedding-1-large | 4096D | documents_solar_with_embeddings_new | documents_bilingual.jsonl |

> 📖 **자세한 사용법**: [Configuration Switcher 가이드](docs/usage/switch-config.md)

---

## 📊 결과

### 🏅 성능 지표

#### 📊 Private Dataset Results
| 메트릭 | 점수 | 상태 |
|--------|------|------|
| **MAP** | `0.9000` | 🏆 **Best Performance** |
| **MRR** | `0.9030` | ⭐ **Excellent** |

#### 🌐 Public Dataset Results
| 메트릭 | 점수 | 상태 |
|--------|------|------|
| **MAP** | `0.8455` | ✅ **Good Performance** |
| **MRR** | `0.8500` | ✅ **Solid Result** |

### 📈 리더보드

#### 🏆 대회 최종 순위
<p align="center">
  <img src="docs/assets/images/leaderboard_w1000.png" alt="Competition Leaderboard Results" width="1000">
</p>

**📊 순위 분석:**
- **3위**: 우리의 RAG 시스템
- **평가 지표**: MAP, MRR 기반 종합 평가
- **데이터셋**: Private + Public 리더보드 결과

> 리더보드 상세 결과 및 비교 분석은 위 차트를 참고하세요.

### 🎯 주요 성과

- ✅ **모듈형 아키텍처**: 각 컴포넌트의 독립적 개발 및 테스트 가능
- ✅ **하이브리드 검색**: BM25와 Dense Vector의 효과적 결합
- ✅ **캐싱 최적화**: Redis를 통한 응답 속도 개선
- ✅ **확장 가능성**: 새로운 임베딩 모델 및 검색 전략 쉽게 추가 가능

---


## 📚 참고 자료

### 📖 문서

- [프로젝트 상세 개요](docs/notes/project-overview.md)
## 📊 결과

### 🏅 성능 지표

#### 📊 Private Dataset Results
| 메트릭 | 점수 | 상태 |
|--------|------|------|
| **MAP** | `0.9000` | 🏆 **Best Performance** |
| **MRR** | `0.9030` | ⭐ **Excellent** |

#### 🌐 Public Dataset Results
| 메트릭 | 점수 | 상태 |
|--------|------|------|
| **MAP** | `0.8455` | ✅ **Good Performance** |
| **MRR** | `0.8500` | ✅ **Solid Result** |

### � 리더보드

#### 🏆 대회 최종 순위
<p align="center">
  <img src="docs/assets/images/leaderboard_w1000.png" alt="Competition Leaderboard Results" width="1000">
</p>


## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---
