# Information Retrieval RAG — Project Template

팀, 환경, 데이터, 및 실행 방법을 담은 템플릿 README입니다. 아래 내용을 프로젝트 상황에 따라 수정하세요.

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |


## 0. Overview
### Environment
- Development OS: Ubuntu 20.04 (recommended)
- Python: 3.10 (Poetry used for dependency management)
- Required tools: curl, tar, make, gcc (for Redis build fallback)

### Requirements
- See `pyproject.toml` for Python dependencies. Install with:

```bash
poetry install
```

## 1. Competition Info

### Overview

This repository provides a modular RAG pipeline skeleton intended for the scientific common-sense retrieval task. It supports Elasticsearch + Redis for indexing and caching, plus utilities for embeddings, indexing, retrieval, and evaluation.

### Timeline

- Start date: YYYY-MM-DD
- Final submission: YYYY-MM-DD

## 2. Components

### Directory (example)

```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data description

See `data/documents.jsonl` and `data/eval.jsonl` for examples. The project-overview in `docs/notes/project-overview.md` contains more detail on datasets, EDA, and evaluation metrics (MAP).

## 4. Modeling

Modeling details (embedding model, dense/sparse retrieval choices, rerankers) should be documented in `docs/` and corresponding notebooks in `notebooks/`.

## 5. Result

Include leaderboard screenshots, model performance, and presentation files here.

## Running services without Docker (local dev)

This repo includes helper scripts in `scripts/` to run Elasticsearch and Redis without Docker for local development. See `docs/docker-less.md` for details. Quick commands:

```bash
# Start background services (downloads if necessary)
./scripts/start-elasticsearch.sh
./scripts/start-redis.sh --prebuilt

# Run the smoke test (starts services, checks endpoints, then stops them)
./scripts/smoke-test.sh

# Cleanup downloaded distros
./scripts/cleanup-distros.sh

# Install systemd user services
./scripts/manage-services.sh install
./scripts/manage-services.sh status
./scripts/manage-services.sh uninstall
```

## Notes
- The project originally included a Windows-built Elasticsearch bundle which was removed; prefer official Linux tarballs or Docker images for production parity.

## Project Overview (competition-specific details)

See `docs/notes/project-overview.md` for a full competition write-up including dataset statistics (4,272 documents for indexing; 220 eval messages), evaluation method (MAP), and RAG architecture notes.