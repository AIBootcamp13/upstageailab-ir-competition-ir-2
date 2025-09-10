#!/usr/bin/env bash
set -e

# ==============================================================================
# RAG 파이프라인의 하이퍼파라미터 튜닝 스크립트
#
# 이 스크립트는 validate_retrieval.py를 사용하여 다양한 alpha 값에 대한
# 검색 성능(MAP 점수)을 체계적으로 테스트합니다.
#
# 사용법:
# 1. 터미널에서 이 스크립트에 실행 권한을 부여합니다:
#    chmod +x scripts/run_hyperparameter_tuning.sh
# 2. 프로젝트 루트 디렉토리에서 스크립트를 실행합니다:
#    ./scripts/run_hyperparameter_tuning.sh
# ==============================================================================

# 테스트할 Alpha 값 목록
# 0.0: 순수 시맨틱 검색 (의미 기반)
# 1.0: 순수 키워드 검색 (BM25)
# 사이 값: 두 검색 방식의 조합
ALPHA_VALUES=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)

echo "🚀 Starting Hyperparameter Tuning for Alpha..."
echo "-------------------------------------------------"
echo "Test will run for the following alpha values: ${ALPHA_VALUES[*]}"
echo "Validation results will be logged to Weights & Biases."
echo "-------------------------------------------------"

for alpha in "${ALPHA_VALUES[@]}"
do
  echo ""
  echo "🧪 Testing with alpha = $alpha"

  # validate_retrieval.py 스크립트를 실행하고, --alpha 인자를 전달합니다.
  # poetry run을 사용하여 poetry 가상환경 내에서 파이썬 스크립트를 실행합니다.
  poetry run python scripts/validate_retrieval.py --alpha "$alpha"

  echo "✅ Finished test for alpha = $alpha"
  echo "-------------------------------------------------"
done

echo ""
echo "🎉 Hyperparameter tuning complete!"
echo "Visit your Weights & Biases project to compare the results and find the optimal alpha value."


### 튜닝 실행 및 최적값 적용

# 1.  **스크립트 실행 권한 부여:**
#     터미널에서 다음 명령어를 한 번만 실행하여 스크립트를 실행 가능하게 만드세요.
#     ```bash
#     chmod +x scripts/run_hyperparameter_tuning.sh
#     ```

# 2.  **하이퍼파라미터 튜닝 실행:**
#     프로젝트 루트 디렉토리에서 아래 명령어를 실행합니다.
#     ```bash
#     ./scripts/run_hyperparameter_tuning.sh
#     ```
