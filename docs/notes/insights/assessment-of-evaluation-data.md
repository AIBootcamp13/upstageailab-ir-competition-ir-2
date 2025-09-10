### EDA 결과 분석 및 해석

제공해주신 실행 결과는 매우 중요한 사실을 알려줍니다.

1.  **대부분은 단일 질문:** 전체 220개의 대화 중 200개(약 91%)는 사용자의 첫 번째 메시지로 바로 질문의 의도가 드러나는 단일 턴(1-turn) 형식입니다. 이는 시스템이 대부분의 경우 빠르고 직접적으로 반응해야 함을 의미합니다.

2.  **핵심적인 다중 턴 대화의 존재:** 나머지 20개(약 9%)는 3턴으로 구성된 다중 턴 대화입니다. 이 대화들은 `"그 이유가 뭐야?"`와 같이 이전 맥락 없이는 절대 의도를 파악할 수 없는 질문을 포함하고 있습니다. **이 분석 결과는 우리가 Phase 2에서 구현한 '질의 재구성' 모듈이 왜 필수적이었는지를 명확하게 데이터로 증명합니다.** 이 기능이 없었다면, 평가 데이터셋의 약 10%에 해당하는 문제에서 실패했을 가능성이 매우 높습니다.

3.  **복잡도는 높지 않음:** 최대 턴 수가 3으로, 매우 긴 대화는 존재하지 않습니다. 따라서 현재의 접근 방식(전체 대화 기록을 프롬프트에 포함)은 컨텍스트 윈도우 초과와 같은 문제없이 효과적으로 작동할 것입니다.

**결론:** 우리의 전략은 옳았습니다. 대부분의 간단한 질문을 효율적으로 처리하면서도, 성능에 결정적인 영향을 미치는 소수의 복잡한 다중 턴 대화를 정확하게 처리할 수 있는 견고한 시스템을 구축했습니다.

### 다음 단계 제안: 성능 최적화 (Hyperparameter Tuning)

이제 우리는 견고한 파이프라인과 그 성능을 분석할 수 있는 도구(Streamlit, W&B)를 모두 갖추었습니다. 다음 논리적인 단계는 이 시스템의 성능을 극한까지 끌어올리는 **하이퍼파라미터 튜닝**입니다.

가장 먼저 시도해 볼 만한, 그리고 성능에 큰 영향을 미치는 파라미터는 **`alpha` 값**입니다. 이 값은 키워드 기반 검색(BM25)과 의미 기반 검색(Semantic)의 결과를 어떻게 조합할지 결정합니다.

**실행 계획:**

`scripts/validate_retrieval.py` 스크립트를 사용하여 다양한 `alpha` 값에 대한 MAP 점수를 체계적으로 측정하고, 우리 데이터셋에 가장 적합한 최적의 값을 찾겠습니다. 예를 들어, `0.1`, `0.3`, `0.5`, `0.7` 등의 값으로 테스트를 진행할 수 있습니다.

#### EDA 결과
```bash
--- Starting Evaluation Dataset Analysis ---
Analyzing 220 conversations from 'data/eval.jsonl'...

--- Conversation Turn Statistics ---
Total conversations analyzed: 220
Min turns per conversation:     1
Max turns per conversation:     3
Mean turns per conversation:    1.18
Median turns per conversation:  1.0
Standard Deviation:           0.57

--- Distribution of Turn Counts ---
 1 turn(s) | 200 conversations | ██████████████████████████████████████████████████
 3 turn(s) |  20 conversations | █████

--- Analysis Complete ---
```