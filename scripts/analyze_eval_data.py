# scripts/analyze_eval_data.py
import os
import sys
import numpy as np
import fire
from collections import Counter

def _add_src_to_path():
    """프로젝트의 src 폴더를 경로에 추가합니다."""
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

def analyze(file_path: str = "data/eval.jsonl"):
    """
    평가 데이터셋(eval.jsonl)을 분석하여 대화 통계를 출력합니다.

    이 스크립트는 대화의 턴(turn) 수 분포를 파악하여 평가 데이터셋의
    전반적인 복잡도를 이해하는 데 도움을 줍니다.

    Args:
        file_path: 분석할 .jsonl 파일의 경로.
    """
    _add_src_to_path()
    from ir_core.utils.core import read_jsonl

    print("--- Starting Evaluation Dataset Analysis ---")

    eval_items = list(read_jsonl(file_path))
    if not eval_items:
        print(f"'{file_path}'에서 분석할 데이터를 찾을 수 없습니다.")
        return

    print(f"Analyzing {len(eval_items)} conversations from '{file_path}'...")

    # 각 대화의 턴(메시지) 수 계산
    turn_counts = [len(item.get("msg", [])) for item in eval_items]

    if not turn_counts:
        print("데이터는 있으나, 'msg' 필드가 비어 있어 분석할 수 없습니다.")
        return

    # --- 통계 계산 및 출력 ---
    counts_np = np.array(turn_counts)
    print("\n--- Conversation Turn Statistics ---")
    print(f"Total conversations analyzed: {len(counts_np)}")
    print(f"Min turns per conversation:     {np.min(counts_np)}")
    print(f"Max turns per conversation:     {np.max(counts_np)}")
    print(f"Mean turns per conversation:    {np.mean(counts_np):.2f}")
    print(f"Median turns per conversation:  {np.median(counts_np)}")
    print(f"Standard Deviation:           {np.std(counts_np):.2f}")

    # --- 턴 수 분포 출력 ---
    print("\n--- Distribution of Turn Counts ---")
    turn_distribution = Counter(turn_counts)

    # 가장 긴 턴 수를 기준으로 정렬하여 출력
    sorted_turns = sorted(turn_distribution.items())

    # 간단한 텍스트 기반 막대 그래프를 위한 스케일 계산
    max_freq = max(turn_distribution.values()) if turn_distribution else 0

    for turns, count in sorted_turns:
        # 막대 길이를 최대 빈도에 비례하여 조정
        bar_length = int(50 * count / max_freq) if max_freq > 0 else 0
        bar = '█' * bar_length
        print(f"{turns:2d} turn(s) | {count:3d} conversations | {bar}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    fire.Fire(analyze)

###  분석 스크립트 실행 및 결과 해석

# 1.  **스크립트 실행:**
#     프로젝트의 루트 디렉토리에서 아래 명령어를 실행하세요.
#     ```bash
#     poetry run python scripts/analyze_eval_data.py
#     ```
