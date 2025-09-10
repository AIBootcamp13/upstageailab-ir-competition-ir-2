# scripts/visualize_results.py
import streamlit as st
import pandas as pd
import json
import os
import sys

# 프로젝트의 src 폴더를 경로에 추가하여 ir_core 모듈을 임포트할 수 있도록 함
def _add_src_to_path():
    scripts_dir = os.path.dirname(__file__)
    repo_dir = os.path.dirname(scripts_dir)
    src_dir = os.path.join(repo_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

_add_src_to_path()
from ir_core.utils.core import read_jsonl

# --- Streamlit UI 설정 ---
st.set_page_config(layout="wide", page_title="RAG Evaluation Dashboard")
st.title("📊 RAG Pipeline 결과 분석 대시보드")
st.markdown("""
이 대시보드는 `eval.jsonl`의 원본 대화와 `submission.jsonl`의 RAG 파이프라인 출력을 비교하여
모델의 성능을 정성적으로 분석할 수 있도록 돕습니다.
""")

# --- 데이터 로딩 함수 ---
@st.cache_data
def load_data(eval_path, submission_path):
    """평가 원본 데이터와 제출 파일을 로드하고 eval_id를 기준으로 병합합니다."""
    if not os.path.exists(eval_path):
        st.error(f"평가 파일을 찾을 수 없습니다: `{eval_path}`")
        return None
    if not os.path.exists(submission_path):
        st.error(f"제출 파일을 찾을 수 없습니다: `{submission_path}`. 먼저 `scripts/evaluate.py`를 실행하세요.")
        return None

    eval_df = pd.DataFrame(read_jsonl(eval_path))
    submission_df = pd.DataFrame(read_jsonl(submission_path))

    merged_df = pd.merge(eval_df, submission_df, on="eval_id", how="left")

    merged_df['standalone_query'] = merged_df['standalone_query'].fillna('')
    merged_df['answer'] = merged_df['answer'].fillna('답변 생성 실패')
    merged_df['topk'] = merged_df['topk'].apply(lambda x: x if isinstance(x, list) else [])
    merged_df['references'] = merged_df['references'].apply(lambda x: x if isinstance(x, list) else [])

    return merged_df

# --- 사이드바 ---
st.sidebar.header("파일 경로 설정")
eval_file = st.sidebar.text_input("평가 데이터 파일", "data/eval.jsonl")
submission_file = st.sidebar.text_input("제출 파일", "outputs/submission.csv")

# --- 메인 페이지 ---
data_df = load_data(eval_file, submission_file)

if data_df is not None:
    st.sidebar.header("분석 옵션")

    unique_ids = data_df['eval_id'].unique().tolist()

    # --- 페이지네이션을 위한 Session State 초기화 ---
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # --- 콜백 함수 정의 ---
    def update_index_from_select():
        # selectbox에서 선택된 값으로 인덱스를 업데이트
        selected_id = st.session_state.eval_id_selector
        st.session_state.current_index = unique_ids.index(selected_id)

    def go_next():
        # 다음 인덱스로 이동 (마지막이면 처음으로)
        st.session_state.current_index = (st.session_state.current_index + 1) % len(unique_ids)

    def go_previous():
        # 이전 인덱스로 이동 (처음이면 마지막으로)
        st.session_state.current_index = (st.session_state.current_index - 1 + len(unique_ids)) % len(unique_ids)

    # --- 사이드바 위젯 ---
    st.sidebar.selectbox(
        "분석할 `eval_id`를 선택하세요:",
        options=unique_ids,
        index=st.session_state.current_index,
        key='eval_id_selector',
        on_change=update_index_from_select
    )

    # 이전/다음 버튼
    nav_cols = st.sidebar.columns(2)
    with nav_cols[0]:
        st.button("⬅️ 이전", on_click=go_previous, use_container_width=True)
    with nav_cols[1]:
        st.button("다음 ➡️", on_click=go_next, use_container_width=True)

    # --- 선택된 결과 표시 ---
    current_id = unique_ids[st.session_state.current_index]
    st.header(f"🔍 `eval_id: {current_id}` 분석 결과 ({st.session_state.current_index + 1} / {len(unique_ids)})")

    record = data_df[data_df['eval_id'] == current_id].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💬 입력: 대화 기록")
        chat_container = st.container()
        with chat_container:
            for message in record['msg']:
                role = "사용자" if message['role'] == 'user' else "어시스턴트"
                with st.chat_message(name=message['role'], avatar="🧑‍💻" if role == "사용자" else "🤖"):
                    st.markdown(f"**{role}:** {message['content']}")

    with col2:
        st.subheader("🤖 모델 출력")
        st.info(f"**재구성된 검색 질의 (Standalone Query):**\n\n`{record['standalone_query'] or '생성되지 않음 (잡담으로 판단)'}`")
        st.success(f"**최종 생성 답변:**\n\n{record['answer']}")

    st.divider()

    st.subheader("📚 검색된 문서 (References)")
    references = record.get('references', [])
    if not references:
        st.warning("검색된 문서가 없습니다.")
    else:
        for i, ref in enumerate(references):
            # score가 None이거나 숫자가 아닐 경우를 대비
            score_val = ref.get('score')
            score_text = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else "N/A"
            with st.expander(f"**문서 {i+1} (Score: {score_text})**", expanded=i<1):
                st.text(ref.get('content', '내용 없음'))

else:
    st.warning("데이터를 로드하는 데 실패했습니다. 파일 경로를 확인해주세요.")

# # --- 실행 방법 안내 ---
# st.sidebar.markdown("---")
# st.sidebar.info("""
# **실행 방법:**
# 1. Poetry 가상환경 활성화
# 2. 프로젝트 루트 디렉토리에서 아래 명령어 실행:
# ```bash
# poetry run streamlit run scripts/visualize_results.py
# ```
# """)

