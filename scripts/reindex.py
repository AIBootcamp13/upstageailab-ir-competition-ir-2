"""재색인(임베딩 계산) 스크립트 (fire 기반 CLI) — 간단한 스켈레톤.
"""
import fire
from src.ir_core.embeddings import encode_texts
from src.ir_core.es_client import get_es


def run(dry: bool = True):
    print('재색인 시작 (dry={})'.format(dry))
    # 실제 인덱싱 코드는 여기에 넣으세요.
    print('노트: 이 스크립트는 스켈레톤입니다. indexing.index_documents 구현 필요.')

if __name__ == '__main__':
    fire.Fire(run)
