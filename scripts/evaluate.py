"""간단한 평가 스크립트: fire 기반 CLI
한국어 출력
"""
import fire
from src.ir_core.config import settings
from src.ir_core import retrieval, evaluation as ir_eval, utils


def run(eval_path: str = 'data/eval.jsonl', out: str = 'outputs/submission.csv'):
    print('평가 실행: ', eval_path)
    items = list(utils.read_jsonl(eval_path))
    rows = []
    for it in items:
        q = it.get('msg', [{'content': ''}])[-1].get('content','')
        hy = retrieval.hybrid_retrieve(q)
        ids = [h['hit']['_id'] for h in hy]
        rows.append({'eval_id': it.get('eval_id'), 'predicted': ids})
    # 간단 CSV 출력
    with open(out,'w',encoding='utf-8') as f:
        f.write('eval_id,predicted\n')
        for r in rows:
            f.write(f"{r['eval_id']},\"{' '.join(r['predicted'])}\"\n")
    print('생성된 제출물:', out)

if __name__ == '__main__':
    fire.Fire(run)
