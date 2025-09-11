# src/ir_core/utils/wandb.py

from omegaconf import DictConfig

def generate_run_name(cfg: DictConfig) -> str:
    """
    Hydra 설정 객체를 기반으로 표준화된 WandB 실행 이름을 생성합니다.
    예: val-KR-SBERT-V40K-alpha_0.5-rerank_10

    Args:
        cfg (DictConfig): Hydra 설정 객체.

    Returns:
        str: 생성된 실행 이름 문자열.
    """
    # 임베딩 모델 이름에서 주요 부분만 추출합니다 (예: 'snunlp/KR-SBERT-V40K-...' -> 'KR-SBERT-V40K')
    try:
        model_name_full = cfg.model.embedding_model
        model_name_short = model_name_full.split("/")[-1]
    except (AttributeError, IndexError):
        model_name_short = "unknown_model"

    # 설정값들을 조합하여 실행 이름을 만듭니다.
    run_name = (
        f"{cfg.wandb.run_name_prefix}-"
        f"{model_name_short}-"
        f"alpha_{cfg.model.alpha}-"
        f"rerank_{cfg.model.rerank_k}"
    )

    return run_name