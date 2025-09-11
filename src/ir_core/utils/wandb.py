# src/ir_core/utils/wandb.py

from omegaconf import DictConfig

def generate_run_name(cfg: DictConfig) -> str:
    """
    Hydra 설정 객체를 기반으로 표준화된 WandB 실행 이름을 생성합니다.
    예: val-KR-SBERT-V40K-alpha_0.5-rerank_10-exp_prompt_tuning

    Args:
        cfg (DictConfig): Hydra 설정 객체.

    Returns:
        str: 생성된 실행 이름 문자열.
    """
    # 임베딩 모델 이름에서 주요 부분만 추출합니다 (예: 'snunlp/KR-SBERT-V40K-...' -> 'KR-SBERT')
    try:
        model_name_full = cfg.model.embedding_model
        model_name_short = model_name_full.split("/")[-1]
        # 모델 이름을 더 짧게 만들기 위해 '-'로 분리하고 처음 두 부분만 사용
        model_parts = model_name_short.split("-")
        if len(model_parts) >= 2:
            model_name_short = "-".join(model_parts[:2])
        elif len(model_parts) == 1:
            model_name_short = model_parts[0][:10]  # 긴 이름일 경우 처음 10자만 사용
    except (AttributeError, IndexError):
        model_name_short = "unknown_model"

    # 실험 이름 추출 (experiment이 설정된 경우)
    experiment_name = ""
    if hasattr(cfg, 'experiment') and cfg.experiment:
        # experiment이 dict인 경우 키를 사용
        if isinstance(cfg.experiment, dict) and cfg.experiment:
            experiment_name = list(cfg.experiment.keys())[0]
        else:
            experiment_name = str(cfg.experiment).replace('/', '_')

    # 설정값들을 조합하여 실행 이름을 만듭니다.
    run_name_parts = [
        cfg.wandb.run_name_prefix,
        model_name_short,
        f"alpha_{cfg.model.alpha}",
        f"rerank_{cfg.model.rerank_k}"
    ]

    if experiment_name:
        run_name_parts.append(f"exp_{experiment_name}")

    run_name = "-".join(run_name_parts)

    return run_name