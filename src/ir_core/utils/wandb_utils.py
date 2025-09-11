# src/ir_core/utils/wandb_utils.py
"""
Weights & Biases (WandB) 실험 추적을 위한 헬퍼 유틸리티입니다.
이 모듈은 WandB 실행을 초기화하고, 설정값을 기반으로 유용한 실행 이름을 생성하며,
결과를 로깅하는 함수를 제공합니다.
"""
import os
from omegaconf import DictConfig, OmegaConf

# wandb 라이브러리를 안전하게 임포트합니다.
# 설치되지 않은 경우, 기능이 비활성화됩니다.
try:
    import wandb
except ImportError:
    wandb = None

def init_wandb(cfg: DictConfig):
    """
    설정(cfg)을 기반으로 WandB 실행을 초기화합니다.

    WandB가 설치되지 않았거나 cfg.wandb.enabled가 false이면 아무 작업도 수행하지 않습니다.

    Args:
        cfg: Hydra 설정 객체 (DictConfig).
    """
    if wandb is None:
        print("경고: 'wandb' 라이브러리가 설치되지 않았습니다. pip install wandb로 설치해주세요. WandB 로깅을 건너뜁니다.")
        return

    if not cfg.wandb.enabled:
        print("WandB 로깅이 비활성화되었습니다. (cfg.wandb.enabled=false)")
        return

    # WANDB_API_KEY가 설정되었는지 확인합니다.
    if not os.getenv("WANDB_API_KEY"):
        print("경고: WANDB_API_KEY 환경 변수가 설정되지 않았습니다. WandB 로깅을 건너뜁니다.")
        return

    try:
        # 실행 이름 생성
        run_name = generate_run_name(cfg)

        # WandB 실행 초기화
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True), # Hydra 설정을 dict로 변환하여 저장
            reinit=True, # 스크립트 내에서 여러 번 호출될 경우를 대비
        )
        print(f"WandB 실행이 시작되었습니다. 실행 이름: {run_name}")

    except Exception as e:
        print(f"WandB 초기화 중 오류 발생: {e}")

def generate_run_name(cfg: DictConfig) -> str:
    """
    Hydra 설정을 기반으로 설명적인 WandB 실행 이름을 생성합니다.
    예: 'dev-alpha_0.2-rerank_15'

    Args:
        cfg: Hydra 설정 객체.

    Returns:
        생성된 실행 이름 문자열.
    """
    prefix = cfg.wandb.run_name_prefix

    # 튜닝 중인 주요 파라미터를 이름에 포함시킵니다.
    alpha = cfg.params.retrieval.alpha
    rerank_k = cfg.params.retrieval.rerank_k

    return f"{prefix}-alpha_{alpha}-rerank_{rerank_k}"
