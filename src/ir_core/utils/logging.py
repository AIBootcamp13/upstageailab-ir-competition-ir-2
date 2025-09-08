"""로깅 설정 — utils에 포함된 헬퍼 모듈.
"""
import logging


def configure_logging(level: int = logging.INFO):
    """간단한 로깅 설정 헬퍼"""
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
