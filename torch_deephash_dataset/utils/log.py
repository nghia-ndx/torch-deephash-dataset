from logging import INFO, basicConfig, getLogger

basicConfig(
    format='[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=INFO,
    force=True,
)

logger = getLogger(__name__)
