import sys

from loguru import logger

from src.settings import LATEST_LOGS_FILE_PATH


def worker_init(level: str) -> None:
    """Initialize worker process with correct logging setup."""
    setup_logger(level)


def setup_logger(level: str = "INFO") -> None:
    """Setup loguru logger with custom formatting and file output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )
    logger.add(
        LATEST_LOGS_FILE_PATH,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="1 day",
        retention="1 month",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )
