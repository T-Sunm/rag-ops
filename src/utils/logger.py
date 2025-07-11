import logging


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()
