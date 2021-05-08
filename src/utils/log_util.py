"""Logger initalization."""
import os
import logging


def initialize():
    """Initialize logger."""
    logger = logging.getLogger(__name__)

    loglevel_name = os.environ.get("LOG_LEVEL", default="INFO")
    log_format = "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
    loglevel = logging.getLevelName(loglevel_name)
    if isinstance(loglevel, str):
        logger.warning(
            "Loglevel-Name '%s' not found in loglevels. Falling back to INFO.",
            loglevel_name,
        )
        loglevel = logging.INFO
    logging.basicConfig(format=log_format, level=loglevel)
