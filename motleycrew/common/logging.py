"""Project logger configuration module

Attributes:
    logger (logging.Logger): project logger
"""
import logging


# init logger
logger = logging.getLogger("motleycrew")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False


def configure_logging(verbose: bool = False, debug: bool = False):
    """Logging configuration

    Args:
        verbose (:obj:`bool`, optional): if true logging level = INFO
        debug (:obj:`bool`, optional): if true logging level = DEBUG else WARNING

    Returns:

    """
    if debug:
        logger.setLevel(logging.DEBUG)
    elif verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
