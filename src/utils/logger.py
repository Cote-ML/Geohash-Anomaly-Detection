import logging
import sys


def setup_logger(name):
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt=formatter)
    logger = logging.getLogger(name=name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.DEBUG)

    return logger