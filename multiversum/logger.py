import logging
import sys

logger = logging.getLogger("multiversum")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
