import logging

from rich.logging import RichHandler

# Configure logger
logger = logging.getLogger("multiversum")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
