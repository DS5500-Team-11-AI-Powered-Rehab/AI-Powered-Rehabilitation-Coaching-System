"""
Centralized logging configuration for the rehabilitation coaching system.

Usage:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Starting coaching session...")
    logger.error("Failed to load model")
"""
import logging
import os


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        # Get log level from environment (default: INFO)
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            log_level = getattr(logging, level, logging.INFO)
        except AttributeError:
            log_level = logging.INFO

        # Create console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(log_level)

    return logger
