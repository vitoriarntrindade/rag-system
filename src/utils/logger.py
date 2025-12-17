"""Centralized logging configuration for the RAG system."""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, uses default settings
        log_to_file: Whether to log to file. If None, uses default settings
    
    Returns:
        Configured logger instance
    """
    # Get default settings (without requiring API key)
    settings = get_settings()
    
    log_level = log_level or settings.log_level
    log_to_file = log_to_file if log_to_file is not None else settings.log_to_file
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            settings.logs_dir / "rag_system.log",
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger for the application
logger = setup_logger("rag_system")
