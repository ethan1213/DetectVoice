"""
Logging utilities for DetectVoice Adversarial Suite.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This code is part of a research project for audio deepfake detection and adversarial robustness.
Use responsibly and in compliance with local laws and ethical guidelines.
DO NOT use for malicious purposes.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Setup a logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ of the module)
        log_dir: Directory to save log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file created at: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
