"""
Logging utilities for MANDIMITRA data pipeline.
Provides consistent logging configuration across all scripts.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with both console and optional file handlers.
    
    Args:
        name: Logger name (typically __name__ of calling module)
        log_file: Path to log file. If None, only console logging.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        date_format: Custom date format string
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("mandi_download", Path("logs/download.log"))
        >>> logger.info("Starting download...")
    """
    # Default formats
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.
    
    Returns:
        ISO formatted UTC timestamp string
        
    Example:
        >>> ts = get_utc_timestamp()
        >>> print(ts)  # "2024-01-15T10:30:45Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_utc_timestamp_safe() -> str:
    """
    Get current UTC timestamp safe for filenames (no colons).
    
    Returns:
        Filename-safe timestamp string
        
    Example:
        >>> ts = get_utc_timestamp_safe()
        >>> print(ts)  # "20240115_103045"
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ProgressLogger:
    """
    Context manager for logging operation progress with timing.
    
    Example:
        >>> with ProgressLogger(logger, "Downloading mandi data") as progress:
        ...     for page in pages:
        ...         download(page)
        ...         progress.update(f"Page {page} complete")
    """
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[datetime] = None
        
    def __enter__(self) -> "ProgressLogger":
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(f"▶ Starting: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"✓ Completed: {self.operation} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"✗ Failed: {self.operation} ({elapsed:.2f}s) - {exc_val}")
        return False  # Don't suppress exceptions
        
    def update(self, message: str):
        """Log a progress update."""
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        self.logger.info(f"  → {message} ({elapsed:.2f}s)")
