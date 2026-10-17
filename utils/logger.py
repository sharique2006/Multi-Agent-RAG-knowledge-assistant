"""
Structured logging for the RAG application.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
            
        return json.dumps(log_entry)


def setup_logger(name: str = "rag_app", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(JSONFormatter())
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


def log_with_data(logger: logging.Logger, level: int, message: str, data: Dict[str, Any] = None):
    """Log message with additional structured data."""
    record = logger.makeRecord(
        logger.name, level, "", 0, message, (), None
    )
    if data:
        record.extra_data = data
    logger.handle(record)


# Create default logger instance
logger = setup_logger()
