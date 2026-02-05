# Utility modules for MANDIMITRA
"""
Shared utility functions for I/O, HTTP, logging, and Maharashtra-specific operations.
"""

from .maharashtra import (
    MAHARASHTRA_STATE_NAME,
    MAHARASHTRA_STATE_CODE,
    MAHARASHTRA_DISTRICTS,
    MAHARASHTRA_DIVISIONS,
    is_maharashtra_state,
    validate_maharashtra_only,
    normalize_district_name,
)

__all__ = [
    "MAHARASHTRA_STATE_NAME",
    "MAHARASHTRA_STATE_CODE", 
    "MAHARASHTRA_DISTRICTS",
    "MAHARASHTRA_DIVISIONS",
    "is_maharashtra_state",
    "validate_maharashtra_only",
    "normalize_district_name",
]
