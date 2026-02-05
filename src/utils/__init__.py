# Utility modules for MANDIMITRA
"""
Shared utility functions for I/O, HTTP, logging, and Maharashtra-specific operations.

Version: 2.0.0 (Production Refactor)
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

from .io_utils import (
    load_config,
    load_locations,
    ensure_directory,
    sanitize_filename,
    save_dataframe,
    save_receipt,
    create_download_receipt,
    redact_sensitive_params,
)

from .http import (
    APIError,
    APIKeyMissingError,
    RateLimitError,
    EmptyResponseError,
    RateLimitMode,
    AdaptiveRateLimiter,
    get_rate_limiter,
    create_session,
    make_request,
    fetch_total_count,
    stream_paginated_records,
    redact_params,
)

from .progress import (
    ChunkStatus,
    ProgressTracker,
)

from .audit import (
    AuditLogger,
)

__all__ = [
    # Maharashtra
    "MAHARASHTRA_STATE_NAME",
    "MAHARASHTRA_STATE_CODE", 
    "MAHARASHTRA_DISTRICTS",
    "MAHARASHTRA_DIVISIONS",
    "is_maharashtra_state",
    "validate_maharashtra_only",
    "normalize_district_name",
    # I/O
    "load_config",
    "load_locations",
    "ensure_directory",
    "sanitize_filename",
    "save_dataframe",
    "save_receipt",
    "create_download_receipt",
    "redact_sensitive_params",
    # HTTP
    "APIError",
    "APIKeyMissingError",
    "RateLimitError",
    "EmptyResponseError",
    "RateLimitMode",
    "AdaptiveRateLimiter",
    "get_rate_limiter",
    "create_session",
    "make_request",
    "fetch_total_count",
    "stream_paginated_records",
    "redact_params",
    # Progress
    "ChunkStatus",
    "ProgressTracker",
    # Audit
    "AuditLogger",
]
