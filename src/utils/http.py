"""
Production-grade HTTP utilities for MANDIMITRA data pipeline.

Features:
- Connection pooling with configurable pool sizes
- Adaptive rate limiting (token bucket + 429 handling)
- Exponential backoff with jitter
- Thread-safe rate limiter for concurrent downloads
- Response header-based rate limit adaptation
- API key redaction in logs/receipts

Author: MANDIMITRA Team
Version: 2.0.0 (Production Refactor)
"""

import atexit
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# EXCEPTIONS
# =============================================================================

class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class APIKeyMissingError(APIError):
    """Raised when required API key is not provided."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class EmptyResponseError(APIError):
    """Raised when API returns empty data unexpectedly."""
    pass


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimitMode(str, Enum):
    """Rate limiting strategy modes."""
    AUTO = "auto"           # Adaptive based on response headers
    FIXED = "fixed"         # Fixed delay between requests
    DISABLED = "disabled"   # No rate limiting (for testing)


@dataclass
class RateLimitState:
    """Thread-safe rate limit state tracker."""
    remaining: int = 1000
    limit: int = 1000
    reset_time: float = 0.0
    last_request_time: float = 0.0
    consecutive_429s: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """Update state from response headers (thread-safe)."""
        with self.lock:
            # Common rate limit headers
            if "X-RateLimit-Remaining" in headers:
                try:
                    self.remaining = int(headers["X-RateLimit-Remaining"])
                except ValueError:
                    pass
            if "X-RateLimit-Limit" in headers:
                try:
                    self.limit = int(headers["X-RateLimit-Limit"])
                except ValueError:
                    pass
            if "X-RateLimit-Reset" in headers:
                try:
                    self.reset_time = float(headers["X-RateLimit-Reset"])
                except ValueError:
                    pass
            self.last_request_time = time.time()
    
    def record_429(self) -> None:
        """Record a 429 response (thread-safe)."""
        with self.lock:
            self.consecutive_429s += 1
            self.remaining = 0
    
    def record_success(self) -> None:
        """Record a successful response (thread-safe)."""
        with self.lock:
            self.consecutive_429s = 0


class AdaptiveRateLimiter:
    """
    Thread-safe adaptive rate limiter using token bucket algorithm.
    
    Features:
    - Adapts to API rate limit headers
    - Handles 429 responses with exponential backoff
    - Supports Retry-After header
    - Thread-safe for concurrent downloads
    
    Example:
        >>> limiter = AdaptiveRateLimiter(mode=RateLimitMode.AUTO, base_delay=0.5)
        >>> with limiter.acquire():
        ...     response = session.get(url)
        >>> limiter.update_from_response(response)
    """
    
    def __init__(
        self,
        mode: RateLimitMode = RateLimitMode.AUTO,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        tokens_per_second: float = 2.0,
        max_tokens: int = 10,
    ):
        """
        Initialize rate limiter.
        
        Args:
            mode: Rate limiting strategy
            base_delay: Base delay between requests (for FIXED mode)
            max_delay: Maximum delay cap
            tokens_per_second: Token refill rate (for AUTO mode)
            max_tokens: Maximum token bucket size
        """
        self.mode = mode
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        
        # Token bucket state
        self._tokens = float(max_tokens)
        self._last_refill = time.time()
        self._lock = threading.Lock()
        
        # Rate limit state from API
        self._state = RateLimitState()
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.tokens_per_second)
        self._last_refill = now
    
    def _calculate_delay(self) -> float:
        """Calculate delay before next request."""
        if self.mode == RateLimitMode.DISABLED:
            return 0.0
        
        if self.mode == RateLimitMode.FIXED:
            return self.base_delay
        
        # AUTO mode: adaptive delay
        with self._state.lock:
            # Check if we need to wait for rate limit reset
            if self._state.remaining <= 0 and self._state.reset_time > time.time():
                return min(self._state.reset_time - time.time(), self.max_delay)
            
            # Exponential backoff for consecutive 429s
            if self._state.consecutive_429s > 0:
                backoff = self.base_delay * (2 ** min(self._state.consecutive_429s, 6))
                jitter = random.uniform(0, backoff * 0.1)
                return min(backoff + jitter, self.max_delay)
        
        # Token bucket: wait if no tokens
        with self._lock:
            self._refill_tokens()
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.tokens_per_second
                return min(wait_time, self.max_delay)
        
        return 0.0
    
    def acquire(self) -> None:
        """Acquire permission to make a request (blocking)."""
        delay = self._calculate_delay()
        if delay > 0:
            time.sleep(delay)
        
        # Consume a token
        with self._lock:
            self._refill_tokens()
            self._tokens = max(0, self._tokens - 1)
    
    def update_from_response(self, response: requests.Response) -> None:
        """Update rate limiter state from response."""
        self._state.update_from_headers(dict(response.headers))
        
        if response.status_code == 429:
            self._state.record_429()
        else:
            self._state.record_success()
    
    def handle_retry_after(self, response: requests.Response) -> float:
        """Get delay from Retry-After header if present."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                # Could be seconds or HTTP date
                return min(float(retry_after), self.max_delay)
            except ValueError:
                pass
        return self.base_delay * (2 ** min(self._state.consecutive_429s, 6))


# Global rate limiter instance (thread-safe singleton)
_global_rate_limiter: Optional[AdaptiveRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(
    mode: RateLimitMode = RateLimitMode.AUTO,
    base_delay: float = 0.5,
) -> AdaptiveRateLimiter:
    """Get or create the global rate limiter (thread-safe singleton)."""
    global _global_rate_limiter
    with _limiter_lock:
        if _global_rate_limiter is None:
            _global_rate_limiter = AdaptiveRateLimiter(mode=mode, base_delay=base_delay)
        return _global_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (for testing)."""
    global _global_rate_limiter
    with _limiter_lock:
        _global_rate_limiter = None


# =============================================================================
# SESSION FACTORY WITH CONNECTION POOLING
# =============================================================================

def create_session(
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    retry_status_codes: Optional[List[int]] = None,
    timeout: int = 60,
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    pool_block: bool = False,
) -> requests.Session:
    """
    Create a production-grade requests Session with connection pooling and retries.
    
    Args:
        max_retries: Maximum retry attempts per request
        backoff_factor: Exponential backoff multiplier (1.0 = 0s, 1s, 2s, 4s...)
        retry_status_codes: HTTP codes to retry (default: 429, 500, 502, 503, 504)
        timeout: Default request timeout in seconds
        pool_connections: Number of connection pools to cache
        pool_maxsize: Maximum connections per pool (for concurrent requests)
        pool_block: Block when pool is full (True) or raise error (False)
        
    Returns:
        Configured requests.Session with connection pooling
        
    Note:
        For concurrent downloads, create ONE session per thread OR use a single
        session with pool_maxsize >= max_workers.
        
    Example:
        >>> session = create_session(pool_maxsize=10, max_retries=3)
        >>> # Use with ThreadPoolExecutor (max_workers <= pool_maxsize)
    """
    if retry_status_codes is None:
        retry_status_codes = [429, 500, 502, 503, 504]
    
    session = requests.Session()
    
    # Configure retry strategy with exponential backoff
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=retry_status_codes,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        raise_on_status=False,
        # Add jitter to prevent thundering herd
        backoff_jitter=0.1,
    )
    
    # Configure connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=pool_block,
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Store config as session attributes
    session.timeout = timeout
    session._pool_maxsize = pool_maxsize
    
    return session


# =============================================================================
# API KEY REDACTION
# =============================================================================

# Keys to redact from logs and receipts
SENSITIVE_KEYS = {"api-key", "api_key", "apikey", "key", "token", "secret", "password", "auth"}


def redact_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Redact sensitive parameters for logging/receipts.
    
    Args:
        params: Original parameters dict
        
    Returns:
        Copy with sensitive values replaced by "[REDACTED]"
    """
    if not params:
        return {}
    
    redacted = {}
    for key, value in params.items():
        key_lower = key.lower().replace("-", "_")
        if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def redact_url(url: str) -> str:
    """Redact API keys from URL query strings."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    
    parsed = urlparse(url)
    if not parsed.query:
        return url
    
    params = parse_qs(parsed.query)
    redacted_params = {}
    for key, values in params.items():
        key_lower = key.lower().replace("-", "_")
        if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
            redacted_params[key] = ["[REDACTED]"]
        else:
            redacted_params[key] = values
    
    # Rebuild URL
    redacted_query = urlencode(redacted_params, doseq=True)
    return urlunparse(parsed._replace(query=redacted_query))


# =============================================================================
# HTTP REQUEST FUNCTIONS
# =============================================================================

def make_request(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    rate_limiter: Optional[AdaptiveRateLimiter] = None,
    max_429_retries: int = 3,
) -> Tuple[Dict[str, Any], requests.Response]:
    """
    Make an HTTP GET request with rate limiting and comprehensive error handling.
    
    Args:
        session: Configured requests.Session
        url: Target URL
        params: Query parameters
        headers: Request headers
        timeout: Request timeout (overrides session default)
        logger: Logger for request details (API keys will be redacted)
        rate_limiter: Optional rate limiter (uses global if None)
        max_429_retries: Maximum retries for rate limit errors
        
    Returns:
        Tuple of (response_json, response_object)
        
    Raises:
        APIError: For API failures
        RateLimitError: When rate limit exhausted after retries
        
    Example:
        >>> session = create_session()
        >>> data, response = make_request(session, url, params={"api-key": key})
    """
    if timeout is None:
        timeout = getattr(session, "timeout", 60)
    
    # Use global rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = get_rate_limiter()
    
    if logger:
        safe_params = redact_params(params)
        logger.debug(f"GET {redact_url(url)} | params: {safe_params}")
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_429_retries:
        try:
            # Acquire rate limit permission
            rate_limiter.acquire()
            
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            
            # Update rate limiter from response
            rate_limiter.update_from_response(response)
            
            # Handle 429 specifically
            if response.status_code == 429:
                retry_count += 1
                delay = rate_limiter.handle_retry_after(response)
                
                if logger:
                    logger.warning(f"Rate limited (429). Retry {retry_count}/{max_429_retries} after {delay:.1f}s")
                
                if retry_count > max_429_retries:
                    raise RateLimitError(
                        f"Rate limit exceeded after {max_429_retries} retries",
                        retry_after=delay
                    )
                
                time.sleep(delay)
                continue
            
            # Handle other errors
            if response.status_code == 401:
                raise APIKeyMissingError("Invalid or missing API key (401 Unauthorized)")
            
            if response.status_code == 403:
                raise APIError("Access forbidden (403). Check API key permissions.")
            
            if response.status_code >= 400:
                raise APIError(f"HTTP {response.status_code}: {response.text[:500]}")
            
            # Parse JSON response
            try:
                data = response.json()
            except ValueError as e:
                raise APIError(f"Invalid JSON response: {e}\nResponse: {response.text[:500]}")
            
            if logger:
                logger.debug(f"Response: {response.status_code} | Size: {len(response.content)} bytes")
            
            return data, response
            
        except requests.exceptions.Timeout:
            raise APIError(f"Request timed out after {timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise APIError(f"Connection error: {e}")
        except (RateLimitError, APIError):
            raise
        except Exception as e:
            last_error = e
            retry_count += 1
            if retry_count > max_429_retries:
                raise APIError(f"Request failed after {max_429_retries} retries: {e}")
    
    raise APIError(f"Request failed: {last_error}")


def fetch_total_count(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    total_key: str = "total",
    logger: Optional[logging.Logger] = None,
    rate_limiter: Optional[AdaptiveRateLimiter] = None,
) -> int:
    """
    Fetch total record count from API (single lightweight request).
    
    Args:
        session: HTTP session
        url: API endpoint
        params: Base parameters (will add limit=1)
        total_key: JSON key for total count
        logger: Optional logger
        rate_limiter: Optional rate limiter
        
    Returns:
        Total record count
    """
    count_params = {**params, "limit": 1, "offset": 0}
    data, _ = make_request(session, url, params=count_params, logger=logger, rate_limiter=rate_limiter)
    return data.get(total_key, 0)


# =============================================================================
# HEALTH CHECK FOR MAHARASHTRA DATA
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a Maharashtra data health check."""
    success: bool
    timestamp_utc: str
    url_redacted: str
    params_redacted: Dict[str, Any]
    status_code: Optional[int] = None
    total_records: int = 0
    first_record_state: Optional[str] = None
    sample_districts: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "success": self.success,
            "timestamp_utc": self.timestamp_utc,
            "url_redacted": self.url_redacted,
            "params_redacted": self.params_redacted,
            "status_code": self.status_code,
            "total_records": self.total_records,
            "first_record_state": self.first_record_state,
            "sample_districts": self.sample_districts,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }


def health_check_maharashtra(
    session: requests.Session,
    api_url: str,
    api_key: str,
    timeout: int = 30,
    logger: Optional[logging.Logger] = None,
) -> HealthCheckResult:
    """
    Perform a cheap health check to verify Maharashtra data availability.
    
    Makes a single request with limit=1 to check:
    - API is responding
    - Maharashtra filter returns data (total > 0)
    - First record is actually from Maharashtra
    
    Args:
        session: HTTP session
        api_url: API endpoint URL
        api_key: Data.gov.in API key
        timeout: Request timeout in seconds
        logger: Optional logger
        
    Returns:
        HealthCheckResult with status and diagnostics
        
    Example:
        >>> result = health_check_maharashtra(session, url, key)
        >>> if not result.success or result.total_records == 0:
        ...     print("Maharashtra data unavailable, using cache")
    """
    from src.utils.maharashtra import build_maharashtra_api_filters, MAHARASHTRA_STATE_NAME
    
    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    
    # Build params with correct Maharashtra filter
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 5,  # Get a few records to verify state
        "offset": 0,
    }
    params.update(build_maharashtra_api_filters())
    
    result = HealthCheckResult(
        success=False,
        timestamp_utc=timestamp,
        url_redacted=redact_url(f"{api_url}?api-key={api_key}"),
        params_redacted=redact_params(params),
    )
    
    try:
        if logger:
            logger.info(f"Health check: {redact_url(api_url)}")
        
        response = session.get(api_url, params=params, timeout=timeout)
        result.status_code = response.status_code
        result.latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 429:
            result.error_message = "Rate limited (429). Try again later."
            return result
        
        if response.status_code >= 400:
            result.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            return result
        
        data = response.json()
        result.total_records = data.get("total", 0)
        
        records = data.get("records", [])
        if records:
            first = records[0]
            result.first_record_state = first.get("state", "UNKNOWN")
            result.sample_districts = list(set(r.get("district", "") for r in records if r.get("district")))[:5]
            
            # Verify all returned records are Maharashtra
            non_mh = [r.get("state") for r in records if r.get("state", "").lower() != "maharashtra"]
            if non_mh:
                result.error_message = f"API filter failure: returned non-MH states: {non_mh}"
                return result
        
        result.success = True
        
        if logger:
            logger.info(
                f"Health check OK: total={result.total_records:,}, "
                f"first_state={result.first_record_state}, "
                f"latency={result.latency_ms:.0f}ms"
            )
        
    except requests.exceptions.Timeout:
        result.error_message = f"Request timed out after {timeout}s"
        result.latency_ms = timeout * 1000
        if logger:
            logger.error(f"Health check timeout: {timeout}s")
            
    except requests.exceptions.ConnectionError as e:
        result.error_message = f"Connection error: {str(e)[:200]}"
        if logger:
            logger.error(f"Health check connection error: {e}")
            
    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)[:200]}"
        if logger:
            logger.error(f"Health check error: {e}")
    
    return result


def save_health_check_result(
    result: HealthCheckResult,
    output_dir: Union[str, "Path"],
) -> "Path":
    """
    Save health check result to JSON file.
    
    Args:
        result: HealthCheckResult to save
        output_dir: Directory to save to
        
    Returns:
        Path to saved JSON file
    """
    import json
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"healthcheck_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Also update latest symlink/copy
    latest_path = output_dir / "healthcheck_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    return filepath


# =============================================================================
# STREAMING PAGINATION (Constant Memory)
# =============================================================================

def stream_paginated_records(
    session: requests.Session,
    url: str,
    base_params: Dict[str, Any],
    page_size: int = 1000,
    records_key: str = "records",
    total_key: str = "total",
    max_pages: Optional[int] = None,
    max_records: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    rate_limiter: Optional[AdaptiveRateLimiter] = None,
    on_page_callback: Optional[Callable[[List[Dict], int, int], None]] = None,
):
    """
    Stream paginated records with constant memory usage.
    
    Yields records one at a time, never storing all records in memory.
    Ideal for discovery operations or processing large datasets.
    
    Args:
        session: HTTP session
        url: API endpoint
        base_params: Base query parameters
        page_size: Records per page
        records_key: JSON key for records array
        total_key: JSON key for total count
        max_pages: Maximum pages to fetch (for quick discovery)
        max_records: Maximum records to yield (early stop)
        logger: Optional logger
        rate_limiter: Optional rate limiter
        on_page_callback: Optional callback(records, page_num, total_fetched)
        
    Yields:
        Individual records (dicts)
        
    Example:
        >>> for record in stream_paginated_records(session, url, params, max_pages=10):
        ...     districts.add(record.get("district"))
    """
    offset = 0
    page = 1
    total_yielded = 0
    total_available = None
    
    while True:
        # Check limits
        if max_pages and page > max_pages:
            if logger:
                logger.info(f"Reached max_pages limit ({max_pages})")
            break
        
        if max_records and total_yielded >= max_records:
            if logger:
                logger.info(f"Reached max_records limit ({max_records})")
            break
        
        # Fetch page
        params = {**base_params, "limit": page_size, "offset": offset}
        data, _ = make_request(session, url, params=params, logger=logger, rate_limiter=rate_limiter)
        
        records = data.get(records_key, [])
        
        if page == 1:
            total_available = data.get(total_key, 0)
            if logger:
                logger.info(f"Total records available: {total_available:,}")
        
        if not records:
            break
        
        # Callback for progress reporting
        if on_page_callback:
            on_page_callback(records, page, total_yielded + len(records))
        
        # Yield records individually (constant memory)
        for record in records:
            if max_records and total_yielded >= max_records:
                return
            yield record
            total_yielded += 1
        
        # Check if done
        if total_available and (offset + len(records)) >= total_available:
            break
        if len(records) < page_size:
            break
        
        offset += page_size
        page += 1
    
    if logger:
        logger.info(f"Streaming complete: yielded {total_yielded:,} records from {page} pages")


# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

# Re-export from http_utils for backward compatibility
# (allows existing code to import from either module)

__all__ = [
    # Exceptions
    "APIError",
    "APIKeyMissingError", 
    "RateLimitError",
    "EmptyResponseError",
    # Rate limiting
    "RateLimitMode",
    "AdaptiveRateLimiter",
    "get_rate_limiter",
    "reset_rate_limiter",
    # Session
    "create_session",
    # Redaction
    "redact_params",
    "redact_url",
    # Requests
    "make_request",
    "fetch_total_count",
    "stream_paginated_records",
    # Health check
    "HealthCheckResult",
    "health_check_maharashtra",
    "save_health_check_result",
]
