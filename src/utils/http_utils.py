"""
HTTP utilities for MANDIMITRA data pipeline.
Provides robust HTTP client with retries, backoff, and error handling.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class APIKeyMissingError(APIError):
    """Raised when required API key is not provided."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class EmptyResponseError(APIError):
    """Raised when API returns empty data unexpectedly."""
    pass


def create_session(
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    retry_status_codes: Optional[List[int]] = None,
    timeout: int = 60,
) -> requests.Session:
    """
    Create a requests Session with retry logic and exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff (2.0 = 1s, 2s, 4s, 8s...)
        retry_status_codes: HTTP status codes to retry on
        timeout: Default request timeout in seconds
        
    Returns:
        Configured requests.Session instance
        
    Example:
        >>> session = create_session(max_retries=3)
        >>> response = session.get("https://api.example.com/data")
    """
    if retry_status_codes is None:
        retry_status_codes = [429, 500, 502, 503, 504]
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=retry_status_codes,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Store timeout as session attribute for convenience
    session.timeout = timeout
    
    return session


def make_request(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Make an HTTP GET request with comprehensive error handling.
    
    Args:
        session: Configured requests.Session
        url: Target URL
        params: Query parameters
        headers: Request headers
        timeout: Request timeout (overrides session default)
        logger: Logger for request details
        
    Returns:
        Tuple of (response_json, status_code)
        
    Raises:
        APIError: For various API-related failures
        RateLimitError: When rate limited (429)
        requests.RequestException: For network-level failures
        
    Example:
        >>> session = create_session()
        >>> data, status = make_request(session, "https://api.example.com/data")
    """
    if timeout is None:
        timeout = getattr(session, "timeout", 60)
    
    if logger:
        # Log URL without sensitive params
        safe_params = {k: v for k, v in (params or {}).items() if "key" not in k.lower()}
        logger.debug(f"GET {url} | params: {safe_params}")
    
    try:
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        
        # Handle specific status codes
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RateLimitError(f"Rate limited. Retry after: {retry_after}s")
        
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
        
        return data, response.status_code
        
    except requests.exceptions.Timeout:
        raise APIError(f"Request timed out after {timeout}s")
    except requests.exceptions.ConnectionError as e:
        raise APIError(f"Connection error: {e}")


def paginated_fetch(
    session: requests.Session,
    url: str,
    base_params: Dict[str, Any],
    page_size: int = 1000,
    total_key: str = "total",
    records_key: str = "records",
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch all pages from a paginated API endpoint.
    
    Args:
        session: Configured requests.Session
        url: API endpoint URL
        base_params: Base query parameters (without pagination)
        page_size: Records per page
        total_key: JSON key containing total record count
        records_key: JSON key containing records array
        logger: Logger for progress updates
        
    Returns:
        Tuple of (all_records, metadata_from_first_response)
        
    Raises:
        EmptyResponseError: If no records found
        APIError: For API-related failures
        
    Example:
        >>> records, meta = paginated_fetch(session, url, {"api-key": key}, page_size=1000)
    """
    all_records = []
    offset = 0
    page = 1
    metadata = {}
    
    while True:
        params = {**base_params, "limit": page_size, "offset": offset}
        
        data, _ = make_request(session, url, params=params, logger=logger)
        
        # Store metadata from first response
        if page == 1:
            metadata = {k: v for k, v in data.items() if k != records_key}
            total = data.get(total_key, 0)
            
            if logger:
                logger.info(f"Total records available: {total:,}")
            
            if total == 0:
                raise EmptyResponseError("API returned 0 records. Check filters.")
        
        records = data.get(records_key, [])
        
        if not records:
            break
            
        all_records.extend(records)
        
        if logger:
            logger.info(f"  Page {page}: fetched {len(records):,} records (total: {len(all_records):,})")
        
        # Check if we've fetched all records
        total = data.get(total_key, 0)
        if len(all_records) >= total or len(records) < page_size:
            break
            
        offset += page_size
        page += 1
        
        # Small delay between pages to be respectful to the API
        time.sleep(0.5)
    
    if logger:
        logger.info(f"Pagination complete: {len(all_records):,} records in {page} pages")
    
    return all_records, metadata
