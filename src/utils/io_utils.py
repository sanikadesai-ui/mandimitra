"""
I/O utilities for MANDIMITRA data pipeline.
Handles file operations, path management, and receipt generation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        
    Example:
        >>> config = load_config("configs/project.yaml")
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_locations(locations_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load locations CSV file.
    
    Args:
        locations_path: Path to locations CSV
        
    Returns:
        DataFrame with location data
        
    Example:
        >>> locations = load_locations("configs/locations.csv")
        >>> for _, loc in locations.iterrows():
        ...     print(loc["location_name"], loc["latitude"], loc["longitude"])
    """
    locations_path = Path(locations_path)
    
    if not locations_path.exists():
        raise FileNotFoundError(f"Locations file not found: {locations_path}")
    
    return pd.read_csv(locations_path)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
        
    Example:
        >>> output_dir = ensure_directory("data/raw/mandi/Maharashtra/Pune")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    """
    Sanitize string for use in file/folder names.
    
    Args:
        name: Original string
        
    Returns:
        Sanitized string safe for filesystem
        
    Example:
        >>> sanitize_filename("Maharashtra / Pune")
        'Maharashtra_Pune'
    """
    # Replace problematic characters
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "_",
        " ": "_",
    }
    
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Remove consecutive underscores
    while "__" in result:
        result = result.replace("__", "_")
    
    return result.strip("_")


def build_mandi_path(
    base_dir: Union[str, Path],
    state: Optional[str] = None,
    district: Optional[str] = None,
    commodity: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Build output path for mandi data following project structure.
    
    Args:
        base_dir: Base data directory (e.g., "data/raw")
        state: State name (optional)
        district: District name (optional)
        commodity: Commodity name (optional)
        timestamp: Timestamp string (optional, auto-generated if None)
        
    Returns:
        Full path for output directory
        
    Example:
        >>> path = build_mandi_path("data/raw", "Maharashtra", "Pune", "Wheat")
        >>> print(path)  # data/raw/mandi/Maharashtra/Pune/Wheat/20240115_103045
    """
    base_dir = Path(base_dir)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Build path components
    path_parts = [base_dir, "mandi"]
    
    if state:
        path_parts.append(sanitize_filename(state))
    else:
        path_parts.append("all_states")
        
    if district:
        path_parts.append(sanitize_filename(district))
    else:
        path_parts.append("all_districts")
        
    if commodity:
        path_parts.append(sanitize_filename(commodity))
    else:
        path_parts.append("all_commodities")
    
    path_parts.append(timestamp)
    
    return Path(*[str(p) for p in path_parts])


def build_weather_path(
    base_dir: Union[str, Path],
    source: str,  # "power_daily" or "openmeteo_forecast"
    location_id: str,
    filename: str,
) -> Path:
    """
    Build output path for weather data.
    
    Args:
        base_dir: Base data directory
        source: Data source identifier
        location_id: Location identifier
        filename: Output filename
        
    Returns:
        Full path for output file
        
    Example:
        >>> path = build_weather_path("data/raw", "power_daily", "LOC001", "power_daily_20230101_20231231.csv")
    """
    return Path(base_dir) / "weather" / source / sanitize_filename(location_id) / filename


def save_dataframe(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    index: bool = False,
) -> Path:
    """
    Save DataFrame to CSV with proper encoding.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        index: Whether to include row index
        
    Returns:
        Path to saved file
        
    Example:
        >>> save_dataframe(df, "data/raw/mandi/output.csv")
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    df.to_csv(output_path, index=index, encoding="utf-8")
    return output_path


def save_receipt(
    output_path: Union[str, Path],
    receipt_data: Dict[str, Any],
) -> Path:
    """
    Save download receipt as JSON.
    
    Args:
        output_path: Output file path for receipt
        receipt_data: Receipt metadata dictionary
        
    Returns:
        Path to saved receipt
        
    Example:
        >>> receipt = {"timestamp": "2024-01-15T10:30:45Z", "total_rows": 5000}
        >>> save_receipt("data/raw/mandi/receipt.json", receipt)
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    # Ensure timestamp is in receipt
    if "download_timestamp_utc" not in receipt_data:
        receipt_data["download_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2, ensure_ascii=False, default=str)
    
    return output_path


def create_download_receipt(
    dataset_id: str,
    source: str,
    filters: Dict[str, Any],
    url_params: Dict[str, Any],
    total_rows: int,
    total_pages: int,
    output_file: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized download receipt dictionary.
    
    Args:
        dataset_id: Identifier for the dataset
        source: Data source name (e.g., "data.gov.in", "NASA POWER")
        filters: Applied filters
        url_params: URL parameters used
        total_rows: Number of rows downloaded
        total_pages: Number of pages fetched
        output_file: Path to output data file
        metadata: Additional metadata from API response
        
    Returns:
        Receipt dictionary ready for JSON serialization
    """
    receipt = {
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "source": source,
        "filters": filters,
        "url_params": {k: v for k, v in url_params.items() if "key" not in k.lower()},  # Exclude API keys
        "statistics": {
            "total_rows": total_rows,
            "total_pages": total_pages,
        },
        "output_file": str(output_file),
    }
    
    if metadata:
        receipt["api_metadata"] = metadata
    
    return receipt


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
