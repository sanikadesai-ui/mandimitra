"""
MANDIMITRA Schemas Package

Pandera schemas for data validation:
- mandi.py: Mandi price data schemas
- weather.py: Weather data schemas (NASA POWER, Open-Meteo)
"""

from src.schemas.mandi import (
    MandiSchema,
    MandiSchemaLenient,
    MANDI_SCHEMA_COLUMNS,
    MANDI_DEDUP_KEYS,
    validate_mandi_dataframe,
    summarize_mandi_data,
    normalize_columns,
    parse_dates,
)

from src.schemas.weather import (
    PowerSchema,
    OpenMeteoSchema,
    validate_power_dataframe,
    validate_openmeteo_dataframe,
    summarize_weather_data,
)

__all__ = [
    # Mandi
    "MandiSchema",
    "MandiSchemaLenient", 
    "MANDI_SCHEMA_COLUMNS",
    "MANDI_DEDUP_KEYS",
    "validate_mandi_dataframe",
    "summarize_mandi_data",
    "normalize_columns",
    "parse_dates",
    # Weather
    "PowerSchema",
    "OpenMeteoSchema",
    "validate_power_dataframe",
    "validate_openmeteo_dataframe",
    "summarize_weather_data",
]
