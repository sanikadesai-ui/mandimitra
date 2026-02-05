# 🌾 MANDIMITRA - Maharashtra Agricultural Data Pipeline

**Production-quality, competition-grade data pipeline for Mandi Price Intelligence + Rainfall/Crop-Risk Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maharashtra Only](https://img.shields.io/badge/Scope-Maharashtra%20Only-orange.svg)]()

---

## ⚠️ IMPORTANT: Maharashtra-Only Pipeline

**This pipeline is STRICTLY limited to Maharashtra state.**

- 🚫 **No state CLI argument** - All scripts are hardcoded for Maharashtra
- 🚫 **Non-MH data rejected** - Any non-Maharashtra records are automatically dropped
- 🚫 **Validation fails** if non-Maharashtra data is detected
- ✅ **36 Districts** - All Maharashtra districts with HQ coordinates pre-configured
- ✅ **Chunked downloads** - Resumable by-district chunking for large datasets

This constraint exists because MANDIMITRA serves **Maharashtra farmers only**.

---

## 📋 Overview

MANDIMITRA is a robust data engineering pipeline that downloads, validates, and organizes agricultural data for Maharashtra:

1. **Mandi Price Data** - Daily commodity prices from AGMARKNET for Maharashtra markets
2. **Historical Rainfall** - NASA POWER daily precipitation for 36 Maharashtra district HQs
3. **Weather Forecasts** - Open-Meteo 16-day rainfall forecasts for Maharashtra

### Key Features

- ✅ **Maharashtra-Only**: Hard constraint - no other state data allowed
- ✅ **Resumable**: Chunked downloads with atomic progress tracking
- ✅ **Discovery Mode**: Memory-safe streaming discovery (constant memory)
- ✅ **Parallel Downloads**: ThreadPoolExecutor with configurable workers
- ✅ **Adaptive Rate Limiting**: Token bucket algorithm with 429 handling
- ✅ **Validated**: Pandera schemas with strict Maharashtra checks
- ✅ **Audited**: Markdown audit reports for compliance tracking
- ✅ **Secure**: No hardcoded secrets; API keys redacted in receipts
- ✅ **Self-Check**: Validation script to verify production standards

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- Data.gov.in API key (free): [Register here](https://data.gov.in/user/register)

### 2. Installation

```bash
# Clone or download the project
cd mandimitra

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your Data.gov.in API key
# DATAGOV_API_KEY=your_actual_api_key_here
```

### 4. Recommended Workflow

#### Quick Start: Download All Data (Recommended)

```bash
# STEP 0: Copy environment template and add your API keys
copy .env.example .env
# Edit .env to add:
#   - DATAGOV_API_KEY (for current mandi data)
#   - KAGGLE_USERNAME and KAGGLE_KEY (for historical mandi data)

# STEP 1: Run the full data pipeline
python scripts/download_all_data.py --historical-source kaggle

# OR: Skip Kaggle and import a local historical file
python scripts/download_all_data.py --historical-source local --historical-file /path/to/history.csv

# OR: Skip historical data if you already have it
python scripts/download_all_data.py --skip-historical
```

The `download_all_data.py` script orchestrates the complete pipeline:
1. **Historical Mandi** - Downloads multi-year training data (Kaggle or local)
2. **Current Mandi** - Downloads latest prices from Data.gov.in
3. **Merge/Upsert** - Combines historical + current with deduplication
4. **NASA POWER** - Downloads 10 years of historical weather
5. **Open-Meteo** - Downloads 16-day forecasts
6. **Completeness Report** - Generates Markdown summary

#### Alternative: Step-by-Step Workflow

```bash
# STEP 0: Run API self-check to verify connectivity and filters
python scripts/self_check_datagov.py --verbose

# STEP 1: Run self-check to validate codebase
python scripts/self_check.py --verbose

# STEP 2: Discover Maharashtra metadata (streaming - constant memory)
python scripts/discover_maharashtra_mandi_metadata.py --discover-fast

# STEP 3: Download Maharashtra mandi prices (parallel)
python scripts/download_mandi_maharashtra.py --download-all --max-workers 4

# STEP 4: Download weather data for all 36 district HQs (parallel)
python scripts/download_weather_maharashtra.py --all --all-districts --max-workers 2

# STEP 5: Validate all downloaded data
python scripts/validate_data.py --all-recent --strict --audit
```

---

## 🔧 Critical Technical Notes

### Data.gov.in API Filter Syntax

**CRITICAL**: The Data.gov.in API requires `filters[field.keyword]` for **exact matching**.

```bash
# ❌ WRONG - fuzzy matching (may return non-Maharashtra data)
?filters[state]=Maharashtra

# ✅ CORRECT - exact matching (Maharashtra only)
?filters[state.keyword]=Maharashtra
```

The `filters[field]` syntax does fuzzy/partial matching which can return records from
other states (e.g., "Madhya Pradesh" when filtering for "Maharashtra"). Always use
`filters[field.keyword]` for exact string matching.

This is handled automatically by `src/utils/maharashtra.py`:

```python
from src.utils.maharashtra import build_maharashtra_api_filters

# Returns: {"filters[state.keyword]": "Maharashtra"}
params = build_maharashtra_api_filters()
```

### Health Check Before Downloads

The download scripts perform an automatic health check before downloading:

1. **API Connectivity**: Verifies data.gov.in is reachable
2. **Filter Validation**: Confirms `state.keyword` filter returns ONLY Maharashtra
3. **Data Availability**: Ensures Maharashtra records exist (total > 0)
4. **Fallback**: Uses cached data if API is unavailable

Health check results are saved to `data/metadata/maharashtra/healthcheck_*.json`.

### Self-Check Script

Run `self_check_datagov.py` to diagnose API issues:

```bash
python scripts/self_check_datagov.py --verbose
```

This script:
- Verifies API key is present and valid
- Tests API connectivity and latency
- Compares `filters[state]` vs `filters[state.keyword]` behavior
- Confirms Maharashtra data availability

---

## 🔥 Full Data Pipeline (Training + Live Updates)

MANDIMITRA supports two types of mandi data:

| Data Type | Source | Purpose | Update Frequency |
|-----------|--------|---------|------------------|
| **Historical** | Kaggle (AGMARKNET archive) | ML model training | One-time download |
| **Current** | Data.gov.in API | Live price updates | Daily |

### Historical Mandi Data

Multi-year historical data for training ML models:

```bash
# Option A: Download from Kaggle (requires KAGGLE_USERNAME + KAGGLE_KEY)
python scripts/download_mandi_history_kaggle.py --download

# Option B: Import local file (CSV, ZIP, or Parquet)
python scripts/import_mandi_history.py --input-file /path/to/data.csv --import

# Preview first 10 rows before importing
python scripts/import_mandi_history.py --input-file /path/to/data.csv --preview
```

**Output:** `data/processed/mandi/history_maharashtra.parquet`

### Current Mandi Data

Daily prices from Data.gov.in API:

```bash
# Download current data (today's prices)
python scripts/download_mandi_current_datagov.py --download

# Force refresh (ignore cache)
python scripts/download_mandi_current_datagov.py --download --force-refresh
```

**Output:** `data/raw/mandi/current/YYYY-MM-DD/mandi_current.csv`

### Merge/Upsert

Combine historical + current into a single training dataset:

```bash
# Merge with current-wins strategy (newer records overwrite)
python scripts/merge_mandi_datasets.py --merge

# Preview merge without saving
python scripts/merge_mandi_datasets.py --dry-run
```

**Deduplication Key:** `[state, district, market, commodity, variety, grade, arrival_date]`

**Output:** `data/processed/mandi/mandi_maharashtra_all.parquet`

### Weather Data

Download weather for all 36 Maharashtra districts:

```bash
# NASA POWER: 10 years historical weather
python scripts/download_weather_power_maharashtra.py --download

# Open-Meteo: 16-day forecasts
python scripts/download_weather_openmeteo_maharashtra.py --download
```

### Data Completeness Report

Generate a comprehensive Markdown report:

```bash
python scripts/generate_completeness_report.py
```

**Output:** `logs/data_completeness_<timestamp>.md`

The report includes:
- Mandi data date ranges and record counts
- District/market/commodity coverage
- Weather data coverage (vs expected 36 districts)
- Missing data identification
- Recommendations for data gaps

---

## � Canonical Data Processing Pipeline (NEW)

After downloading raw data, run the **canonical data processing pipeline** to create:
- **Deduplicated** datasets with zero duplicate keys
- **Normalized** district names (35 mandi districts → 36 canonical)
- **Join-ready** outputs for ML training

### Quick Start: Build All Processed Data

```bash
# Run the full processing pipeline
python scripts/build_all_processed.py

# Dry-run (preview without saving)
python scripts/build_all_processed.py --dry-run
```

This runs three steps in order:
1. **build_canonical_mandi.py** → Deduplicated mandi data
2. **process_weather.py** → Standardized weather data
3. **build_model_datasets.py** → ML-ready joined datasets

### Output Files

```
data/processed/
├── mandi/
│   └── mandi_canonical.parquet       # Deduplicated mandi (6M+ rows)
├── weather/
│   ├── power_daily_maharashtra.parquet  # NASA POWER historical
│   └── forecast_maharashtra.parquet     # Open-Meteo forecast
├── model/
│   ├── mandi_only_2001_2026.parquet     # Full mandi history (no weather)
│   └── mandi_weather_2016plus.parquet   # Mandi + weather joined
├── dim_districts.csv                    # 36 canonical districts
└── dim_commodities.csv                  # Commodity dimension table
```

### Deduplication Strategy

The canonical pipeline uses **DuckDB** for memory-efficient SQL-based deduplication:

**Natural Key:** `(state, district, market, commodity, variety, grade, arrival_date)`

**Priority Rules:**
1. `current` source beats `history` source (newer data wins)
2. Rows with more complete prices preferred
3. Tiebreaker: highest `modal_price`

### District Normalization

Maps 35 raw mandi district names to 36 canonical Maharashtra districts:

| Raw Name | Canonical Name |
|----------|----------------|
| Sholapur | Solapur |
| Gondiya | Gondia |
| Amarawati | Amravati |
| Chattrapati Sambhajinagar | Aurangabad |
| Dharashiv (Usmanabad) | Osmanabad |
| Jalana | Jalna |
| Vashim | Washim |

### Model Datasets

**`mandi_only_2001_2026.parquet`** (full history, no weather)
- Use for: Long-term trend analysis, commodity-only models, seasonal patterns
- Rows: ~6 million
- Date range: 2001-2026

**`mandi_weather_2016plus.parquet`** (mandi + weather joined)
- Use for: Weather-aware price prediction, climate impact analysis
- Joined on: `(date, district)`
- Weather columns: `t2m_max`, `t2m_min`, `humidity`, `precipitation`, `wind_speed`, `solar_radiation`
- Date range: 2016+ (NASA POWER availability)

### QC Reports

Processing generates quality reports in `logs/`:
- `mandi_dedup_report_<timestamp>.md` - Deduplication stats
- `weather_qc_report_<timestamp>.md` - Weather validation
- `model_datasets_report_<timestamp>.md` - Final dataset summary
- `unmapped_districts_<timestamp>.md` - Any unmapped district names

---

## �📁 Project Structure

```
mandimitra/
├── configs/
│   ├── project.yaml              # Central configuration (Maharashtra settings)
│   ├── data_sources.yaml         # Data source configurations (Kaggle, APIs)
│   └── maharashtra_locations.csv # 36 district HQ coordinates
├── data/
│   ├── metadata/
│   │   └── maharashtra/          # Discovery outputs
│   │       ├── districts.csv
│   │       ├── markets.csv
│   │       ├── commodities.csv
│   │       └── discovery_receipt.json
│   ├── processed/
│   │   └── mandi/
│   │       ├── history_maharashtra.parquet  # Historical data
│   │       └── mandi_maharashtra_all.parquet # Merged training data
│   └── raw/
│       ├── mandi/
│       │   ├── maharashtra/
│       │   │   ├── {district}/        # Chunked by district
│       │   │   │   ├── mandi_{timestamp}.csv
│       │   │   │   └── receipt_{timestamp}.json
│       │   │   ├── merged/            # Combined files
│       │   │   │   └── merged_{timestamp}.csv
│       │   │   └── progress.json      # Resumability state
│       │   └── current/
│       │       └── YYYY-MM-DD/        # Date-partitioned current data
│       │           └── mandi_current.csv
│       └── weather/
│           ├── power_daily/
│           │   └── maharashtra/
│           │       └── {district}/    # Per-district historical
│           │           ├── power_daily_{start}_{end}.csv
│           │           └── receipt_{start}_{end}.json
│           └── openmeteo_forecast/
│               └── maharashtra/
│                   └── {district}/    # Per-district forecasts
│                       ├── forecast_{timestamp}.csv
│                       └── receipt_{timestamp}.json
├── logs/
│   ├── download.log
│   ├── validation.log
│   ├── data_completeness_*.md     # Completeness reports
│   └── maharashtra_*.md           # Audit reports
├── scripts/
│   ├── discover_maharashtra_mandi_metadata.py  # Streaming discovery
│   ├── download_all_data.py                    # Full pipeline orchestrator
│   ├── download_mandi_history_kaggle.py        # Kaggle historical download
│   ├── import_mandi_history.py                 # Local file import
│   ├── download_mandi_current_datagov.py       # Current mandi from API
│   ├── download_mandi_maharashtra.py           # Legacy mandi download
│   ├── merge_mandi_datasets.py                 # Upsert historical+current
│   ├── download_weather_power_maharashtra.py   # NASA POWER historical
│   ├── download_weather_openmeteo_maharashtra.py # Open-Meteo forecast
│   ├── download_weather_maharashtra.py         # Legacy weather download
│   ├── generate_completeness_report.py         # Data completeness report
│   ├── validate_data.py                        # Data validation
│   ├── self_check.py                           # Codebase validation
│   └── self_check_datagov.py                   # API connectivity check
├── src/
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── mandi.py              # Mandi data Pandera schemas
│   │   └── weather.py            # Weather data Pandera schemas
│   └── utils/
│       ├── __init__.py
│       ├── http.py               # HTTP client + rate limiting (new)
│       ├── http_utils.py         # Legacy HTTP client (deprecated)
│       ├── io_utils.py           # File I/O and receipts
│       ├── logging_utils.py      # Logging configuration
│       ├── maharashtra.py        # Maharashtra constants & validation
│       ├── progress.py           # Batched progress tracking (atomic)
│       └── audit.py              # Markdown audit reports
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📖 Usage Guide

### Step 1: Discovery (`discover_maharashtra_mandi_metadata.py`)

Query the API to find all available Maharashtra data using streaming (constant memory).

```bash
# Fast discovery (first 50 pages - recommended for quick checks)
python scripts/discover_maharashtra_mandi_metadata.py --discover-fast

# Full discovery (all records - streaming, no memory growth)
python scripts/discover_maharashtra_mandi_metadata.py --discover-full

# Force refresh existing discovery data
python scripts/discover_maharashtra_mandi_metadata.py --discover-fast --no-resume

# Verbose output
python scripts/discover_maharashtra_mandi_metadata.py --discover-fast --verbose
```

**Memory Safety**: Uses streaming pagination - memory usage is constant regardless of total records.

**Outputs:**
- `data/metadata/maharashtra/districts.csv` - Unique districts
- `data/metadata/maharashtra/markets.csv` - Unique markets
- `data/metadata/maharashtra/commodities.csv` - Unique commodities
- `data/metadata/maharashtra/discovery_receipt.json` - Metadata

### Step 2: Download Mandi Prices (`download_mandi_maharashtra.py`)

Download Maharashtra commodity prices with parallel processing and adaptive rate limiting.

```bash
# Download ALL Maharashtra data (parallel, 4 workers)
python scripts/download_mandi_maharashtra.py --download-all --max-workers 4

# Download for specific district only
python scripts/download_mandi_maharashtra.py --district "Pune"

# Resume interrupted download
python scripts/download_mandi_maharashtra.py --resume

# Force re-download (ignore progress)
python scripts/download_mandi_maharashtra.py --download-all --no-resume

# Custom rate limiting
python scripts/download_mandi_maharashtra.py --download-all --rate-limit auto --base-delay 0.5

# Trust API filter (skip per-record MH validation)
python scripts/download_mandi_maharashtra.py --download-all --trust-api-filter

# Verbose mode
python scripts/download_mandi_maharashtra.py --download-all --verbose
```

**Optimizations:**
- **Single Count Call**: Fetches total count once per chunk (no duplicate API calls)
- **Parallel Downloads**: ThreadPoolExecutor with configurable workers (max 8)
- **Adaptive Rate Limiting**: Token bucket algorithm with 429 handling
- **Batched Progress**: Atomic saves every 10 chunks
- **Summary Logging**: Non-MH violations logged in batch, not per-record

**Outputs:**
```
data/raw/mandi/maharashtra/
├── Ahmednagar/
│   ├── mandi_20260204_103045.csv
│   └── receipt_20260204_103045.json
├── Akola/
│   └── ...
├── merged/
│   └── merged_20260204_110000.csv  # All districts combined
└── progress.json                    # Resumability state
```

### Step 3: Download Weather Data (`download_weather_maharashtra.py`)

Download weather data for all 36 Maharashtra district headquarters with parallel processing.

```bash
# Download BOTH NASA POWER historical AND Open-Meteo forecasts (parallel)
python scripts/download_weather_maharashtra.py --all --all-districts --max-workers 2

# Download NASA POWER historical only (last 365 days)
python scripts/download_weather_maharashtra.py --power --all-districts

# Download Open-Meteo forecasts only (16-day)
python scripts/download_weather_maharashtra.py --openmeteo --all-districts

# Download for specific district only
python scripts/download_weather_maharashtra.py --district "Pune" --all

# Custom date range for historical
python scripts/download_weather_maharashtra.py --power --all-districts --start-date 20240101 --end-date 20241231

# Resume interrupted download
python scripts/download_weather_maharashtra.py --power --all-districts --resume

# Custom rate limiting
python scripts/download_weather_maharashtra.py --all --all-districts --rate-limit auto --base-delay 1.0
```

**Optimizations:**
- **Parallel Downloads**: ThreadPoolExecutor with configurable workers (max 4 for weather APIs)
- **Shared Rate Limiter**: Thread-safe across all workers
- **Per-Worker Sessions**: Connection pooling for each thread

**Outputs:**
```
data/raw/weather/power_daily/maharashtra/
├── Ahmednagar/
│   ├── power_daily_20240204_20250203.csv
│   └── receipt_20240204_20250203.json
├── Pune/
│   └── ...
└── progress.json

data/raw/weather/openmeteo_forecast/maharashtra/
├── Ahmednagar/
│   ├── forecast_20260204_103045.csv
│   └── receipt_20260204_103045.json
└── ...
```

### Step 4: Validate Data (`validate_data.py`)

Validate downloaded data with strict Maharashtra checks.

```bash
# Validate all recent Maharashtra files
python scripts/validate_data.py --all-recent

# Strict mode (exit code 1 if invalid, exit code 2 if non-MH found)
python scripts/validate_data.py --all-recent --strict

# Generate Markdown audit report
python scripts/validate_data.py --all-recent --audit

# Validate specific file
python scripts/validate_data.py --mandi data/raw/mandi/maharashtra/merged/merged_2025.csv

# Summary only
python scripts/validate_data.py --all-recent --summary-only
```

**Exit Codes:**
- `0` - All valid
- `1` - Validation errors (strict mode)
- `2` - **HARD CONSTRAINT VIOLATION**: Non-Maharashtra data found!
- `99` - Unexpected error

---

## ⚙️ Configuration

### Project Configuration (`configs/project.yaml`)

```yaml
project:
  name: "mandimitra"
  version: "1.0.0"
  description: "Maharashtra agricultural data pipeline"

# ========================================
# MAHARASHTRA-ONLY HARD CONSTRAINT
# ========================================
maharashtra:
  state_name: "Maharashtra"
  state_code: "MH"
  total_districts: 36

mandi:
  resource_id: "9ef84268-d588-465a-a308-a864a43d0070"
  page_size: 1000
  state_filter: "Maharashtra"  # LOCKED - cannot be overridden
  
  # Chunked download settings
  max_rows_for_bulk: 500000    # Threshold for chunked downloads
  chunk_by: "district"         # Group by district

# Weather data for district HQs
nasa_power:
  parameters: ["PRECTOTCORR", "T2M", "RH2M"]
  default_days_back: 365

openmeteo:
  forecast_days: 16
  timezone: "Asia/Kolkata"
```

### Maharashtra Locations (`configs/maharashtra_locations.csv`)

Pre-configured coordinates for all 36 Maharashtra district headquarters:

| location_id | district | district_hq | latitude | longitude | region | division |
|-------------|----------|-------------|----------|-----------|--------|----------|
| MH_PUNE | Pune | Pune | 18.5204 | 73.8567 | West | Pune |
| MH_MUMBAI | Mumbai | Mumbai | 19.0760 | 72.8777 | Konkan | Konkan |
| MH_NAGPUR | Nagpur | Nagpur | 21.1458 | 79.0882 | East | Nagpur |
| ... | ... | ... | ... | ... | ... | ... |

---

## 📊 Data Schemas

### Mandi Price Data (Maharashtra)

| Column | Type | Description | Constraint |
|--------|------|-------------|------------|
| state | string | State name | **MUST be "Maharashtra"** |
| district | string | District name | Must be valid MH district |
| market | string | Market/Mandi name | - |
| commodity | string | Commodity name | - |
| variety | string | Commodity variety | - |
| arrival_date | string | Date (DD/MM/YYYY) | - |
| min_price | float | Minimum price (Rs/Q) | ≥ 0 |
| max_price | float | Maximum price (Rs/Q) | ≥ min_price |
| modal_price | float | Modal price (Rs/Q) | ≥ 0 |

### NASA POWER Daily

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Observation date |
| PRECTOTCORR | float | Precipitation (mm/day) |
| T2M | float | Temperature at 2m (°C) |
| RH2M | float | Relative humidity (%) |

### Open-Meteo Forecast

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Forecast date |
| precipitation_sum | float | Total precipitation (mm) |
| precipitation_probability_max | float | Max probability (%) |
| temperature_2m_max | float | Max temperature (°C) |
| temperature_2m_min | float | Min temperature (°C) |

---

## 🔄 Resumability & Progress Tracking

Downloads are resumable via `progress.json`:

```json
{
  "session_id": "mandi_download_20260204_103045",
  "state": "Maharashtra",
  "strategy": "CHUNKED",
  "chunks": {
    "Ahmednagar": {"status": "COMPLETED", "rows": 12543},
    "Akola": {"status": "IN_PROGRESS", "rows": 0},
    "Amravati": {"status": "PENDING", "rows": 0}
  },
  "started_at": "2026-02-04T10:30:45Z",
  "updated_at": "2026-02-04T11:15:22Z"
}
```

To resume an interrupted download:
```bash
python scripts/download_mandi_maharashtra.py --resume
```

---

## 📝 Audit Reports

Validation can generate Markdown audit reports in `logs/`:

```markdown
# Maharashtra Data Validation

## Configuration
- **Target State**: Maharashtra
- **Strict Mode**: True
- **Data Directory**: d:\mandimitra\data\raw

## Summary
| Metric | Value |
|--------|-------|
| Total Files | 38 |
| Valid Files | 38 |
| Total Rows | 2,543,876 |
| Non-MH Records | 0 |

## Status: ✅ PASSED
Maharashtra-only constraint verified.
```

---

## 🔧 Error Handling

| Error Type | Handling |
|------------|----------|
| Missing API key | Clear error message with setup instructions |
| Rate limiting (429) | Adaptive retry with token bucket backoff |
| Server errors (5xx) | Retry up to 5 times with exponential backoff |
| Non-Maharashtra data | **AUTOMATIC DROP** - logged as batch summary |
| Validation failure | Exit code 1 (strict) or 2 (constraint violation) |
| Interrupted download | Resume with `--resume` flag |
| Progress corruption | Atomic writes prevent partial saves |

---

## 🔍 Self-Check Validation

Run the self-check script to validate the codebase meets production standards:

```bash
# Basic check
python scripts/self_check.py

# Verbose output with details
python scripts/self_check.py --verbose
```

**Checks Performed:**
- ✓ Security: No exposed API keys in code
- ✓ Security: .gitignore protects secrets
- ✓ Memory: No unbounded list growth in pagination
- ✓ Memory: CSV loaders handle comment lines
- ✓ Constraint: Maharashtra is hardcoded, not parameterized
- ✓ Quality: Using new consolidated http module
- ✓ Quality: No bare except clauses
- ✓ Performance: Rate limiting in download scripts
- ✓ Performance: Progress tracking with batched saves

---

## 🧪 Testing

```bash
# Run self-check to validate codebase
python scripts/self_check.py --verbose

# Run discovery (quick validation of API access)
python scripts/discover_maharashtra_mandi_metadata.py --discover-fast

# Download small sample (one district)
python scripts/download_mandi_maharashtra.py --district "Pune"

# Validate all data (strict mode)
python scripts/validate_data.py --all-recent --strict --audit
```

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- [Data.gov.in](https://data.gov.in) - AGMARKNET mandi price data
- [NASA POWER](https://power.larc.nasa.gov/) - Historical weather data
- [Open-Meteo](https://open-meteo.com/) - Free weather forecast API

---

**Built with ❤️ for Maharashtra Farmers**
