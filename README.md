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
- ✅ **Resumable**: Chunked downloads with progress tracking (progress.json)
- ✅ **Discovery Mode**: Query unique districts/markets/commodities before download
- ✅ **Validated**: Pandera schemas with strict Maharashtra checks
- ✅ **Audited**: Markdown audit reports for compliance tracking
- ✅ **Secure**: No hardcoded secrets; uses `.env` for API keys

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

```bash
# STEP 1: Discover Maharashtra metadata (districts, markets, commodities)
python scripts/discover_maharashtra_mandi_metadata.py

# STEP 2: Download Maharashtra mandi prices
python scripts/download_mandi_maharashtra.py --download-all

# STEP 3: Download weather data for all 36 district HQs
python scripts/download_weather_maharashtra.py --all

# STEP 4: Validate all downloaded data
python scripts/validate_data.py --all-recent --strict --audit
```

---

## 📁 Project Structure

```
mandimitra/
├── configs/
│   ├── project.yaml              # Central configuration (Maharashtra settings)
│   └── maharashtra_locations.csv # 36 district HQ coordinates
├── data/
│   ├── metadata/
│   │   └── maharashtra/          # Discovery outputs
│   │       ├── districts.csv
│   │       ├── markets.csv
│   │       ├── commodities.csv
│   │       └── discovery_receipt.json
│   └── raw/
│       ├── mandi/
│       │   └── maharashtra/
│       │       ├── {district}/        # Chunked by district
│       │       │   ├── mandi_{timestamp}.csv
│       │       │   └── receipt_{timestamp}.json
│       │       ├── merged/            # Combined files
│       │       │   └── merged_{timestamp}.csv
│       │       └── progress.json      # Resumability state
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
│   └── maharashtra_*.md           # Audit reports
├── scripts/
│   ├── discover_maharashtra_mandi_metadata.py  # Discovery step
│   ├── download_mandi_maharashtra.py           # Maharashtra mandi
│   ├── download_weather_maharashtra.py         # Maharashtra weather
│   └── validate_data.py                        # Data validation
├── src/
│   └── utils/
│       ├── __init__.py
│       ├── http_utils.py       # HTTP client with retries
│       ├── io_utils.py         # File I/O and receipts
│       ├── logging_utils.py    # Logging configuration
│       ├── maharashtra.py      # Maharashtra constants & validation
│       ├── progress.py         # Download progress tracking
│       └── audit.py            # Markdown audit reports
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📖 Usage Guide

### Step 1: Discovery (`discover_maharashtra_mandi_metadata.py`)

Query the API to find all available Maharashtra data before downloading.

```bash
# Discover unique districts, markets, and commodities
python scripts/discover_maharashtra_mandi_metadata.py

# Force refresh existing discovery data
python scripts/discover_maharashtra_mandi_metadata.py --force

# Verbose output
python scripts/discover_maharashtra_mandi_metadata.py --verbose
```

**Outputs:**
- `data/metadata/maharashtra/districts.csv` - Unique districts
- `data/metadata/maharashtra/markets.csv` - Unique markets
- `data/metadata/maharashtra/commodities.csv` - Unique commodities
- `data/metadata/maharashtra/discovery_receipt.json` - Metadata

### Step 2: Download Mandi Prices (`download_mandi_maharashtra.py`)

Download Maharashtra commodity prices with automatic chunking.

```bash
# Download ALL Maharashtra data (auto-selects bulk or chunked)
python scripts/download_mandi_maharashtra.py --download-all

# Download for specific district only
python scripts/download_mandi_maharashtra.py --district "Pune"

# Resume interrupted download
python scripts/download_mandi_maharashtra.py --resume

# Force re-download (ignore progress)
python scripts/download_mandi_maharashtra.py --download-all --force

# Verbose mode
python scripts/download_mandi_maharashtra.py --download-all --verbose
```

**Download Strategy:**
- **Bulk Mode**: If <500K rows total, downloads all at once
- **Chunked Mode**: If ≥500K rows, downloads by district for resumability

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

Download weather data for all 36 Maharashtra district headquarters.

```bash
# Download BOTH NASA POWER historical AND Open-Meteo forecasts
python scripts/download_weather_maharashtra.py --all

# Download NASA POWER historical only (last 365 days)
python scripts/download_weather_maharashtra.py --power

# Download Open-Meteo forecasts only (16-day)
python scripts/download_weather_maharashtra.py --openmeteo

# Download for specific district only
python scripts/download_weather_maharashtra.py --district "Pune" --all

# Download for all districts
python scripts/download_weather_maharashtra.py --all-districts --all

# Custom date range for historical
python scripts/download_weather_maharashtra.py --power --start 20240101 --end 20241231

# Resume interrupted download
python scripts/download_weather_maharashtra.py --power --resume
```

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
| Rate limiting (429) | Automatic retry with exponential backoff |
| Server errors (5xx) | Retry up to 5 times with backoff |
| Non-Maharashtra data | **AUTOMATIC DROP** - logged as warning |
| Validation failure | Exit code 1 (strict) or 2 (constraint violation) |
| Interrupted download | Resume with `--resume` flag |

---

## 🧪 Testing

```bash
# Run discovery (quick validation of API access)
python scripts/discover_maharashtra_mandi_metadata.py

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
