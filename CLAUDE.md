# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project is a meteorological data processing system that downloads and processes BUFKIT atmospheric sounding data from the Iowa Environmental Mesonet (IEM). It consists of two main components:

1. **Data Fetcher** (`fetch_all_daily_data_multi_model.py`) - Downloads BUFKIT profiles and AFOS text products for all Weather Forecast Offices (WFOs) for specified dates and models
2. **Sounding Processor** (`sounding_processor/`) - Parses BUFKIT files and calculates derived meteorological parameters

## Architecture

The project uses a multi-stage pipeline:

1. **Data Acquisition**: Downloads raw BUFKIT files and AFOS products organized by date/WFO
2. **Parsing**: Converts BUFKIT text format to structured data using `BufkitParser`
3. **Calculation**: Computes derived meteorological parameters using MetPy via `SoundingCalculator`
4. **Output**: Generates both verbose and LLM-optimized JSONL formats

Key architectural patterns:
- Uses MetPy for meteorological calculations with graceful fallbacks when unavailable
- Implements a `MetPyManager` singleton to handle MetPy imports and function availability
- Uses dataclasses (`Sounding`) for structured data representation
- Separates concerns between parsing, calculation, and output formatting

## Common Commands

### Data Fetching
```bash
# Download data for a specific date and model
python fetch_all_daily_data_multi_model.py 2023-10-26 ./wfo_data_archive --hour 12 --model HRRR

# Download multiple models
python fetch_all_daily_data_multi_model.py 2023-10-26 ./wfo_data_archive --hour 12 --model HRRR RAP GFS

# Skip BUFKIT or AFOS downloads
python fetch_all_daily_data_multi_model.py 2023-10-26 ./wfo_data_archive --skip-afos
```

### Sounding Processing
```bash
# Process all BUFKIT files for a specific date
python -m sounding_processor.main ./wfo_data_archive 20231026

# Run from project root
cd /path/to/claude-bufkit
python -m sounding_processor.main wfo_data_archive 20231026
```

### Direct LLM Conversion
```bash
# Convert existing verbose JSONL to LLM-optimized format
python -m sounding_processor.convert_to_llm_optimized_jsonl
```

## Key Configuration

- **WFO Data**: `sounding_processor/config.py` contains mappings of WFO IDs to locations and names
- **API Endpoints**: IEM base URL and API endpoints are configured in `config.py`
- **Calculation Parameters**: MetPy calculation settings, pressure levels, and thresholds
- **Output Formats**: LLM optimization mappings and key selections

## Data Flow

1. **Input**: BUFKIT files (`.buf.txt`) containing atmospheric profile data
2. **Parsing**: Extract metadata, level data, and summary parameters
3. **Calculation**: Compute CAPE, wind shear, composite indices, etc. using MetPy
4. **Output**: 
   - `*_default.jsonl` - Verbose format with all calculated parameters
   - `*_llm_optimized.jsonl` - Compact format optimized for LLM consumption

## File Structure

- `/fetch_all_daily_data_multi_model.py` - Main data acquisition script
- `/sounding_processor/` - Core processing package
  - `main.py` - Batch processing entry point
  - `bufkit_parser.py` - BUFKIT format parser
  - `sounding_calculator.py` - Meteorological calculations
  - `sounding_data.py` - Data structures
  - `config.py` - Configuration and constants
  - `convert_to_llm_optimized_jsonl.py` - Output format conversion

## Dependencies

The project relies on:
- MetPy for meteorological calculations (with graceful degradation)
- pandas/numpy for data manipulation
- requests for API calls
- Standard library modules for file I/O and argument parsing

## Error Handling

- Robust error handling for network failures during data fetching
- Graceful degradation when MetPy is unavailable
- Detailed logging to `data/output/debug_logs/`
- Continue processing even if individual soundings fail