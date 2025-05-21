# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project is a meteorological data processing system that downloads and processes BUFKIT atmospheric sounding data from the Iowa Environmental Mesonet (IEM). It consists of two main components:

1. **Data Fetcher** (`fetch_all_daily_data_multi_model.py`) - Downloads BUFKIT profiles and AFOS text products for all Weather Forecast Offices (WFOs) for specified dates and models
2. **Sounding Processor** (`sounding_processor/`) - Parses BUFKIT files and calculates derived meteorological parameters

## Architecture

The project uses a scalable multi-stage pipeline with batch processing:

### Batch Processing Architecture
1. **Batch Download**: Downloads WFO data in configurable batches (default: 10 WFOs per batch)
   - Multiple download batches can run concurrently (default: 3 workers)
   - Each WFO download is isolated and failure-resistant
2. **Batch Processing**: Processes downloaded data in larger batches (default: 25 WFOs per batch)
   - Sequential processing to avoid system overload
   - Individual WFO processing failures don't stop the batch

### Core Processing Pipeline
1. **Data Acquisition**: Downloads raw BUFKIT files and AFOS products organized by date/WFO
2. **Parsing**: Converts BUFKIT text format to structured data using `BufkitParser`
3. **Calculation**: Computes derived meteorological parameters using MetPy via `SoundingCalculator`
4. **Output**: Generates both verbose and LLM-optimized JSONL formats

Key architectural patterns:
- **Batch Processing**: Scalable concurrent downloads with sequential processing
- **Fault Isolation**: Individual WFO failures don't impact other WFOs in the batch
- **Resource Management**: Configurable batch sizes and worker limits to control system load
- **MetPy Integration**: Graceful fallbacks when MetPy is unavailable
- **Structured Data**: Uses dataclasses (`Sounding`) for consistent data representation
- **Modular Design**: Separates downloading, parsing, calculation, and output formatting

## Common Commands

### Batch Processing (Recommended)
```bash
# Complete pipeline: Download WFOs in batches of 10, process in batches of 25
python batch_fetch_and_process.py 2023-10-26 ./wfo_data_archive --hour 12 --model HRRR RAP

# Customize batch sizes
python batch_fetch_and_process.py 2023-10-26 ./wfo_data_archive \
  --download-batch-size 5 --process-batch-size 15 --max-download-workers 2

# Download only (then process separately)
python batch_fetch_and_process.py 2023-10-26 ./wfo_data_archive --download-only

# Process only (assuming data already downloaded)
python batch_fetch_and_process.py 2023-10-26 ./wfo_data_archive --process-only
```

### Legacy Single-Process Commands
```bash
# Download data for all WFOs sequentially (legacy)
python fetch_all_daily_data_multi_model.py 2023-10-26 ./wfo_data_archive --hour 12 --model HRRR

# Download single WFO
python fetch_single_wfo.py ABQ 35.04 -106.62 2023-10-26 ./wfo_data_archive --hour 12 --model RAP

# Process all BUFKIT files for a specific date
python -m sounding_processor.main ./wfo_data_archive 20231026

# Process single WFO
python -m sounding_processor.main ./wfo_data_archive 20231026 --wfo ABQ
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

### Main Scripts
- `/batch_fetch_and_process.py` - **NEW**: Main batch processing script with configurable batch sizes
- `/fetch_single_wfo.py` - **NEW**: Single WFO downloader (used by batch processor)
- `/run_batch_example.py` - **NEW**: Example script showing batch processing usage
- `/fetch_all_daily_data_multi_model.py` - Legacy: Sequential data acquisition script

### Core Processing Package (`/sounding_processor/`)
- `main.py` - Batch processing entry point (now supports single WFO processing)
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