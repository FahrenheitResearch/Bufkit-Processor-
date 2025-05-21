# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude-Bufkit Status Report

### What Works ✅

#### Downloads (Fast & Reliable)
- **enhanced_ultra_processor.py** - EXCELLENT for downloads
  - Downloads all models: `--model GFS HRRR NAM NAM4KM RAP`
  - Downloads all AFOS products (AFDs, storm reports, warnings)
  - Ultra-fast: 50-100 concurrent workers
  - Example: 2356 files (122 BUFKIT + 2234 AFOS) in 31 seconds
  - Command: `python enhanced_ultra_processor.py 2025-03-14 ./wfo_data_archive --model GFS HRRR RAP --download-only`

#### Processing (Slow but Works)
- **batch_fetch_and_process.py** - WORKS but slow
  - Processes sequentially (one WFO at a time)
  - Takes ~40+ seconds per WFO with multiple models
  - Command: `python batch_fetch_and_process.py 2025-03-14 ./wfo_data_archive --process-only`

#### Individual WFO Processing
- **sounding_processor/main.py** - WORKS for single WFOs
  - Fast for individual processing
  - Command: `cd sounding_processor && python main.py --wfo ABQ ../wfo_data_archive/20250314/ABQ`

### What Doesn't Work ❌

#### Parallel Processing
- **enhanced_ultra_processor.py** processing phase - BROKEN
- **ultra_parallel_processor.py** processing phase - BROKEN  
- **All multiprocessing approaches** - BROKEN
- Error: `cannot pickle '_thread.lock' object`
- Root cause: MetPy/sounding calculation objects contain unpickleable threading locks

### Performance Analysis

#### Current State
- **Downloads**: 🚀 BLAZING FAST (3.9 WFOs/second)
- **Processing**: 🐌 PAINFULLY SLOW (1 WFO every 40+ seconds)

#### For MCP Server Requirements
The current architecture is **NOT suitable** for real-time MCP server requests because:

1. **Processing bottleneck**: 40+ seconds per WFO is too slow for API responses
2. **No parallel processing**: Cannot utilize multiple CPU cores for processing
3. **Monolithic approach**: Must process entire WFO directory instead of single files

### Recommended Architecture for MCP Server

#### Option 1: Pre-computed Cache
- Pre-download and process all WFOs daily
- MCP server serves from pre-computed JSONL files
- Fast response times (instant)

#### Option 2: On-demand Single File Processing
- Create lightweight single-file processor
- Process only requested model/WFO combination
- Skip unnecessary calculations for speed

#### Option 3: Hybrid Approach
- Keep ultra-fast downloads (enhanced_ultra_processor.py)
- Create optimized single-file processor for MCP requests
- Pre-cache popular requests

### Commands That Work

#### Download all models for a date:
```bash
python enhanced_ultra_processor.py 2025-03-14 ./wfo_data_archive --model GFS HRRR NAM RAP --download-only
```

#### Process downloaded data (slow):
```bash
python batch_fetch_and_process.py 2025-03-14 ./wfo_data_archive --process-only
```

#### Process single WFO:
```bash
cd sounding_processor && python main.py --wfo ABQ ../wfo_data_archive/20250314/ABQ
```

### Next Steps for MCP Server

1. **Create fast single-file processor** - bypass directory processing
2. **Optimize sounding calculations** - only compute what's needed for LLM responses  
3. **Implement caching strategy** - pre-compute popular locations
4. **Add streaming responses** - return partial results while processing

The download infrastructure is production-ready. The processing layer needs complete redesign for MCP server use case.

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

## File Structure

### Main Scripts (Performance Tiers)
- `/insane_parallel_processor.py` - **NEW**: Maximum performance async processor (18x speedup)
- `/ultra_parallel_processor.py` - **NEW**: High-performance parallel processor (7x speedup)  
- `/batch_fetch_and_process.py` - **NEW**: Configurable batch processor (2x speedup)
- `/performance_comparison.py` - **NEW**: Performance analysis and recommendations
- `/fetch_single_wfo.py` - **NEW**: Single WFO downloader (used by processors)
- `/run_batch_example.py` - **NEW**: Example scripts and usage demos
- `/fetch_all_daily_data_multi_model.py` - Legacy: Sequential processor (baseline)

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