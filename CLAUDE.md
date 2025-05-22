# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview - BUFKIT Weather Data Processing ⚡

This repository processes **BUFKIT atmospheric sounding data** with maximum speed and efficiency. Our goal is to **download specific model runs + all AFOS products for that day** and process them as fast as possible.

### Core Objective 🎯

**Retrieve weather model data and text products efficiently:**
- **BUFKIT files**: Specific model run (e.g., HRRR 12Z) 
- **AFOS products**: All text products for the entire day (AFD, LSR, SVR, TOR, etc.)
- **Process**: Generate verbose + LLM-optimized JSONL outputs with 16-thread parallelization

## Current System Status ✅

### Ultra-Fast Download + Processing Pipeline 🚀

**Primary Tool: `ultra_fast_download_flexible.py`** - Maximum speed with flexibility
- **Single WFO Mode**: Download just one WFO location with its BUFKIT + AFOS data
- **All WFO Mode**: Downloads ALL 122 WFOs simultaneously (122 parallel threads) 
- **BUFKIT**: Gets specific model run (e.g., HRRR 12Z)
- **AFOS**: Gets ALL products for the entire day with **duplicate detection**
  - Checks if AFOS products already exist before downloading
  - Compares content to avoid duplicate files with different names
  - Skips identical products, saving time and storage
- **Processing**: Automatically processes data after download (configurable)
- **Speed**: Single WFO in seconds, all WFOs in minimal time

**Usage Examples:**
```bash
# Download single WFO (Fort Worth) for specific date/hour
python3 ultra_fast_download_flexible.py 2025-02-12 --wfo FWD --model HRRR --hour 12

# Download single WFO without processing
python3 ultra_fast_download_flexible.py 2025-02-12 --wfo OKX --model RAP --hour 00 --no-process

# Download ALL 122 WFOs (original ultra-fast behavior)
python3 ultra_fast_download_flexible.py 2025-02-12 --model HRRR --hour 12

# See all options
python3 ultra_fast_download_flexible.py --help
```

**Legacy Tool: `ultra_fast_download.py`** - Original hardcoded version
- Fixed for HRRR 2025-02-12 12Z
- Downloads all 122 WFOs only

### Batch Processing System

**Secondary Tool: `batch_fetch_and_process.py`** - Configurable approach
- **Threading**: Now defaults to 16 processing threads (increased from 4)
- **Download**: Batched downloads (10 WFOs per batch, 3 concurrent batches)
- **Processing**: 16-thread parallel processing of BUFKIT files
- **Flexible**: Supports multiple models, dates, and configurations

**Key Commands:**
```bash
# Process existing data with 16 threads (default)
python batch_fetch_and_process.py 2025-02-12 ./wfo_data_archive --model HRRR --process-only

# Download + process complete pipeline
python batch_fetch_and_process.py 2025-02-12 ./wfo_data_archive --model HRRR --hour 12

# Customize thread count if needed
python batch_fetch_and_process.py 2025-02-12 ./wfo_data_archive --model HRRR --max-process-workers 32
```

### Core Processing Engine ✅

**sounding_processor/** package - Rock solid foundation
- **Parser**: Robust BUFKIT text file parsing
- **Calculator**: MetPy-based meteorological calculations (CAPE, CIN, wind shear, etc.)
- **Output**: Dual format - verbose JSONL + LLM-optimized JSONL
- **Thread-Safe**: Works perfectly with ThreadPoolExecutor

## File Structure

```
/home/ubuntu2/claude-bufkit/
├── ultra_fast_download_flexible.py     # 🚀 PRIMARY: Flexible ultra-fast downloader
├── ultra_fast_download.py              # 🚀 LEGACY: 122 simultaneous downloads (hardcoded)
├── batch_fetch_and_process.py          # ⚙️  Configurable batch processing (16 threads)
├── fetch_all_daily_data_multi_model.py # 📥 Original stable downloader
├── sounding_processor/                 # 🧠 Core processing engine
│   ├── main.py                         # Entry point for individual processing
│   ├── bufkit_parser.py               # BUFKIT format parser
│   ├── sounding_calculator.py         # MetPy calculations
│   ├── sounding_data.py               # Data structures
│   ├── config.py                      # WFO locations & settings
│   └── convert_to_llm_optimized_jsonl.py # Output formatting
├── wfo_data_archive/                   # 📁 Processed data storage
└── bufkit-env/                        # 🐍 Python virtual environment
```

## Data Flow

1. **Download**: BUFKIT model files + AFOS text products → `wfo_data_archive/YYYYMMDD/WFO/`
2. **Parse**: BUFKIT text → structured atmospheric sounding data
3. **Calculate**: MetPy → derived parameters (CAPE, CIN, wind shear, storm motion, etc.)
4. **Output**: Generate verbose + LLM-optimized JSONL files for AI consumption

## Threading Architecture ⚡

### Current Implementation - PROVEN FAST ✅

**ThreadPoolExecutor** approach successfully optimized:
- **Download**: 122 simultaneous threads (all WFOs at once)
- **Processing**: 16 threads (default, configurable)
- **Why Threading Works**: I/O bound operations benefit despite Python GIL
- **MetPy Compatible**: No pickle serialization issues with threading
- **Thread Safety**: Proper locking for shared statistics

### Performance Characteristics
- **Download Speed**: All 122 WFOs downloaded simultaneously
- **Processing Speed**: 16 threads handle BUFKIT parsing + MetPy calculations
- **Reliability**: Graceful handling of network failures and processing errors
- **Scalability**: Can handle full CONUS (122 WFOs) efficiently

## AFOS Product Handling 📄

### Corrected Implementation ✅
- **Time Range**: Downloads ALL products for the entire day (00Z-23Z)
- **Proper Parsing**: Uses `\x01` control character separator (not line breaks)
- **File Structure**: Each AFOS product saved as one complete file
- **Naming**: Timestamped filenames (e.g., `AFD_2502121435.txt`)
- **Product Types**: AFD, LSR, SPS, SVR, TOR, FFW, WSW

### Intelligent Duplicate Detection 🔍
- **Content-Based Checking**: Compares actual file content, not just filenames
- **Efficient Storage**: Prevents duplicate AFOS products when downloading different model runs
- **Smart Skipping**: Reports how many products were skipped vs downloaded
- **Use Case**: Run HRRR 12Z, then HRRR 13Z, then RAP 12Z - AFOS products download only once
- **Performance**: Subsequent runs for same date are much faster (3s vs 30s+)

## Future Development Roadmap 🛣️

### Primary Goal: Enhanced Ultra-Fast Pipeline
- **Make `ultra_fast_download.py` configurable** (date, model, hour parameters)
- **Integrate with main workflow** as primary download method
- **Add intelligent caching** (skip already downloaded data)
- **Extend to multiple models** simultaneously

### Secondary Enhancements
- **MCP Server Integration**: Expose processed data via Model Context Protocol
- **Real-time Processing**: Monitor for new model runs and auto-process
- **Data Validation**: Enhanced QC checks for downloaded BUFKIT files

## Architecture Guidelines

### DO Use ✅
- **ultra_fast_download.py** for maximum speed downloads
- **ThreadPoolExecutor** for all parallel processing
- **16+ processing threads** for optimal BUFKIT processing
- **Existing sounding_processor/** package (battle-tested)
- **Current AFOS parsing logic** (properly handles control characters)

### DON'T Use ❌
- **ProcessPoolExecutor** or multiprocessing (MetPy pickle issues)
- **GPU acceleration** (unnecessary complexity for this workload)
- **Async/await** for MetPy calculations (synchronous library)
- **Line-break parsing** for AFOS products (incorrect separator)

### Virtual Environment
- **Location**: `bufkit-env/`
- **Key Dependencies**: requests, pandas, numpy, MetPy 1.7.0
- **Installation**: `pip install -r requirements-mcp.txt`

## Data Formats

### Input
- **BUFKIT**: Text-based atmospheric profile data from IEM
- **AFOS**: NWS text products (AFD, LSR, warnings, watches, etc.)

### Output
- **Raw**: `WFO_MODEL_YYYYMMDDHH.buf.txt` (original BUFKIT)
- **Verbose**: `WFO_MODEL_YYYYMMDDHH_default.jsonl` (full meteorological data)
- **LLM-Optimized**: `WFO_MODEL_YYYYMMDDHH_default_llm_optimized.jsonl` (AI-friendly format)
- **AFOS**: `PIL_YYMMDDHHMM.txt` (individual text products)

This system is **production-ready** and optimized for **maximum speed** while maintaining **data integrity** and **reliability**.