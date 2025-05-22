# Ultra-Fast BUFKIT Weather Data System 🚀

A high-performance system for downloading and processing BUFKIT atmospheric sounding data and AFOS text products from the Iowa Environmental Mesonet (IEM).

## Quick Start

### Single WFO Download
```bash
# Download and process Fort Worth (FWD) HRRR 12Z data
python3 ultra_fast_download_flexible.py 2025-02-12 --wfo FWD --model HRRR --hour 12

# Download without processing
python3 ultra_fast_download_flexible.py 2025-02-12 --wfo OKX --model RAP --hour 00 --no-process
```

### All WFOs Download (122 locations)
```bash
# Download all WFOs simultaneously
python3 ultra_fast_download_flexible.py 2025-02-12 --model HRRR --hour 12
```

## Features

- **Ultra-fast downloads**: 122 simultaneous threads for all WFOs
- **Intelligent duplicate detection**: Prevents redundant AFOS downloads
- **Multiple models**: HRRR, RAP, NAM, NAM4KM, GFS
- **Flexible hours**: 00Z, 06Z, 12Z, 18Z
- **Automatic processing**: Generates JSONL files with meteorological calculations

## Options

```
positional arguments:
  date                  Date in YYYY-MM-DD format

options:
  --hour HOUR           Model run hour (default: 12)
  --model MODEL         Model name (default: HRRR)
  --wfo WFO             Single WFO ID to download (e.g., FWD)
  --output-dir OUTPUT_DIR  Output directory (default: ./wfo_data_archive)
  --no-process          Skip processing after download
```

## Output Structure

```
wfo_data_archive/
└── YYYYMMDD/
    └── WFO/
        ├── WFO_MODEL_YYYYMMDDHH.buf.txt              # Raw BUFKIT
        ├── WFO_MODEL_YYYYMMDDHH_default.jsonl        # Verbose output
        ├── WFO_MODEL_YYYYMMDDHH_llm_optimized.jsonl  # AI-friendly
        └── AFOS_*.txt                                 # Text products
```

## Performance

- Single WFO: ~3 seconds (download + process)
- All 122 WFOs: ~30-60 seconds depending on network
- Duplicate AFOS detection saves 90%+ time on repeat runs

## System Requirements

- Python 3.8+
- MetPy 1.7.0
- 16+ GB RAM recommended for processing all WFOs
- Fast internet connection

See `CLAUDE.md` for detailed technical documentation.