#!/bin/bash

echo "=== Testing Ultra Fast Download - Single WFO Mode ==="
echo

# Example 1: Download single WFO (FWD - Fort Worth) for a specific date and hour
echo "Example 1: Download FWD (Fort Worth) for 2025-02-12 12Z HRRR"
echo "Command: python3 ultra_fast_download_flexible.py 2025-02-12 --wfo FWD --model HRRR --hour 12"
echo

# Example 2: Download single WFO without processing
echo "Example 2: Download OKX (New York) for 2025-02-12 00Z RAP (no processing)"
echo "Command: python3 ultra_fast_download_flexible.py 2025-02-12 --wfo OKX --model RAP --hour 00 --no-process"
echo

# Example 3: Download all WFOs (original behavior)
echo "Example 3: Download ALL WFOs for 2025-02-12 12Z HRRR"
echo "Command: python3 ultra_fast_download_flexible.py 2025-02-12 --model HRRR --hour 12"
echo

echo "Note: Each download includes:"
echo "  - BUFKIT file for the specific model run (e.g., HRRR 12Z)"
echo "  - ALL AFOS text products for the entire day"
echo "  - Automatic processing to generate JSONL files (unless --no-process is used)"