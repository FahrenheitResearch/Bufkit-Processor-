#!/usr/bin/env python3
"""
Single WFO fetcher that uses the existing fetch infrastructure.
This script is called by the batch processor to download data for individual WFOs.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Import from existing modules
try:
    from sounding_processor.config import (
        BUFKT_SAVE_EXTENSION, BUFKT_API_ENDPOINT, AFOS_API_ENDPOINT,
        IEM_BASE_URL, AFOS_PILS_GENERIC, WMO_TIMESTAMP_REGEX
    )
    from fetch_all_daily_data_multi_model import (
        make_api_request, download_and_save_bufkit_for_wfo, 
        download_afos_products_for_wfo
    )
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download BUFKIT and AFOS data for a single WFO."
    )
    parser.add_argument("wfo_id", help="WFO identifier (e.g., ABQ)")
    parser.add_argument("wfo_lat", type=float, help="WFO latitude")
    parser.add_argument("wfo_lon", type=float, help="WFO longitude")
    parser.add_argument("run_date", help="Run date in YYYY-MM-DD format")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument("--hour", default="12", choices=["00", "06", "12", "18"],
                       help="Model run hour (default: 12)")
    parser.add_argument("--model", nargs='+', default=["RAP"],
                       choices=['GFS', 'HRRR', 'NAM', 'NAM4KM', 'RAP'],
                       help="Models to download (default: RAP)")
    parser.add_argument("--skip-bufkit", action="store_true",
                       help="Skip downloading BUFKIT files")
    parser.add_argument("--skip-afos", action="store_true",
                       help="Skip downloading AFOS text products")
    
    args = parser.parse_args()
    
    # Validate and parse date
    try:
        target_run_dt_naive = datetime.strptime(f"{args.run_date} {args.hour}", '%Y-%m-%d %H')
        target_run_dt_utc = target_run_dt_naive
        target_date_only = target_run_dt_utc.date()
    except ValueError:
        print("Error: Invalid date or hour format. Use YYYY-MM-DD and HH.")
        sys.exit(1)
    
    # Create WFO directory
    output_path = Path(args.output_dir)
    target_wfo_dir = output_path / args.wfo_id
    target_wfo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data for WFO {args.wfo_id}...")
    print(f"  Location: {args.wfo_lat}, {args.wfo_lon}")
    print(f"  Models: {', '.join(args.model)}")
    print(f"  Target directory: {target_wfo_dir}")
    
    # Track success/failure
    bufkit_success_count = 0
    bufkit_fail_count = 0
    afos_saved_count = 0
    overall_success = True
    
    # Download BUFKIT files
    if not args.skip_bufkit:
        print("Downloading BUFKIT files...")
        for model in args.model:
            try:
                success = download_and_save_bufkit_for_wfo(
                    args.wfo_id, args.wfo_lat, args.wfo_lon, 
                    target_run_dt_utc, model, str(target_wfo_dir)
                )
                if success:
                    bufkit_success_count += 1
                else:
                    bufkit_fail_count += 1
                    overall_success = False
            except Exception as e:
                print(f"  Error downloading BUFKIT for {model}: {e}")
                bufkit_fail_count += 1
                overall_success = False
    
    # Download AFOS products
    if not args.skip_afos:
        print("Downloading AFOS products...")
        try:
            afos_saved_count = download_afos_products_for_wfo(
                args.wfo_id, target_date_only, str(target_wfo_dir)
            )
        except Exception as e:
            print(f"  Error downloading AFOS products: {e}")
            overall_success = False
    
    # Summary
    print(f"\nDownload summary for {args.wfo_id}:")
    print(f"  BUFKIT successes: {bufkit_success_count}")
    print(f"  BUFKIT failures: {bufkit_fail_count}")
    print(f"  AFOS products saved: {afos_saved_count}")
    print(f"  Overall success: {overall_success}")
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()