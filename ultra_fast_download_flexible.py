#!/usr/bin/env python3
"""
ULTRA FAST FLEXIBLE: Download either single WFO or all WFOs with specified model run
Supports both individual WFO downloads and full 122-WFO simultaneous downloads
"""

import requests
import os
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from urllib.parse import urljoin

# Import configuration
try:
    from sounding_processor.config import (
        WFO_LOCATIONS, AFOS_PILS_GENERIC, BUFKT_SAVE_EXTENSION, 
        BUFKT_API_ENDPOINT, AFOS_API_ENDPOINT, IEM_BASE_URL, WMO_TIMESTAMP_REGEX
    )
except ImportError:
    print("Error: Could not import configuration")
    exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_fast_flexible.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_bufkit_for_wfo(wfo_id, lat, lon, run_datetime, model, target_dir):
    """Download BUFKIT for single WFO"""
    try:
        wfo_dir = target_dir / wfo_id
        wfo_dir.mkdir(parents=True, exist_ok=True)
        
        # BUFKIT download
        bufkit_url = urljoin(IEM_BASE_URL, BUFKT_API_ENDPOINT)
        runtime_str = run_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
        params = {
            'lat': lat,
            'lon': lon,
            'model': model,
            'runtime': runtime_str,
            'fall': 1
        }
        
        filename_ts = run_datetime.strftime('%Y%m%d%H')
        save_filename = f"{wfo_id}_{model}_{filename_ts}Z{BUFKT_SAVE_EXTENSION}"
        save_path = wfo_dir / save_filename
        
        if save_path.exists():
            logger.info(f"✅ {wfo_id}: BUFKIT already exists")
            bufkit_success = True
        else:
            response = requests.get(bufkit_url, params=params, timeout=60)
            if response.status_code == 200 and len(response.text) > 100:
                with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(response.text)
                logger.info(f"✅ {wfo_id}: BUFKIT downloaded")
                bufkit_success = True
            else:
                logger.error(f"❌ {wfo_id}: BUFKIT failed ({response.status_code})")
                bufkit_success = False
        
        # AFOS products download (ALL DAY, not just specific hour)
        afos_url = urljoin(IEM_BASE_URL, AFOS_API_ENDPOINT)
        afos_count = 0
        afos_skipped = 0
        target_date = run_datetime.date()
        start_date_str = target_date.strftime('%Y-%m-%dT00:00:00Z')
        end_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
        
        # Get list of existing AFOS files to check for duplicates
        existing_afos_files = set()
        for existing_file in wfo_dir.glob('*.txt'):
            if any(existing_file.name.startswith(pil) for pil in AFOS_PILS_GENERIC):
                existing_afos_files.add(existing_file.name)
        
        for pil_generic in AFOS_PILS_GENERIC:
            pil_specific = f"{pil_generic}{wfo_id}"
            afos_params = {
                'pil': pil_specific,
                'sdate': start_date_str,
                'edate': end_date_str,
                'fmt': 'text',
                'limit': 9999
            }
            
            try:
                afos_response = requests.get(afos_url, params=afos_params, timeout=45)
                if afos_response.status_code == 200 and len(afos_response.text) > 50:
                    raw_text = afos_response.text.strip()
                    if raw_text and not raw_text.isspace() and "ERROR" not in raw_text:
                        # Split on control character (proper AFOS separator)
                        products = raw_text.split('\x01')
                        
                        for product_text in products:
                            product_text = product_text.strip()
                            if len(product_text) > 50:  # Valid product
                                # Extract timestamp from product header
                                timestamp_str = "unknown"
                                lines = product_text.split('\n', 3)
                                if len(lines) > 1:
                                    # Use proper WMO timestamp regex
                                    match = WMO_TIMESTAMP_REGEX.search(lines[1])
                                    if match:
                                        ddhhmm = match.group(1)
                                        timestamp_str = f"{target_date.strftime('%y%m')}{ddhhmm}"
                                
                                if timestamp_str == "unknown":
                                    timestamp_str = f"item{afos_count+1}"
                                
                                # Create unique filename
                                afos_filename = f"{pil_generic}_{timestamp_str}.txt"
                                
                                # Check if this base filename already exists
                                if afos_filename in existing_afos_files:
                                    # Check if content is identical
                                    existing_path = wfo_dir / afos_filename
                                    if existing_path.exists():
                                        with open(existing_path, 'r', encoding='utf-8', errors='ignore') as f:
                                            existing_content = f.read()
                                        if existing_content == product_text:
                                            afos_skipped += 1
                                            continue  # Skip identical file
                                
                                afos_path = wfo_dir / afos_filename
                                
                                # Handle duplicates with different content
                                counter = 1
                                while afos_path.exists():
                                    # Read existing file to check if content is identical
                                    with open(afos_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        existing_content = f.read()
                                    if existing_content == product_text:
                                        afos_skipped += 1
                                        break  # Skip identical file
                                    
                                    afos_filename = f"{pil_generic}_{timestamp_str}_{counter}.txt"
                                    afos_path = wfo_dir / afos_filename
                                    counter += 1
                                else:
                                    # Only write if we didn't find identical content
                                    with open(afos_path, 'w', encoding='utf-8', errors='ignore') as f:
                                        f.write(product_text)
                                    afos_count += 1
                                    existing_afos_files.add(afos_filename)
            except:
                pass  # Continue on AFOS failures
        
        if afos_count > 0 or afos_skipped > 0:
            logger.info(f"✅ {wfo_id}: {afos_count} AFOS products downloaded, {afos_skipped} skipped (already exist)")
        
        return wfo_id, bufkit_success, afos_count, afos_skipped
        
    except Exception as e:
        logger.error(f"❌ {wfo_id}: Exception - {e}")
        return wfo_id, False, 0, 0

def process_data(run_date, output_dir, model, wfo_list=None):
    """Process downloaded data"""
    if wfo_list and len(wfo_list) == 1:
        # For single WFO, use sounding processor with WFO filter
        wfo_id = wfo_list[0]
        date_folder = datetime.strptime(run_date, '%Y-%m-%d').strftime('%Y%m%d')
        
        logger.info(f"Processing {wfo_id} with sounding processor")
        
        cmd = [
            'python3', '-m', 'sounding_processor.main',
            str(output_dir),  # archive_base_dir
            date_folder,      # run_date (YYYYMMDD format)
            '--wfo', wfo_id   # Process only this WFO
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Processing completed for {wfo_id}")
            # Try to count processed files from output
            if "Successfully Processed" in result.stdout:
                logger.info("Processing summary:")
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ["Successfully Processed", "Failed Processing", "Total Files Found"]):
                        logger.info(f"  {line.strip()}")
            return True
        else:
            logger.error(f"❌ Processing failed for {wfo_id}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
            return False
    else:
        # For multiple WFOs, use batch processor
        cmd = [
            'python3', 'batch_fetch_and_process.py',
            run_date, str(output_dir),
            '--model', model,
            '--process-only'
        ]
        
        result = subprocess.run(cmd)
        return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description='Ultra-fast BUFKIT + AFOS downloader - single WFO or all WFOs'
    )
    parser.add_argument('date', help='Date in YYYY-MM-DD format')
    parser.add_argument('--hour', type=str, default='12', help='Model run hour (default: 12)')
    parser.add_argument('--model', default='HRRR', help='Model name (default: HRRR)')
    parser.add_argument('--wfo', type=str, help='Single WFO ID to download (e.g., FWD)')
    parser.add_argument('--output-dir', default='./wfo_data_archive', help='Output directory')
    parser.add_argument('--no-process', action='store_true', help='Skip processing after download')
    
    args = parser.parse_args()
    
    # Validate inputs
    try:
        run_datetime = datetime.strptime(f"{args.date} {args.hour}", '%Y-%m-%d %H')
    except ValueError:
        logger.error(f"Invalid date/hour format: {args.date} {args.hour}")
        return 1
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    date_folder = run_datetime.strftime('%Y%m%d')
    target_date_dir = output_dir / date_folder
    target_date_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which WFOs to download
    if args.wfo:
        # Single WFO mode
        wfo_id = args.wfo.upper()
        if wfo_id not in WFO_LOCATIONS:
            logger.error(f"Invalid WFO ID: {wfo_id}")
            logger.info(f"Valid WFO IDs: {', '.join(sorted(WFO_LOCATIONS.keys()))}")
            return 1
        
        wfos_to_download = {wfo_id: WFO_LOCATIONS[wfo_id]}
        logger.info(f"🎯 SINGLE WFO MODE: {wfo_id}")
    else:
        # All WFOs mode
        wfos_to_download = WFO_LOCATIONS
        logger.info(f"🚀 ALL WFO MODE: {len(wfos_to_download)} locations")
    
    logger.info(f"Date: {args.date}, Model: {args.model}, Hour: {args.hour}Z")
    logger.info(f"Output: {target_date_dir}")
    logger.info("="*60)
    
    start_time = time.time()
    successful_wfos = []
    failed_wfos = []
    total_afos = 0
    total_afos_skipped = 0
    
    # Download phase
    if len(wfos_to_download) == 1:
        # Single WFO - no threading needed
        wfo_id, (lat, lon) = list(wfos_to_download.items())[0]
        logger.info(f"Downloading {wfo_id}...")
        wfo_id, bufkit_success, afos_count, afos_skipped = download_bufkit_for_wfo(
            wfo_id, lat, lon, run_datetime, args.model, target_date_dir
        )
        if bufkit_success:
            successful_wfos.append(wfo_id)
        else:
            failed_wfos.append(wfo_id)
        total_afos = afos_count
        total_afos_skipped = afos_skipped
    else:
        # Multiple WFOs - use threading
        max_workers = min(len(wfos_to_download), 122)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_wfo = {}
            
            for wfo_id, (lat, lon) in wfos_to_download.items():
                future = executor.submit(
                    download_bufkit_for_wfo, wfo_id, lat, lon, 
                    run_datetime, args.model, target_date_dir
                )
                future_to_wfo[future] = wfo_id
            
            # Collect results
            for future in as_completed(future_to_wfo):
                wfo_id, bufkit_success, afos_count, afos_skipped = future.result()
                if bufkit_success:
                    successful_wfos.append(wfo_id)
                else:
                    failed_wfos.append(wfo_id)
                total_afos += afos_count
                total_afos_skipped += afos_skipped
    
    download_time = time.time() - start_time
    
    # Download summary
    logger.info("="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"✅ BUFKIT successful: {len(successful_wfos)}")
    if failed_wfos:
        logger.info(f"❌ BUFKIT failed: {len(failed_wfos)} - {', '.join(failed_wfos)}")
    logger.info(f"📄 AFOS products: {total_afos} new")
    if total_afos_skipped > 0:
        logger.info(f"⏭️  AFOS skipped: {total_afos_skipped} (already exist)")
    logger.info(f"⚡ Download time: {download_time:.1f} seconds")
    logger.info("="*60)
    
    # Process phase
    if successful_wfos and not args.no_process:
        logger.info("🔧 PROCESSING DATA")
        
        process_start = time.time()
        success = process_data(args.date, output_dir, args.model, successful_wfos)
        process_time = time.time() - process_start
        
        total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("🎉 PIPELINE COMPLETE")
        logger.info(f"⚡ Download: {download_time:.1f}s")
        logger.info(f"🔧 Process: {process_time:.1f}s")
        logger.info(f"🏁 Total: {total_time:.1f}s")
        logger.info("="*60)
        
        return 0 if success else 1
    elif args.no_process:
        logger.info("⏭️  Processing skipped (--no-process flag)")
        return 0
    else:
        logger.error("❌ No successful downloads - skipping processing")
        return 1

if __name__ == "__main__":
    exit(main())