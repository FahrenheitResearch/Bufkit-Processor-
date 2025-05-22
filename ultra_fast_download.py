#!/usr/bin/env python3
"""
ULTRA FAST: Download all 122 WFOs simultaneously + process with 16 threads
"""

import requests
import os
import time
import logging
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
        logging.FileHandler('ultra_fast.log'),
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
        
        # AFOS products download (ALL DAY, not just 12Z)
        afos_url = urljoin(IEM_BASE_URL, AFOS_API_ENDPOINT)
        afos_count = 0
        target_date = run_datetime.date()
        start_date_str = target_date.strftime('%Y-%m-%dT00:00:00Z')
        end_date_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
        
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
                                afos_path = wfo_dir / afos_filename
                                
                                # Handle duplicates
                                counter = 1
                                while afos_path.exists():
                                    afos_filename = f"{pil_generic}_{timestamp_str}_{counter}.txt"
                                    afos_path = wfo_dir / afos_filename
                                    counter += 1
                                
                                with open(afos_path, 'w', encoding='utf-8', errors='ignore') as f:
                                    f.write(product_text)
                                afos_count += 1
            except:
                pass  # Continue on AFOS failures
        
        if afos_count > 0:
            logger.info(f"✅ {wfo_id}: {afos_count} AFOS products downloaded")
        
        return wfo_id, bufkit_success, afos_count
        
    except Exception as e:
        logger.error(f"❌ {wfo_id}: Exception - {e}")
        return wfo_id, False, 0

def main():
    # Fixed parameters for your request
    run_date = "2025-02-12"
    hour = "12"
    model = "HRRR"
    output_dir = Path("./wfo_data_archive")
    
    run_datetime = datetime.strptime(f"{run_date} {hour}", '%Y-%m-%d %H')
    date_folder = run_datetime.strftime('%Y%m%d')
    target_date_dir = output_dir / date_folder
    target_date_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 ULTRA FAST DOWNLOAD STARTING")
    logger.info(f"Date: {run_date}, Model: {model}, Hour: {hour}Z")
    logger.info(f"Target: {len(WFO_LOCATIONS)} WFOs")
    logger.info("="*60)
    
    start_time = time.time()
    
    # DOWNLOAD ALL 122 WFOs SIMULTANEOUSLY
    successful_wfos = []
    failed_wfos = []
    total_afos = 0
    
    with ThreadPoolExecutor(max_workers=122) as executor:  # ALL AT ONCE!
        future_to_wfo = {}
        
        for wfo_id, (lat, lon) in WFO_LOCATIONS.items():
            future = executor.submit(download_bufkit_for_wfo, wfo_id, lat, lon, run_datetime, model, target_date_dir)
            future_to_wfo[future] = wfo_id
        
        # Collect results
        for future in as_completed(future_to_wfo):
            wfo_id, bufkit_success, afos_count = future.result()
            if bufkit_success:
                successful_wfos.append(wfo_id)
            else:
                failed_wfos.append(wfo_id)
            total_afos += afos_count
    
    download_time = time.time() - start_time
    
    logger.info("="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"✅ BUFKIT successful: {len(successful_wfos)}")
    logger.info(f"❌ BUFKIT failed: {len(failed_wfos)}")
    logger.info(f"📄 AFOS products: {total_afos}")
    logger.info(f"⚡ Download time: {download_time:.1f} seconds")
    logger.info("="*60)
    
    # PROCESS WITH 16 THREADS
    if successful_wfos:
        logger.info("🔧 PROCESSING WITH 16 THREADS")
        
        process_start = time.time()
        
        cmd = [
            'python', 'batch_fetch_and_process.py',
            run_date, str(output_dir),
            '--model', model,
            '--process-only'
        ]
        
        result = subprocess.run(cmd)
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("🎉 ULTRA FAST PIPELINE COMPLETE")
        logger.info(f"⚡ Download: {download_time:.1f}s")
        logger.info(f"🔧 Process: {process_time:.1f}s")
        logger.info(f"🏁 Total: {total_time:.1f}s")
        logger.info("="*60)
    else:
        logger.error("❌ No successful downloads - skipping processing")

if __name__ == "__main__":
    main()