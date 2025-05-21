#!/usr/bin/env python3
"""
Batch processing script that downloads WFO data in batches of 10 
and processes them in batches of 25.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import configuration
try:
    from sounding_processor.config import WFO_LOCATIONS, BUFKT_SAVE_EXTENSION
except ImportError:
    print("Error: Could not import configuration from sounding_processor.config")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Manages batch downloading and processing of WFO data."""
    
    def __init__(self, output_base_dir: str, run_date: str, run_hour: str = "12", 
                 models: List[str] = None, download_batch_size: int = 10, 
                 process_batch_size: int = 25, max_download_workers: int = 3):
        self.output_base_dir = Path(output_base_dir)
        self.run_date = run_date
        self.run_hour = run_hour
        self.models = models or ["RAP"]
        self.download_batch_size = download_batch_size
        self.process_batch_size = process_batch_size
        self.max_download_workers = max_download_workers
        
        # Create date directory
        self.date_folder = datetime.strptime(run_date, '%Y-%m-%d').strftime('%Y%m%d')
        self.target_date_dir = self.output_base_dir / self.date_folder
        self.target_date_dir.mkdir(parents=True, exist_ok=True)
        
        # Track progress
        self.download_stats = {
            'completed_wfos': set(),
            'failed_wfos': set(),
            'total_bufkit_success': 0,
            'total_bufkit_fail': 0,
            'total_afos_saved': 0
        }
        self.processing_stats = {
            'completed_wfos': set(),
            'failed_wfos': set(),
            'total_processed': 0
        }
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
    def chunk_wfos(self, wfo_list: List[Tuple[str, Tuple[float, float]]], chunk_size: int) -> List[List[Tuple[str, Tuple[float, float]]]]:
        """Split WFO list into chunks of specified size."""
        chunks = []
        for i in range(0, len(wfo_list), chunk_size):
            chunks.append(wfo_list[i:i + chunk_size])
        return chunks
    
    def download_wfo_batch(self, wfo_batch: List[Tuple[str, Tuple[float, float]]], 
                          batch_num: int, total_batches: int) -> Dict:
        """Download data for a batch of WFOs."""
        logger.info(f"Starting download batch {batch_num}/{total_batches} with {len(wfo_batch)} WFOs")
        
        batch_stats = {
            'completed': [],
            'failed': [],
            'bufkit_success': 0,
            'bufkit_fail': 0,
            'afos_saved': 0
        }
        
        for wfo_id, (wfo_lat, wfo_lon) in wfo_batch:
            try:
                wfo_success, bufkit_success, bufkit_fail, afos_saved = self._download_single_wfo(
                    wfo_id, wfo_lat, wfo_lon
                )
                
                if wfo_success:
                    batch_stats['completed'].append(wfo_id)
                    batch_stats['bufkit_success'] += bufkit_success
                    batch_stats['bufkit_fail'] += bufkit_fail
                    batch_stats['afos_saved'] += afos_saved
                else:
                    batch_stats['failed'].append(wfo_id)
                    
            except Exception as e:
                logger.error(f"Error downloading WFO {wfo_id}: {e}")
                batch_stats['failed'].append(wfo_id)
        
        logger.info(f"Completed download batch {batch_num}/{total_batches}: "
                   f"{len(batch_stats['completed'])} successful, {len(batch_stats['failed'])} failed")
        
        return batch_stats
    
    def _download_single_wfo(self, wfo_id: str, wfo_lat: float, wfo_lon: float) -> Tuple[bool, int, int, int]:
        """Download data for a single WFO using the existing fetch script."""
        logger.info(f"Downloading data for WFO: {wfo_id}")
        
        # Use the existing fetch script as a subprocess
        cmd = [
            sys.executable, "fetch_single_wfo.py",
            wfo_id, str(wfo_lat), str(wfo_lon),
            self.run_date, str(self.target_date_dir),
            "--hour", self.run_hour,
            "--model"
        ] + self.models
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse output for basic statistics
                bufkit_success = len(self.models)  # Assume all models succeeded
                return True, bufkit_success, 0, 1  # Success, bufkit_success, bufkit_fail, afos_saved
            else:
                logger.error(f"Download failed for {wfo_id}: {result.stderr}")
                return False, 0, len(self.models), 0
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for {wfo_id}")
            return False, 0, len(self.models), 0
        except Exception as e:
            logger.error(f"Download error for {wfo_id}: {e}")
            return False, 0, len(self.models), 0
    
    
    def process_wfo_batch(self, wfo_batch: List[str], batch_num: int, total_batches: int) -> Dict:
        """Process a batch of WFO directories."""
        logger.info(f"Starting processing batch {batch_num}/{total_batches} with {len(wfo_batch)} WFOs")
        
        batch_stats = {
            'completed': [],
            'failed': [],
            'total_processed': 0
        }
        
        for wfo_id in wfo_batch:
            try:
                wfo_dir = self.target_date_dir / wfo_id
                if wfo_dir.exists():
                    success, processed_count = self._process_single_wfo(wfo_id, wfo_dir)
                    if success:
                        batch_stats['completed'].append(wfo_id)
                        batch_stats['total_processed'] += processed_count
                    else:
                        batch_stats['failed'].append(wfo_id)
                else:
                    logger.warning(f"WFO directory not found: {wfo_dir}")
                    batch_stats['failed'].append(wfo_id)
                    
            except Exception as e:
                logger.error(f"Error processing WFO {wfo_id}: {e}")
                batch_stats['failed'].append(wfo_id)
        
        logger.info(f"Completed processing batch {batch_num}/{total_batches}: "
                   f"{len(batch_stats['completed'])} successful, {len(batch_stats['failed'])} failed")
        
        return batch_stats
    
    def _process_single_wfo(self, wfo_id: str, wfo_dir: Path) -> Tuple[bool, int]:
        """Process BUFKIT files for a single WFO."""
        logger.info(f"Processing WFO: {wfo_id}")
        
        # Count BUFKIT files before processing
        bufkit_files_before = len(list(wfo_dir.glob(f"*{BUFKT_SAVE_EXTENSION}")))
        if bufkit_files_before == 0:
            logger.warning(f"No BUFKIT files found in {wfo_dir}")
            return True, 0  # Not an error, just nothing to process
        
        # Use the existing sounding processor with single WFO mode
        cmd = [
            sys.executable, "-m", "sounding_processor.main",
            str(self.target_date_dir.parent), self.date_folder,
            "--wfo", wfo_id
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Count processed files (JSONL outputs)
                processed_files = len(list(wfo_dir.glob("*_default.jsonl")))
                logger.info(f"Processing completed for {wfo_id}: {processed_files} files processed")
                return True, processed_files
            else:
                logger.error(f"Processing failed for {wfo_id}: {result.stderr}")
                return False, 0
                
        except subprocess.TimeoutExpired:
            logger.error(f"Processing timeout for {wfo_id}")
            return False, 0
        except Exception as e:
            logger.error(f"Processing error for {wfo_id}: {e}")
            return False, 0
    
    def run_batch_download(self) -> bool:
        """Run the batch download process."""
        logger.info("Starting batch download process")
        
        # Get all WFOs as list of tuples
        wfo_items = list(WFO_LOCATIONS.items())
        logger.info(f"Total WFOs to download: {len(wfo_items)}")
        
        # Create batches
        download_batches = self.chunk_wfos(wfo_items, self.download_batch_size)
        logger.info(f"Created {len(download_batches)} download batches of size {self.download_batch_size}")
        
        # Process batches with thread pool
        with ThreadPoolExecutor(max_workers=self.max_download_workers) as executor:
            future_to_batch = {
                executor.submit(self.download_wfo_batch, batch, i+1, len(download_batches)): i
                for i, batch in enumerate(download_batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_stats = future.result()
                    
                    # Update global stats thread-safely
                    with self.stats_lock:
                        self.download_stats['completed_wfos'].update(batch_stats['completed'])
                        self.download_stats['failed_wfos'].update(batch_stats['failed'])
                        self.download_stats['total_bufkit_success'] += batch_stats['bufkit_success']
                        self.download_stats['total_bufkit_fail'] += batch_stats['bufkit_fail']
                        self.download_stats['total_afos_saved'] += batch_stats['afos_saved']
                        
                except Exception as e:
                    logger.error(f"Download batch {batch_idx + 1} failed: {e}")
        
        # Log final download statistics
        logger.info("="*60)
        logger.info("DOWNLOAD PHASE COMPLETE")
        logger.info(f"WFOs completed: {len(self.download_stats['completed_wfos'])}")
        logger.info(f"WFOs failed: {len(self.download_stats['failed_wfos'])}")
        logger.info(f"BUFKIT successes: {self.download_stats['total_bufkit_success']}")
        logger.info(f"BUFKIT failures: {self.download_stats['total_bufkit_fail']}")
        logger.info(f"AFOS products saved: {self.download_stats['total_afos_saved']}")
        logger.info("="*60)
        
        return len(self.download_stats['failed_wfos']) == 0
    
    def run_batch_processing(self) -> bool:
        """Run the batch processing process."""
        logger.info("Starting batch processing process")
        
        # Get list of WFO directories that were successfully downloaded
        available_wfos = [
            d.name for d in self.target_date_dir.iterdir() 
            if d.is_dir() and d.name in self.download_stats['completed_wfos']
        ]
        
        logger.info(f"WFOs available for processing: {len(available_wfos)}")
        
        if not available_wfos:
            logger.warning("No WFO directories available for processing")
            return False
        
        # Create processing batches
        process_batches = [
            available_wfos[i:i + self.process_batch_size]
            for i in range(0, len(available_wfos), self.process_batch_size)
        ]
        
        logger.info(f"Created {len(process_batches)} processing batches of size {self.process_batch_size}")
        
        # Process batches sequentially (to avoid overwhelming the system)
        for i, batch in enumerate(process_batches):
            try:
                batch_stats = self.process_wfo_batch(batch, i+1, len(process_batches))
                
                # Update global stats
                self.processing_stats['completed_wfos'].update(batch_stats['completed'])
                self.processing_stats['failed_wfos'].update(batch_stats['failed'])
                self.processing_stats['total_processed'] += batch_stats['total_processed']
                
                # Brief pause between batches
                if i < len(process_batches) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Processing batch {i + 1} failed: {e}")
        
        # Log final processing statistics
        logger.info("="*60)
        logger.info("PROCESSING PHASE COMPLETE")
        logger.info(f"WFOs processed: {len(self.processing_stats['completed_wfos'])}")
        logger.info(f"WFOs failed: {len(self.processing_stats['failed_wfos'])}")
        logger.info(f"Total files processed: {self.processing_stats['total_processed']}")
        logger.info("="*60)
        
        return len(self.processing_stats['failed_wfos']) == 0
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete download and processing pipeline."""
        logger.info("Starting complete batch pipeline")
        start_time = time.time()
        
        # Phase 1: Download
        download_success = self.run_batch_download()
        
        # Phase 2: Process
        processing_success = self.run_batch_processing()
        
        # Final summary
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("COMPLETE PIPELINE SUMMARY")
        logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Download success: {download_success}")
        logger.info(f"Processing success: {processing_success}")
        logger.info(f"Total WFOs in system: {len(WFO_LOCATIONS)}")
        logger.info(f"WFOs downloaded: {len(self.download_stats['completed_wfos'])}")
        logger.info(f"WFOs processed: {len(self.processing_stats['completed_wfos'])}")
        logger.info("="*80)
        
        return download_success and processing_success


def main():
    parser = argparse.ArgumentParser(
        description="Batch download and process WFO data with configurable batch sizes."
    )
    parser.add_argument("run_date", help="Run date in YYYY-MM-DD format")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument("--hour", default="12", choices=["00", "06", "12", "18"],
                       help="Model run hour (default: 12)")
    parser.add_argument("--model", nargs='+', default=["RAP"],
                       choices=['GFS', 'HRRR', 'NAM', 'NAM4KM', 'RAP'],
                       help="Models to download (default: RAP)")
    parser.add_argument("--download-batch-size", type=int, default=10,
                       help="Number of WFOs to download simultaneously (default: 10)")
    parser.add_argument("--process-batch-size", type=int, default=25,
                       help="Number of WFOs to process in each batch (default: 25)")
    parser.add_argument("--max-download-workers", type=int, default=3,
                       help="Maximum concurrent download batches (default: 3)")
    parser.add_argument("--download-only", action="store_true",
                       help="Only download, skip processing")
    parser.add_argument("--process-only", action="store_true",
                       help="Only process existing downloads, skip downloading")
    
    args = parser.parse_args()
    
    # Validate date
    try:
        datetime.strptime(args.run_date, '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Create batch processor
    processor = BatchProcessor(
        output_base_dir=args.output_dir,
        run_date=args.run_date,
        run_hour=args.hour,
        models=args.model,
        download_batch_size=args.download_batch_size,
        process_batch_size=args.process_batch_size,
        max_download_workers=args.max_download_workers
    )
    
    # Run requested operations
    if args.download_only:
        success = processor.run_batch_download()
    elif args.process_only:
        success = processor.run_batch_processing()
    else:
        success = processor.run_complete_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()