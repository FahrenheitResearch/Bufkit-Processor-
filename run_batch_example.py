#!/usr/bin/env python3
"""
Example script showing how to use the batch processor.
This demonstrates the batch downloading and processing functionality.
"""

import subprocess
import sys
from datetime import datetime, timedelta

def run_batch_example():
    """Run an example batch processing job."""
    
    # Example: Process data for yesterday
    yesterday = datetime.now() - timedelta(days=1)
    run_date = yesterday.strftime("%Y-%m-%d")
    
    print(f"Running batch processing example for date: {run_date}")
    print("="*60)
    
    # Configuration
    output_dir = "./wfo_data_archive"
    models = ["RAP", "HRRR"]
    hour = "12"
    download_batch_size = 5  # Smaller batches for testing
    process_batch_size = 10
    
    # Build command
    cmd = [
        sys.executable, "batch_fetch_and_process.py",
        run_date, output_dir,
        "--hour", hour,
        "--model"] + models + [
        "--download-batch-size", str(download_batch_size),
        "--process-batch-size", str(process_batch_size),
        "--max-download-workers", "2"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        # Run the batch processor
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("BATCH PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("BATCH PROCESSING FAILED!")
            print(f"Exit code: {result.returncode}")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n\nBatch processing interrupted by user.")
    except Exception as e:
        print(f"\nError running batch processor: {e}")

def run_download_only_example():
    """Example of running download-only mode."""
    
    yesterday = datetime.now() - timedelta(days=1)
    run_date = yesterday.strftime("%Y-%m-%d")
    
    print(f"Running DOWNLOAD-ONLY example for date: {run_date}")
    print("="*60)
    
    cmd = [
        sys.executable, "batch_fetch_and_process.py",
        run_date, "./wfo_data_archive",
        "--hour", "12",
        "--model", "RAP",
        "--download-batch-size", "3",
        "--download-only"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, text=True)
        print(f"\nDownload-only completed with exit code: {result.returncode}")
    except Exception as e:
        print(f"Error: {e}")

def run_process_only_example():
    """Example of running process-only mode."""
    
    yesterday = datetime.now() - timedelta(days=1)
    run_date = yesterday.strftime("%Y-%m-%d")
    
    print(f"Running PROCESS-ONLY example for date: {run_date}")
    print("="*60)
    
    cmd = [
        sys.executable, "batch_fetch_and_process.py",
        run_date, "./wfo_data_archive",
        "--process-batch-size", "15",
        "--process-only"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("="*60)
    
    try:
        result = subprocess.run(cmd, text=True)
        print(f"\nProcess-only completed with exit code: {result.returncode}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function with examples."""
    print("Batch Processing Examples")
    print("="*60)
    print("1. Full pipeline (download + process)")
    print("2. Download only")
    print("3. Process only")
    print("4. Quit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                run_batch_example()
                break
            elif choice == "2":
                run_download_only_example()
                break
            elif choice == "3":
                run_process_only_example()
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()