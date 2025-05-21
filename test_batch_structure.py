#!/usr/bin/env python3
"""
Simple test to verify the batch processing structure works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from sounding_processor.config import WFO_LOCATIONS, BUFKT_SAVE_EXTENSION
        print(f"✓ Config imported: {len(WFO_LOCATIONS)} WFOs available")
        print(f"✓ BUFKIT extension: {BUFKT_SAVE_EXTENSION}")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from batch_fetch_and_process import BatchProcessor
        print("✓ BatchProcessor class imported")
    except ImportError as e:
        print(f"✗ BatchProcessor import failed: {e}")
        return False
    
    return True

def test_batch_processor_creation():
    """Test creating a BatchProcessor instance."""
    print("\nTesting BatchProcessor creation...")
    
    try:
        from batch_fetch_and_process import BatchProcessor
        
        processor = BatchProcessor(
            output_base_dir="./test_output",
            run_date="2023-10-26",
            run_hour="12",
            models=["RAP"],
            download_batch_size=5,
            process_batch_size=10
        )
        
        print(f"✓ BatchProcessor created for date: {processor.date_folder}")
        print(f"✓ Target directory: {processor.target_date_dir}")
        print(f"✓ Download batch size: {processor.download_batch_size}")
        print(f"✓ Process batch size: {processor.process_batch_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ BatchProcessor creation failed: {e}")
        return False

def test_wfo_chunking():
    """Test WFO chunking functionality."""
    print("\nTesting WFO chunking...")
    
    try:
        from batch_fetch_and_process import BatchProcessor
        from sounding_processor.config import WFO_LOCATIONS
        
        processor = BatchProcessor(
            output_base_dir="./test_output",
            run_date="2023-10-26"
        )
        
        # Test with first 20 WFOs
        test_wfos = list(WFO_LOCATIONS.items())[:20]
        chunks = processor.chunk_wfos(test_wfos, 5)
        
        print(f"✓ Chunked {len(test_wfos)} WFOs into {len(chunks)} chunks of size 5")
        print(f"✓ First chunk has {len(chunks[0])} WFOs: {[wfo[0] for wfo in chunks[0]]}")
        
        return True
        
    except Exception as e:
        print(f"✗ WFO chunking failed: {e}")
        return False

def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "batch_fetch_and_process.py",
        "fetch_single_wfo.py",
        "run_batch_example.py",
        "sounding_processor/main.py",
        "sounding_processor/config.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("BATCH PROCESSING STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_batch_processor_creation,
        test_wfo_chunking
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ PASSED")
            else:
                failed += 1
                print("✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED with exception: {e}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Batch processing structure is ready!")
    else:
        print("⚠️  Some tests failed - check the issues above")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)