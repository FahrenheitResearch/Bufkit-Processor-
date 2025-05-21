# sounding_processor/main.py
"""
Main entry point for batch processing BUFKIT Sounding files
found within the WFO-organized archive structure.
"""
import sys
import argparse
import json
import logging
from pathlib import Path
import time
from typing import List, Dict # For type hinting
import traceback # For more detailed error logging in batch mode

# --- Project-local imports ---
# Attempt import, provide guidance if fails
try:
    from . import config as app_config
    from .utils import setup_logging, convert_numpy_types_for_json
    from .metpy_manager import metpy_manager_instance
    from .bufkit_parser import BufkitParser
    from .sounding_calculator import SoundingCalculator
    from .sounding_data import Sounding
    from .convert_to_llm_optimized_jsonl import process_file as convert_to_llm_format
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure this script is run correctly relative to the 'sounding_processor' package.")
    print("You might need to install the package (`pip install .`) or run using `python -m sounding_processor.main ...`")
    sys.exit(1)

# Ensure BUFKT_SAVE_EXTENSION is defined in config, default if missing
BUFKT_INPUT_EXTENSION = getattr(app_config, 'BUFKT_SAVE_EXTENSION', '.buf.txt')

module_logger = logging.getLogger(__name__)


def run_processing_single_file(input_file_path: Path, output_file_path: Path):
    """
    Runs the full sounding processing workflow for a SINGLE input file.
    Saves verbose JSONL and converts to LLM-optimized format.

    Args:
        input_file_path: Path to the input BUFKIT file (.buf.txt).
        output_file_path: Path for the primary output JSONL file (e.g., _default.jsonl).
                          The LLM version will be derived from this.
    Returns:
        bool: True if processing was successful (or file already processed), False otherwise.
    """
    processing_start_time = time.time()
    # Derive LLM output path now to check if BOTH outputs already exist
    llm_output_filename_str = output_file_path.stem + "_llm_optimized" + output_file_path.suffix
    llm_output_file_path = output_file_path.parent / llm_output_filename_str

    # --- Check if BOTH output files already exist ---
    if output_file_path.exists() and llm_output_file_path.exists():
        module_logger.debug(f"Outputs already exist, skipping: {input_file_path.name}")
        return True # Treat as success if already processed

    module_logger.info(f"Processing: {input_file_path.name}")

    if not metpy_manager_instance.metpy_available:
        module_logger.warning("MetPy not available. Processing continuing with limited capabilities.")
        # Decide if this should be a critical failure? For now, allow proceeding.

    # Step 1: Read BUFKIT file content
    try:
        if not input_file_path.exists():
            module_logger.error(f"Input file not found: {input_file_path}")
            return False # File disappeared?

        bufkit_content = input_file_path.read_text(encoding='latin-1') # Use latin-1 for robustness
        if not bufkit_content.strip():
            module_logger.error(f"Input file is empty: {input_file_path}")
            return False
    except Exception as e_read:
        module_logger.error(f"Error reading input file {input_file_path}: {e_read}", exc_info=False) # Less verbose log
        return False

    # Step 2: Parse BUFKIT data
    parser = BufkitParser()
    try:
        parsed_soundings_list: List[Sounding] = parser.parse(bufkit_content)
        if not parsed_soundings_list:
            module_logger.warning(f"No soundings parsed from: {input_file_path.name}")
            return False # Treat as failure if no soundings
    except Exception as e_parse:
        module_logger.error(f"Parsing error in {input_file_path.name}: {e_parse}", exc_info=False)
        return False

    # Step 3: Calculate derived parameters
    calculator = SoundingCalculator()
    processed_soundings_for_output: List[Dict] = []
    calculation_errors = False

    for i, snd_obj in enumerate(parsed_soundings_list):
        sounding_id_str = f"{snd_obj.metadata.get('STID', 'N/A')} {snd_obj.metadata.get('TIME', 'N/A')}"
        try:
            calculator.calculate_all_derived_params(snd_obj)
            sounding_dict_for_json = snd_obj.to_dict(serialize_df=False) # Usually False for JSONL
            cleaned_sounding_dict = convert_numpy_types_for_json(sounding_dict_for_json)
            processed_soundings_for_output.append(cleaned_sounding_dict)
        except Exception as e_calc:
            calculation_errors = True
            module_logger.error(f"Calc error for {sounding_id_str} in {input_file_path.name}: {e_calc}", exc_info=False)
            # Optionally include error placeholder in output? For now, just log.
            # If one hour fails, should the whole file fail? Let's allow partial output.

    if not processed_soundings_for_output:
        module_logger.error(f"No soundings successfully calculated in {input_file_path.name}.")
        return False # Failed if nothing could be calculated

    # Step 4: Save processed soundings (verbose/default format)
    verbose_saved = False
    try:
        # Ensure parent directory exists (should, but good practice)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with output_file_path.open('w', encoding='utf-8') as outfile:
            for snd_data_out in processed_soundings_for_output:
                json.dump(snd_data_out, outfile, ensure_ascii=False)
                outfile.write('\n')
        module_logger.debug(f"Saved verbose: {output_file_path.name}")
        verbose_saved = True
    except Exception as e_json:
        module_logger.error(f"Error saving verbose JSONL {output_file_path.name}: {e_json}", exc_info=False)
        # Attempt to clean up partially written file? Maybe risky.
        return False # Fail if we can't save the primary output

    # Step 5: Convert to LLM-optimized format (only if verbose save succeeded)
    llm_saved = False
    if verbose_saved:
        try:
            # Pass the path of the just-written verbose file as input
            convert_to_llm_format(output_file_path, llm_output_file_path)
            module_logger.debug(f"Saved LLM Opt: {llm_output_file_path.name}")
            llm_saved = True
        except Exception as e_llm_convert:
            module_logger.error(f"Error during LLM conversion for {output_file_path.name}: {e_llm_convert}", exc_info=False)
            # Don't necessarily return False here, primary output was saved.

    processing_duration = time.time() - processing_start_time
    module_logger.info(f"Finished processing {input_file_path.name} in {processing_duration:.2f}s "
                       f"(Verbose Saved: {verbose_saved}, LLM Saved: {llm_saved}, Calc Errors: {calculation_errors})")

    # Define overall success: verbose must save, LLM save is bonus. Calc errors are logged but don't fail the file.
    return verbose_saved


def process_directory(target_date_dir: Path):
    """
    Finds and processes all BUFKIT files within a specific date directory.
    """
    module_logger.info(f"--- Starting Batch Processing for Directory: {target_date_dir} ---")
    module_logger.info(f"Searching for files ending with: '{BUFKT_INPUT_EXTENSION}'")

    if not target_date_dir.is_dir():
        module_logger.error(f"Target directory does not exist: {target_date_dir}")
        return

    # Find all input files recursively within the date directory
    input_files = list(target_date_dir.rglob(f"*{BUFKT_INPUT_EXTENSION}"))

    if not input_files:
        module_logger.warning(f"No '{BUFKT_INPUT_EXTENSION}' files found in {target_date_dir} or its subdirectories.")
        return

    module_logger.info(f"Found {len(input_files)} potential input files to process.")

    # Process each file
    overall_start_time = time.time()
    success_count = 0
    fail_count = 0
    skip_count = 0 # Counted based on return value of run_processing_single_file

    for i, input_fpath in enumerate(input_files):
        module_logger.info(f"--- File {i+1}/{len(input_files)} ---")
        try:
            # Determine output path based on input path
            output_dir = input_fpath.parent
            output_base_name = input_fpath.stem # Filename without extension
            primary_output_filename = f"{output_base_name}_default.jsonl"
            primary_output_fpath = output_dir / primary_output_filename

            success = run_processing_single_file(input_fpath, primary_output_fpath)
            if success:
                # Check if the files actually existed before processing call
                # The function now handles this check internally and returns True if skipped
                if not primary_output_fpath.exists(): # Should not happen if success is True unless skipped
                     module_logger.warning(f"Processing reported success but output missing for {input_fpath.name}")
                     fail_count +=1 # Count as failure if output is missing despite success report
                else:
                    # This check is a bit redundant now as run_processing... handles it
                    # We rely on its return value: True means success or skip due to existing files
                    success_count += 1
            else:
                fail_count += 1

        except Exception as e_main_loop:
            # Catch unexpected errors in the main loop itself
            fail_count += 1
            module_logger.critical(f"UNEXPECTED MAIN LOOP ERROR processing {input_fpath.name}: {e_main_loop}")
            module_logger.critical(traceback.format_exc()) # Print full traceback for critical errors

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    module_logger.info("--- Batch Processing Summary ---")
    module_logger.info(f"Directory Processed: {target_date_dir}")
    module_logger.info(f"Total Files Found: {len(input_files)}")
    module_logger.info(f"Successfully Processed (or Skipped): {success_count}")
    module_logger.info(f"Failed Processing: {fail_count}")
    module_logger.info(f"Total Duration: {total_duration:.2f} seconds")
    module_logger.info("-------------------------------")


def main_cli():
    project_root = Path(__file__).resolve().parent.parent

    # Default log dir setup - consider making this configurable
    log_dir_name = getattr(app_config, 'LOG_DIRECTORY_NAME', 'debug_logs')
    log_filename = getattr(app_config, 'LOG_FILENAME', 'sounding_processor_batch.log') # Different log name
    log_dir = project_root / "data" / "output" / log_dir_name # Place logs relative to project root
    log_file_path = log_dir / log_filename

    setup_logging(log_file_path=log_file_path,
                  file_level=getattr(app_config, 'LOG_LEVEL', logging.INFO), # Default to INFO
                  console_level=getattr(app_config, 'CONSOLE_LOG_LEVEL', logging.INFO)) # Default to INFO

    module_logger.info("--- Sounding Processor Batch Run Start ---")
    module_logger.debug(f"Project Root (derived): {project_root}")
    module_logger.debug(f"Log file path: {log_file_path}")

    parser = argparse.ArgumentParser(
        description=f"Batch process BUFKIT ({BUFKT_INPUT_EXTENSION}) files found in a WFO-organized archive date directory."
    )
    parser.add_argument(
        "archive_base_dir",
        help="The base directory of the WFO-organized archive (e.g., ./wfo_organized_archive)."
    )
    parser.add_argument(
        "run_date",
        help="The specific run date subfolder (YYYYMMDD) within the archive base directory to process."
    )
    parser.add_argument(
        "--wfo",
        help="Process only this specific WFO ID (e.g., ABQ). If not specified, process all WFOs in the date directory."
    )
    # Removed single input/output file args, verification args might be less useful here

    args = parser.parse_args()

    # Construct the target directory path
    try:
        # Basic validation for date format YYYYMMDD
        if len(args.run_date) != 8 or not args.run_date.isdigit():
            raise ValueError("Date folder name must be YYYYMMDD format.")
        target_date_dir_path = Path(args.archive_base_dir).resolve() / args.run_date
    except ValueError as e:
        module_logger.error(f"Invalid run_date argument: {e}")
        sys.exit(1)
    except Exception as e_path:
         module_logger.error(f"Error constructing target path: {e_path}")
         sys.exit(1)

    # Handle single WFO processing
    if args.wfo:
        single_wfo_dir = target_date_dir_path / args.wfo
        if not single_wfo_dir.exists():
            module_logger.error(f"WFO directory not found: {single_wfo_dir}")
            sys.exit(1)
        module_logger.info(f"Processing single WFO: {args.wfo}")
        process_directory(single_wfo_dir)
    else:
        process_directory(target_date_dir_path)

    module_logger.info("--- Sounding Processor Batch Run End ---")

if __name__ == "__main__":
    main_cli()