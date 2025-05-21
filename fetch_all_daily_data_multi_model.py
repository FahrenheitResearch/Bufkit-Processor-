# fetch_all_daily_data_multi_model.py

import requests
import os
import time
import argparse
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin
import sys

# --- Attempt to import config ---
try:
    # Assumes this script is run from the project root directory
    from sounding_processor.config import (
        WFO_NAMES, WFO_LOCATIONS, AFOS_PILS_GENERIC, DEFAULT_MODEL,
        BUFKT_SAVE_EXTENSION, BUFKT_API_ENDPOINT, AFOS_API_ENDPOINT,
        IEM_BASE_URL, WMO_TIMESTAMP_REGEX
    )
except ImportError:
    print("Error: Could not import configuration from sounding_processor.config")
    # ... (rest of error message) ...
    sys.exit(1)

# --- Helper: Robust API Request ---
def make_api_request(url, params=None, timeout=60, retries=2, delay=5):
    """Makes an API request with basic retry logic."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Python Automated Script)'}
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            return response # Return response regardless of status for handling later
        except requests.exceptions.Timeout:
            print(f" ! Timeout ({timeout}s) connecting to {url}. Retrying..." if attempt < retries else " ! Timeout. Giving up.")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f" ! Request Error for {url}: {e}. Retrying..." if attempt < retries else f" ! Request Error. Giving up: {e}")
            time.sleep(delay)
    return None # Return None after all retries fail


# --- BUFKIT Download Function (No changes needed here) ---
def download_and_save_bufkit_for_wfo(wfo_id, wfo_lat, wfo_lon, run_datetime, model, target_wfo_dir):
    """
    Downloads the closest BUFKIT profile for a WFO/location for a specific run/model
    and saves it. Returns True if successful, False otherwise.
    """
    bufkit_url = urljoin(IEM_BASE_URL, BUFKT_API_ENDPOINT)
    runtime_str = run_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
    params = {
        'lat': wfo_lat,
        'lon': wfo_lon,
        'model': model,
        'runtime': runtime_str,
        'fall': 1
    }

    filename_ts = run_datetime.strftime('%Y%m%d%H')
    save_filename = f"{wfo_id}_{model}_{filename_ts}Z{BUFKT_SAVE_EXTENSION}"
    save_path = os.path.join(target_wfo_dir, save_filename)

    if os.path.exists(save_path):
        return True # Existing file is success

    # --- MODIFIED: Print model being fetched ---
    print(f"  Fetching Bufkit for {wfo_id} ({model} {filename_ts}Z)... ", end="")

    response = make_api_request(bufkit_url, params=params)

    if response is None:
        print("Failed (no response after retries).")
        return False

    if response.status_code == 200:
        if len(response.text) < 100 or "Traceback" in response.text or "ERROR" in response.text[:10]:
             print(f"Failed (API returned status 200 but response seems empty or indicates error for {wfo_id}/{model}).")
             return False
        try:
            with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(response.text)
            print("Success.")
            time.sleep(0.3)
            return True
        except IOError as e:
            print(f"Failed (Error saving file {save_path}: {e}).")
            return False
    elif response.status_code == 422:
        print(f"Failed (API Validation Error 422 - Likely no profile found for {wfo_id}/{model} params).")
        return False
    elif response.status_code == 404:
         print(f"Failed (API Endpoint Not Found 404).")
         return False
    else:
        print(f"Failed (HTTP Status {response.status_code}).")
        return False


# --- AFOS Download Function (No changes needed here) ---
def download_afos_products_for_wfo(wfo_id, target_date, target_wfo_dir):
    """
    Downloads AFOS text products for a specific WFO for the target date.
    Returns number of products saved.
    """
    afos_url = urljoin(IEM_BASE_URL, AFOS_API_ENDPOINT)
    start_date_str = target_date.strftime('%Y-%m-%dT00:00:00Z')
    end_date = target_date + timedelta(days=1)
    end_date_str = end_date.strftime('%Y-%m-%dT00:00:00Z')

    saved_total_for_wfo = 0
    print(f"  Fetching AFOS for {wfo_id} ({target_date.strftime('%Y-%m-%d')}):")

    for pil_generic in AFOS_PILS_GENERIC:
        pil_specific = f"{pil_generic}{wfo_id}"
        params = {
            'pil': pil_specific,
            'sdate': start_date_str,
            'edate': end_date_str,
            'fmt': 'text',
            'limit': 9999
        }

        print(f"    Querying {pil_specific}... ", end="")
        response = make_api_request(afos_url, params=params, timeout=45)

        if response is None or response.status_code != 200:
            status = f"HTTP {response.status_code}" if response else "No Response"
            print(f"Failed ({status}).")
            continue

        raw_text = response.text
        if not raw_text or raw_text.isspace() or "ERROR" in raw_text:
            print("No products found.")
            continue

        products = raw_text.strip().split('\x01')
        saved_count_for_pil = 0
        for product_text in products:
            product_text = product_text.strip()
            if not product_text: continue

            timestamp_str = "unknown"
            lines = product_text.split('\n', 3)
            if len(lines) > 1:
                match = WMO_TIMESTAMP_REGEX.search(lines[1])
                if match:
                    ddhhmm = match.group(1)
                    timestamp_str = f"{target_date.strftime('%y%m')}{ddhhmm}"

            if timestamp_str == "unknown": timestamp_str = f"item{saved_count_for_pil+1}"

            save_filename = f"{pil_generic}_{timestamp_str}.txt"
            save_path = os.path.join(target_wfo_dir, save_filename)

            counter = 1
            while os.path.exists(save_path):
                 save_filename = f"{pil_generic}_{timestamp_str}_{counter}.txt"
                 save_path = os.path.join(target_wfo_dir, save_filename)
                 counter += 1

            try:
                with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(product_text)
                saved_count_for_pil += 1
            except IOError as e:
                print(f"\n     ! Error saving AFOS file {save_path}: {e}")

        print(f"Saved {saved_count_for_pil} product(s).")
        saved_total_for_wfo += saved_count_for_pil
        time.sleep(0.3)

    return saved_total_for_wfo


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Download BUFKIT profiles closest to each WFO and AFOS products for specific model run date/hour(s).",
        epilog=f"Example: python fetch_all_daily_data_multi_model.py 2025-03-14 ./wfo_organized_archive --hour 12 --model RAP HRRR"
    )
    parser.add_argument(
        "run_date",
        help="The model run date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "output_base_dir",
        help="The base directory where dated folders (YYYYMMDD/WFO_ID/) will be created/updated."
    )
    parser.add_argument(
        "--hour",
        default="06",
        choices=["00", "06", "12", "18"],
        help="The model run hour (UTC) to download (default: 06)."
    )
    # --- MODIFIED: Accept multiple models ---
    parser.add_argument(
        "--model",
        nargs='+', # Accept one or more space-separated values
        default=[DEFAULT_MODEL], # Default is a list containing the default model
        choices=['GFS', 'HRRR', 'NAM', 'NAM4KM', 'RAP'],
        help=f"Space-separated list of model names (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--skip-bufkit",
        action="store_true",
        help="Skip downloading BUFKIT files."
    )
    parser.add_argument(
        "--skip-afos",
        action="store_true",
        help="Skip downloading AFOS text products."
    )

    args = parser.parse_args()

    # Validate date format
    try:
        target_run_dt_naive = datetime.strptime(f"{args.run_date} {args.hour}", '%Y-%m-%d %H')
        target_run_dt_utc = target_run_dt_naive
        target_date_only = target_run_dt_utc.date()
    except ValueError:
        print("Error: Invalid date or hour format. Use YYYY-MM-DD and HH.")
        sys.exit(1)

    run_hour_str = args.hour.zfill(2)
    # --- MODIFIED: models_to_run is now a list ---
    models_to_run = [m.upper() for m in args.model]
    date_folder_name = target_date_only.strftime('%Y%m%d')
    final_date_dir = os.path.join(args.output_base_dir, date_folder_name)

    print("="*60)
    print(f"Starting Data Fetch for Run Date: {target_run_dt_utc.strftime('%Y-%m-%d %H:%M')}Z")
    # --- MODIFIED: Print all models ---
    print(f"Models to Process: {', '.join(models_to_run)}")
    print(f"Output Directory Base: {os.path.abspath(final_date_dir)}")
    print(f"Bufkit Downloads: {'SKIPPED' if args.skip_bufkit else 'ENABLED'}")
    print(f"AFOS Downloads: {'SKIPPED' if args.skip_afos else 'ENABLED'}")
    print("="*60)

    os.makedirs(final_date_dir, exist_ok=True)

    # --- Main Loop ---
    total_wfos = len(WFO_LOCATIONS)
    processed_wfo_count = 0
    bufkit_success_count = 0
    bufkit_fail_count = 0
    afos_saved_count = 0

    for wfo_id, (wfo_lat, wfo_lon) in WFO_LOCATIONS.items():
        processed_wfo_count += 1
        print(f"\nProcessing WFO {processed_wfo_count}/{total_wfos}: {wfo_id}...")
        target_wfo_dir = os.path.join(final_date_dir, wfo_id)
        os.makedirs(target_wfo_dir, exist_ok=True)

        # --- Download Bufkit (Inner loop for models) ---
        if not args.skip_bufkit:
            # --- ADDED: Loop through specified models ---
            for current_model in models_to_run:
                success = download_and_save_bufkit_for_wfo(
                    wfo_id, wfo_lat, wfo_lon, target_run_dt_utc, current_model, target_wfo_dir
                )
                # Counts accumulate across all models for this WFO run
                if success:
                    bufkit_success_count += 1
                else:
                    bufkit_fail_count += 1

        # --- Download AFOS (Run once per WFO, outside model loop) ---
        if not args.skip_afos:
            afos_saved_for_wfo = download_afos_products_for_wfo(
                wfo_id, target_date_only, target_wfo_dir
            )
            afos_saved_count += afos_saved_for_wfo

    # --- Final Summary ---
    print("\n" + "="*60)
    print("Overall Script Finished")
    print(f"Target Run Date/Hour: {target_run_dt_utc.strftime('%Y-%m-%d %H:%M')}Z")
    print(f"Models Processed: {', '.join(models_to_run)}")
    print(f"Data organized in: {os.path.abspath(final_date_dir)}")
    if not args.skip_bufkit:
        # --- MODIFIED: Clarify success count meaning ---
        print(f"BUFKIT Files Processed: {bufkit_success_count} successful attempts (incl. existing), {bufkit_fail_count} failed/not found.")
        print(f"  (Note: Success count includes attempts for {len(models_to_run)} model(s) per WFO)")
    if not args.skip_afos:
        print(f"AFOS Products Saved: {afos_saved_count}")
    print("="*60)