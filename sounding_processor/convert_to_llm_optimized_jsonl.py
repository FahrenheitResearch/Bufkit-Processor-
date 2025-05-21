# sounding_processor/convert_to_llm_optimized_jsonl.py
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys # For direct run logging
from typing import Dict, Any, List, Optional, Union

# Assuming this script might be run directly or via main.py
# If run directly, config might not be fully set up via main module.
try:
    from . import config as app_config
except ImportError: # Fallback for direct execution from project root
    import config as app_config


logger = logging.getLogger(__name__)

# --- Configuration for LLM Optimization (Aggressive) ---

# Verbose Key -> Ultra-Short Key
LLM_KEY_MAPPING = {
    # Metadata
    "STID": "id", "TIME": "tm", "SLAT": "lat", "SLON": "lon", "SELV": "elv",

    # Parcel Thermodynamics
    "sfc_cape_jkg": "sCP", "sfc_cin_jkg": "sCN",
    "mucape_jkg": "mCP", "mucin_jkg": "mCN",
    "sfc_lcl_h_m_agl": "sLCLh",
    "mu_lfc_h_m_agl": "mLFCh", "mu_el_h_m_agl": "mELh",
    "dcape_jkg": "DCP", # Downdraft CAPE

    # Detailed Kinematics
    "bulk_shear_0_6km_kts": "BS06",
    "srh_0_1km_m2s2_eff_sm": "SRH1e",
    "srh_0_3km_m2s2_eff_sm": "SRH3e",
    # "bunkers_right_mover_u_kts": "BRMu", # Omitting for max brevity, can be added back
    # "bunkers_right_mover_v_kts": "BRMv", # Omitting for max brevity

    # Environmental Thermodynamics
    "precipitable_water_mm": "PW", "lifted_index_c": "LI",
    "freezing_level_h_m_agl": "FZLh", "wet_bulb_zero_h_m_agl": "WBZh",
    "lapse_rate_700_500mb_c_km": "LR75", # One representative mid-level lapse rate

    # Composite Indices
    "stp_fixed_ml": "STP",
    "ship_calculated_v1": "SHIP",

    # Key Levels Data - internal keys for level objects
    "p_hpa": "p", "h_m_agl": "h", "t_c": "t", "td_c": "td",
    "wind_spd_kts": "ws", "wind_dir_deg": "wd",
}

# Define which verbose sections and which specific keys from those sections to process
CRITICAL_PARAMS_SECTIONS = {
    "metadata": ["STID", "TIME", "SLAT", "SLON", "SELV"],
    "parcel_thermodynamics": [
        "sfc_cape_jkg", "sfc_cin_jkg", "mucape_jkg", "mucin_jkg",
        "sfc_lcl_h_m_agl", "mu_lfc_h_m_agl", "mu_el_h_m_agl", "dcape_jkg"
    ],
    "detailed_kinematics": [
        "bulk_shear_0_6km_kts", "srh_0_1km_m2s2_eff_sm", "srh_0_3km_m2s2_eff_sm",
        # "bunkers_right_mover_u_kts", "bunkers_right_mover_v_kts" # Add if needed
    ],
    "environmental_thermodynamics": [
        "precipitable_water_mm", "lifted_index_c",
        "freezing_level_h_m_agl", "wet_bulb_zero_h_m_agl", "lapse_rate_700_500mb_c_km"
    ],
    "composite_indices": [
        "stp_fixed_ml", "ship_calculated_v1"
    ],
}

# Define which key levels to include and which params from them
CRITICAL_KEY_LEVELS_PARAMS = {
    "sfc":   ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "850mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "700mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "500mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"], # Keep Td for 500mb dryness
    "250mb": ["p_hpa", "h_m_agl", "t_c", "wind_spd_kts", "wind_dir_deg"]          # Td often less critical at jet level
}
# Order of key levels for output array
KEY_LEVEL_OUTPUT_ORDER = ["sfc", "850mb", "700mb", "500mb", "250mb"]


# Precision for these specific LLM keys
LLM_ROUNDING_PRECISION = {
    "sCP": 0, "sCN": 0, "mCP": 0, "mCN": 0, "DCP":0,
    "sLCLh": 0, "mLFCh": 0, "mELh": 0, "FZLh": 0, "WBZh": 0, "h":0, "elv":0,
    "BS06": 0, "SRH1e": 0, "SRH3e": 0,
    "BRMu": 0, "BRMv": 0, # If added back
    "PW": 1, "LI": 1, "LR75": 1,
    "STP": 1, "SHIP": 1,
    "lat": 2, "lon": 2,
    "p": 0, "t": 0, "td": 0, "ws": 0, "wd": 0, # Integer values for key levels
}

def get_llm_precision(llm_key: str) -> int:
    """Determines rounding precision based on the LLM key."""
    return LLM_ROUNDING_PRECISION.get(llm_key, 0) # Default to 0 for aggressive token saving

def llm_custom_round(value: Any, llm_key_for_precision_rule: str) -> Any:
    """Rounds a numerical value based on its LLM key for precision rules."""
    if value is None or (isinstance(value, float) and np.isnan(value)): # pd.isna also covers np.nan
        return None
    if isinstance(value, (int, float, np.number)):
        precision = get_llm_precision(llm_key_for_precision_rule)
        rounded_value = round(float(value), precision)
        # Convert to int if precision is 0 and value is whole number (e.g. 15.0 -> 15)
        if precision == 0 and rounded_value == int(rounded_value):
            return int(rounded_value)
        return rounded_value
    return value # Return non-numeric types as is (shouldn't be many in selected data)


# sounding_processor/convert_to_llm_optimized_jsonl.py
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys 
from typing import Dict, Any, List, Optional, Union

try:
    from . import config as app_config
except ImportError: 
    import config as app_config


logger = logging.getLogger(__name__)

# --- Configuration for LLM Optimization (Aggressive V2) ---

LLM_KEY_MAPPING = {
    "STID": "id", "TIME": "tm", "SLAT": "lat", "SLON": "lon", "SELV": "elv",
    "sfc_cape_jkg": "sCP", "sfc_cin_jkg": "sCN",
    "mucape_jkg": "mCP", "mucin_jkg": "mCN",
    "sfc_lcl_h_m_agl": "sLCLh",
    "mu_lfc_h_m_agl": "mLFCh", "mu_el_h_m_agl": "mELh",
    "dcape_jkg": "DCPk", # DCPk to distinguish from composite DCP
    "bulk_shear_0_6km_kts": "BS06",
    "srh_0_1km_m2s2_eff_sm": "SRH1e",
    "srh_0_3km_m2s2_eff_sm": "SRH3e",
    "precipitable_water_mm": "PW", "lifted_index_c": "LI",
    "freezing_level_h_m_agl": "FZLh", "wet_bulb_zero_h_m_agl": "WBZh",
    "lapse_rate_700_500mb_c_km": "LR75",
    "stp_fixed_ml": "STP",
    "ship_calculated_v1": "SHIP",
    # Key Levels internal keys
    "p_hpa": "p", "h_m_agl": "h", "t_c": "t", "td_c": "td",
    "wind_spd_kts": "ws", "wind_dir_deg": "wd",
}

# List of verbose keys to extract and map to top-level LLM keys
CRITICAL_VERBOSE_KEYS_TO_FLATTEN = [
    # Metadata
    "STID", "TIME", "SLAT", "SLON", "SELV",
    # Parcel Thermo
    "sfc_cape_jkg", "sfc_cin_jkg", "mucape_jkg", "mucin_jkg",
    "sfc_lcl_h_m_agl", "mu_lfc_h_m_agl", "mu_el_h_m_agl", "dcape_jkg",
    # Kinematics
    "bulk_shear_0_6km_kts", "srh_0_1km_m2s2_eff_sm", "srh_0_3km_m2s2_eff_sm",
    # Environmental
    "precipitable_water_mm", "lifted_index_c", "freezing_level_h_m_agl",
    "wet_bulb_zero_h_m_agl", "lapse_rate_700_500mb_c_km",
    # Composites
    "stp_fixed_ml", "ship_calculated_v1",
]

# Define which key levels to include and which params from them
CRITICAL_KEY_LEVELS_PARAMS = {
    "sfc":   ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "850mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "700mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "500mb": ["p_hpa", "h_m_agl", "t_c", "td_c", "wind_spd_kts", "wind_dir_deg"],
    "250mb": ["p_hpa", "h_m_agl", "t_c",        "wind_spd_kts", "wind_dir_deg"] # Td omitted for jet
}
KEY_LEVEL_OUTPUT_ORDER = ["sfc", "850mb", "700mb", "500mb", "250mb"]

LLM_ROUNDING_PRECISION = {
    "id": None, "tm": None, "lat": 2, "lon": 2, "elv": 0, # Meta (None means no rounding for string types)
    "sCP": 0, "sCN": 0, "mCP": 0, "mCN": 0, "DCPk": 0,    # Thermo
    "sLCLh": 0, "mLFCh": 0, "mELh": 0,
    "BS06": 0, "SRH1e": 0, "SRH3e": 0,                   # Kinematics
    "PW": 1, "LI": 1, "FZLh": 0, "WBZh": 0, "LR75": 1,   # Environmental
    "STP": 1, "SHIP": 1,                                # Composites
    "p": 0, "h": 0, "t": 0, "td": 0, "ws": 0, "wd": 0    # Key Levels
}

def get_llm_precision(llm_key: str) -> Optional[int]:
    return LLM_ROUNDING_PRECISION.get(llm_key) # Allow None for no rounding

def llm_custom_round(value: Any, llm_key_for_precision_rule: str) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
        
    precision = get_llm_precision(llm_key_for_precision_rule)
    if precision is None: # No rounding for this key (e.g., string IDs)
        return value

    if isinstance(value, (int, float, np.number)):
        rounded_value = round(float(value), precision)
        if precision == 0 and rounded_value == int(rounded_value):
            return int(rounded_value)
        return rounded_value
    return value


def find_value_in_nested_dict(data_dict: Dict, target_key: str) -> Any:
    """Helper to find a verbose key in potentially nested verbose structure."""
    if target_key in data_dict:
        return data_dict[target_key]
    for k, v in data_dict.items():
        if isinstance(v, dict):
            found = find_value_in_nested_dict(v, target_key)
            if found is not None: # Return the first found value
                return found
    return None


def transform_sounding_to_efficient(verbose_sounding: Dict[str, Any]) -> Dict[str, Any]:
    efficient_sounding = {}

    # Flatten critical parameters from various sections
    for verbose_key in CRITICAL_VERBOSE_KEYS_TO_FLATTEN:
        value = find_value_in_nested_dict(verbose_sounding, verbose_key)
        if value is not None: # Only process if key was found and value is not None
            llm_key = LLM_KEY_MAPPING.get(verbose_key)
            if llm_key:
                rounded_value = llm_custom_round(value, llm_key)
                if rounded_value is not None:
                    efficient_sounding[llm_key] = rounded_value
    
    # Process Key Levels Data
    key_levels_list_efficient = []
    verbose_key_levels_data = verbose_sounding.get("key_levels_data", {})
    if verbose_key_levels_data:
        for level_name_ordered in KEY_LEVEL_OUTPUT_ORDER:
            if level_name_ordered in verbose_key_levels_data and \
               level_name_ordered in CRITICAL_KEY_LEVELS_PARAMS:
                
                level_data_verbose = verbose_key_levels_data[level_name_ordered]
                critical_params_for_level = CRITICAL_KEY_LEVELS_PARAMS[level_name_ordered]
                
                transformed_level_dict = {}
                level_id_str = level_name_ordered.replace("mb", "") if level_name_ordered != "sfc" else "sfc"
                try: transformed_level_dict["lvl"] = int(level_id_str)
                except ValueError: transformed_level_dict["lvl"] = level_id_str

                has_data_for_level = False
                for verbose_param_key in critical_params_for_level:
                    if verbose_param_key in level_data_verbose:
                        value = level_data_verbose[verbose_param_key]
                        llm_param_key = LLM_KEY_MAPPING.get(verbose_param_key)
                        if llm_param_key:
                            rounded_value = llm_custom_round(value, llm_param_key)
                            if rounded_value is not None:
                                transformed_level_dict[llm_param_key] = rounded_value
                                has_data_for_level = True # Mark that this level has at least one piece of data
                
                if has_data_for_level or transformed_level_dict["lvl"] == "sfc": # Always include sfc structure if it exists
                    # Add only if it has substantial data beyond just 'lvl' or if it's 'sfc'
                    if len(transformed_level_dict) > 1 : 
                        key_levels_list_efficient.append(transformed_level_dict)
        
        if key_levels_list_efficient:
            efficient_sounding["kl"] = key_levels_list_efficient
            
    return efficient_sounding


def process_file(input_jsonl_path: Path, output_jsonl_path: Path):
    logger.info(f"Starting LLM optimization process for: {input_jsonl_path}")
    if not input_jsonl_path.exists():
        logger.error(f"Input file for LLM optimization not found: {input_jsonl_path}")
        print(f"ERROR: Input file for LLM optimization not found: {input_jsonl_path}", file=sys.stderr)
        return

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0

    with input_jsonl_path.open('r', encoding='utf-8') as infile, \
         output_jsonl_path.open('w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                verbose_sounding = json.loads(line)
                efficient_sounding = transform_sounding_to_efficient(verbose_sounding)
                if efficient_sounding: 
                    json.dump(efficient_sounding, outfile, separators=(',', ':')) 
                    outfile.write('\n')
                    processed_count += 1
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e_transform:
                logger.error(f"Error transforming sounding on line {line_num}: {e_transform}", exc_info=True)
                error_count += 1
                
    logger.info(f"LLM optimization complete. Processed: {processed_count} soundings. Errors: {error_count}.")
    logger.info(f"Efficient output saved to: {output_jsonl_path}")


if __name__ == "__main__":
    project_root_path = Path(__file__).resolve().parent.parent 
    
    # Determine if app_config was imported from .config or just config
    config_module_loaded = 'app_config' in globals() and hasattr(app_config, 'LOG_DIRECTORY_NAME')
    
    log_dir_name = app_config.LOG_DIRECTORY_NAME if config_module_loaded else "debug_logs"
    log_dir_path = project_root_path / "data" / "output" / log_dir_name
    log_file_name = "llm_conversion_direct_run_debug.txt"
    
    log_dir_path.mkdir(parents=True, exist_ok=True)
    full_log_path = log_dir_path / log_file_name

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(full_log_path, mode='w'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logger.info("Running LLM optimization script directly.")

    default_verbose_fn = app_config.DEFAULT_OUTPUT_JSONL_FILENAME if config_module_loaded else "processed_soundings_default.jsonl"
    default_verbose_input_path = project_root_path / "data" / "output" / default_verbose_fn
    
    default_llm_output_path = project_root_path / "data" / "output" / \
                              (default_verbose_input_path.stem + "_llm_slim" + default_verbose_input_path.suffix) # Changed suffix

    print(f"Attempting to process: {default_verbose_input_path}")
    print(f"Outputting LLM optimized to: {default_llm_output_path}")
    
    process_file(default_verbose_input_path, default_llm_output_path)

def process_file(input_jsonl_path: Path, output_jsonl_path: Path):
    """
    Reads the verbose JSONL, transforms each line, and writes to the efficient JSONL.
    """
    logger.info(f"Starting LLM optimization process for: {input_jsonl_path}")
    if not input_jsonl_path.exists():
        logger.error(f"Input file for LLM optimization not found: {input_jsonl_path}")
        print(f"ERROR: Input file for LLM optimization not found: {input_jsonl_path}", file=sys.stderr)
        return

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0

    with input_jsonl_path.open('r', encoding='utf-8') as infile, \
         output_jsonl_path.open('w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                verbose_sounding = json.loads(line)
                efficient_sounding = transform_sounding_to_efficient(verbose_sounding)
                if efficient_sounding: # Only write if there's something to write
                    json.dump(efficient_sounding, outfile, separators=(',', ':')) # Most compact JSON
                    outfile.write('\n')
                    processed_count += 1
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e_transform:
                logger.error(f"Error transforming sounding on line {line_num}: {e_transform}", exc_info=True)
                error_count += 1
                
    logger.info(f"LLM optimization complete. Processed: {processed_count} soundings. Errors: {error_count}.")
    logger.info(f"Efficient output saved to: {output_jsonl_path}")


if __name__ == "__main__":
    # Setup basic logging if run directly
    # This allows running this script directly for testing/conversion purposes.
    project_root_path = Path(__file__).resolve().parent.parent # sounding_processor_project
    
    # Check if app_config was imported successfully (i.e., not ImportError)
    if 'app_config' in globals() and hasattr(app_config, 'LOG_DIRECTORY_NAME'):
        log_dir_path = project_root_path / "data" / "output" / app_config.LOG_DIRECTORY_NAME
        log_file_name = "llm_conversion_direct_run_debug.txt"
    else: # Minimal fallback if config couldn't be imported (e.g. path issues on direct run)
        log_dir_path = project_root_path / "data" / "output" / "debug_logs" # Guess
        log_file_name = "llm_conversion_direct_run_debug.txt"
        print("Warning: app_config not fully loaded, using default log path for direct run.", file=sys.stderr)


    log_dir_path.mkdir(parents=True, exist_ok=True)
    full_log_path = log_dir_path / log_file_name

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(full_log_path, mode='w'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logger.info("Running LLM optimization script directly.")

    # Define default input and output paths for direct execution
    # Input is the output of the main processing script
    default_verbose_input_path = project_root_path / "data" / "output" / \
                                 (app_config.DEFAULT_OUTPUT_JSONL_FILENAME if hasattr(app_config, 'DEFAULT_OUTPUT_JSONL_FILENAME') else "processed_soundings_default.jsonl")
    
    default_llm_output_path = project_root_path / "data" / "output" / \
                              (default_verbose_input_path.stem + "_llm_optimized_direct" + default_verbose_input_path.suffix)


    print(f"Attempting to process: {default_verbose_input_path}")
    print(f"Outputting LLM optimized to: {default_llm_output_path}")
    
    process_file(default_verbose_input_path, default_llm_output_path)