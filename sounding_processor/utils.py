# sounding_processor/utils.py
"""
General utility functions for the sounding processor.
"""
import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Union
from pathlib import Path # Added Path

logger = logging.getLogger(__name__)

def convert_numpy_types_for_json(obj: Any) -> Any:
    """
    Recursively converts NumPy and Pandas types in a nested structure
    to JSON-serializable Python native types.
    Pandas DataFrames and Series are replaced with a placeholder string.
    NaN/NaT values are converted to None.
    """
    if pd.isna(obj): # This handles np.nan, pd.NaT, None already
        return None
    if isinstance(obj, pd.DataFrame):
        # logger.debug("Skipping Pandas DataFrame serialization, returning placeholder.")
        return "DataFrame object - Not Serialized"
    if isinstance(obj, pd.Series):
        # logger.debug("Skipping Pandas Series serialization, returning placeholder.")
        return "Series object - Not Serialized"
    if isinstance(obj, dict):
        return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types_for_json(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        # np.isnan already handled by pd.isna at the start for scalar np.floating
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.datetime64, pd.Timestamp)): # Handle datetime objects
        return obj.isoformat()

    # For MetPy quantities, try to get magnitude if simple, else string representation
    # This part is tricky as it depends on MetPyManager's units being available.
    # For a generic util, it's better to avoid direct MetPy dependency here.
    # The _round_quantity_magnitude in MetPyManager should handle this before it gets here.
    # If a raw MetPy quantity somehow reaches here, convert to string.
    if hasattr(obj, 'magnitude') and hasattr(obj, 'units'):
        # Check if magnitude is simple (scalar number)
        if isinstance(obj.magnitude, (int, float, np.integer, np.floating)):
            if pd.isna(obj.magnitude): # Redundant due to initial pd.isna(obj) check if obj itself is a quantity
                return None
            # Return just the magnitude, units are lost here for JSON.
            # If units are desired, they should be formatted into a string or separate field earlier.
            mag = obj.magnitude
            return float(mag) if isinstance(mag, (float, np.floating)) else int(mag)
        elif isinstance(obj.magnitude, np.ndarray) and obj.magnitude.ndim == 0: # 0-d array
             mag_item = obj.magnitude.item()
             if pd.isna(mag_item): return None
             return float(mag_item) if isinstance(mag_item, (float, np.floating)) else int(mag_item)
        # For array quantities or complex magnitudes, string representation is safer for generic JSON.
        logger.debug(f"Serializing MetPy Quantity {obj} as string.")
        return str(obj)

    # Standard float NaN check (should be caught by pd.isna if obj is float, but as safeguard)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def setup_logging(log_file_path: Path, file_level: int, console_level: int) -> None:
    """
    Sets up basic logging to a file and to the console.
    Ensures handlers are not duplicated if called multiple times (though ideally called once).
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger() # Get the root logger
    
    # Prevent adding handlers multiple times if this function is called again
    if root_logger.hasHandlers():
        # Clear existing handlers if re-configuring, or simply return if setup is idempotent
        # For simplicity in this first pass, let's assume it's called once.
        # If re-configuration is needed, handler management becomes more complex.
        logger.debug("Logging already configured. Skipping reconfiguration.")
        # If you want to allow re-configuration, you might clear handlers:
        # for handler in root_logger.handlers[:]:
        #     root_logger.removeHandler(handler)
        # But this is often not what you want. Best to call setup_logging once.
        # For now, if handlers exist, we assume it's the desired configuration.
        # Check if specifically our FileHandler and StreamHandler are present
        # This is still tricky. Let's assume if any handlers, it's configured.
        return


    # Configure root logger (this will affect all loggers unless they have specific handlers)
    root_logger.setLevel(min(file_level, console_level)) # Set root to the lowest level needed by handlers

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log file each run
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: [%(name)s] %(message)s') # Simpler format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    initial_logger = logging.getLogger(__name__) # Logger for this util module
    initial_logger.info(f"Logging initialized. Log file: {log_file_path}")
    initial_logger.debug("Debug logging enabled for file.")