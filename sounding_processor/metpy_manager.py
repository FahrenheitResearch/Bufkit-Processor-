# sounding_processor/sounding_processor/metpy_manager.py
"""
Manages MetPy library interactions, availability checks, and provides
MetPy functions or fallbacks.
"""
import sys
import logging
import numpy as np
import pandas as pd
import traceback
from typing import Optional, Tuple, Any, Union, Callable

# Use a relative import for config within the package
from . import config as app_config # Renamed to avoid clash with global 'config' if any

logger = logging.getLogger(__name__)

# --- Early MetPy Import Diagnostics (from original script) ---
if app_config.EARLY_DIAGNOSTICS_ENABLED:
    # Using logger for diagnostics now, configured by main.py
    logger.info("--- DIAGNOSTIC: Early sys.path (MetPyManager) ---")
    for p_diag in sys.path:
        logger.info(p_diag)
    logger.info("--- DIAGNOSTIC: Attempting to import metpy and metpy.calc (MetPyManager) ---")
    try:
        import metpy
        logger.info(f"DIAGNOSTIC: metpy imported. Version: {metpy.__version__}, Path: {metpy.__file__}")
        import metpy.calc
        logger.info(f"DIAGNOSTIC: metpy.calc imported. Path: {metpy.calc.__file__}")
        logger.info(f"DIAGNOSTIC: Does metpy.calc have 'mean_wind'? {hasattr(metpy.calc, 'mean_wind')}")
        logger.info(f"DIAGNOSTIC: Does metpy.calc have 'lfc'? {hasattr(metpy.calc, 'lfc')}")
        if hasattr(metpy.calc, 'lfc'):
            import inspect
            logger.info(f"DIAGNOSTIC: metpy.calc.lfc signature: {inspect.signature(metpy.calc.lfc)}")
    except ImportError as e_diag_imp:
        logger.error(f"DIAGNOSTIC: ImportError during MetPy/calc import: {e_diag_imp}")
    except AttributeError as e_diag_attr:
        logger.error(f"DIAGNOSTIC: AttributeError during MetPy/calc access: {e_diag_attr}")
    except Exception as e_diag: #skipcq: PYL-W0703
        logger.error(f"DIAGNOSTIC: Other exception during MetPy/calc diagnostics: {e_diag}")
    logger.info("--- DIAGNOSTIC: End of MetPy/calc diagnostics (MetPyManager) ---")


class MetPyManager:
    """
    Singleton class to manage MetPy imports and provide access to its
    functionalities, with fallbacks where appropriate.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MetPyManager, cls).__new__(cls, *args, **kwargs)
            # Initialize only once when the instance is created
            cls._instance._init_metpy_components()
        return cls._instance

    def _init_metpy_components(self):
        # This method is called by __new__ only on first instance creation.
        if hasattr(self, '_initialized') and self._initialized:
            return # Already initialized
        
        self.metpy_available = False
        self.mpcalc = None
        self.units = None
        self.mpinterp = None # Store the interpolate module itself
        self.interpolate_1d_func: Optional[Callable] = None
        self.log_interpolate_1d_func: Optional[Callable] = None
        self.metpy_version = "N/A"
        self.metpy_path = "N/A"

        try:
            # `metpy` module itself doesn't need to be global here if only used via self.
            import metpy as metpy_module # Alias to avoid conflict if 'metpy' used as var name
            self.metpy_version = metpy_module.__version__
            self.metpy_path = metpy_module.__file__
            logger.info(f"MetPy version: {self.metpy_version}, Path: {self.metpy_path}")

            from metpy import calc as mpcalc_imported
            from metpy.units import units as units_imported
            from metpy import interpolate as mpinterp_imported

            self.mpcalc = mpcalc_imported
            self.units = units_imported
            self.mpinterp = mpinterp_imported # Store the module

            if hasattr(self.mpinterp, 'interpolate_1d'):
                self.interpolate_1d_func = self.mpinterp.interpolate_1d
                app_config.METPY_INTERPOLATE_1D_AVAILABLE = True # Update config
                logger.debug("MetPy mpinterp.interpolate_1d IS available.")
            else:
                app_config.METPY_INTERPOLATE_1D_AVAILABLE = False
                logger.warning("MetPy mpinterp.interpolate_1d IS ***NOT*** available. Linear interpolation will use NumPy.")

            if hasattr(self.mpinterp, 'log_interpolate_1d'):
                self.log_interpolate_1d_func = self.mpinterp.log_interpolate_1d
                app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE = True # Update config
                logger.debug("MetPy mpinterp.log_interpolate_1d IS available.")
            else:
                app_config.METPY_LOG_INTERPOLATE_1D_AVAILABLE = False
                logger.warning("MetPy mpinterp.log_interpolate_1d IS ***NOT*** available. Log interpolation will use NumPy.")

            self.metpy_available = True
            logger.info("MetPy successfully imported and components initialized.")

        except ImportError as e_metpy_top:
            self.metpy_available = False # Ensure it's false
            logger.critical(f"Failed to import MetPy: {e_metpy_top}. Sounding calculations will be severely limited.")
            # Print to stderr for immediate visibility if logging isn't fully set up or noticed
            print(f"CRITICAL WARNING: MetPy library not found or import failed: {e_metpy_top}. Key derived parameters will be missing.", file=sys.stderr)
        except Exception as e_other_top: #skipcq: PYL-W0703
            self.metpy_available = False # Ensure it's false
            logger.critical(f"Some other error during MetPy setup: {e_other_top}\n{traceback.format_exc()}")
            print(f"CRITICAL WARNING: Error during MetPy setup: {e_other_top}. Key derived parameters may be missing.", file=sys.stderr)
        
        self._initialized = True


    def _round_quantity_magnitude(self, quantity: Any, precision: int, target_units_str: Optional[str] = None) -> Union[float, int, np.ndarray, Any]:
        """
        Rounds the magnitude of a MetPy Quantity, optionally converting units.
        Returns np.nan if MetPy is not available or input is invalid.
        """
        if not self.metpy_available or self.units is None:
            return np.nan
        
        if pd.isna(quantity): 
            return np.nan
        
        if isinstance(quantity, (float, int)): 
            if isinstance(quantity, float) and np.isnan(quantity): return np.nan
            return round(quantity, precision)

        # PATCH: Handle raw 0-d numpy arrays that might be magnitudes without units
        if isinstance(quantity, np.ndarray) and quantity.ndim == 0 and not hasattr(quantity, 'units'):
            val_to_round_scalar = quantity.item()
            if pd.isna(val_to_round_scalar): return np.nan
            
            # This case is tricky if target_units_str is provided, as we don't have original units.
            # Assuming target_units_str would only be 'dimensionless' or for re-interpretation of a raw number.
            # If target_units_str implies a conversion that can't be done, it's an issue with the caller.
            if target_units_str and target_units_str != str(self.units.dimensionless) and target_units_str != "dimensionless": # Check against actual dimensionless unit string
                 logger.warning(f"Rounding 0-d numpy array '{quantity}' with target_units '{target_units_str}' "
                                f"but no original units. Output will be unitless magnitude.")
            try:
                return round(float(val_to_round_scalar), precision)
            except (TypeError, ValueError) as e_round_np0d:
                logger.warning(f"Could not round 0-d numpy array item: {val_to_round_scalar}. Error: {e_round_np0d}")
                return np.nan
        # END PATCH

        is_quantity_like = hasattr(quantity, 'magnitude') and hasattr(quantity, 'units')

        if not is_quantity_like:
            # If it's not a Quantity, and not a number (handled above), and not a 0-d array (handled above)
            # then it's an unexpected type.
            logger.warning(f"Attempting to round non-MetPy Quantity like object: {quantity} (type: {type(quantity)}). This function expects MetPy quantities or numbers.")
            return np.nan

        if pd.isna(quantity.magnitude): 
            return np.nan
        
        quantity_for_rounding = quantity
        if target_units_str:
            try:
                target_units_obj = self.units(target_units_str)
                quantity_for_rounding = quantity.to(target_units_obj)
            except Exception as e_conv: 
                logger.error(f"Unit conversion failed for {quantity} to {target_units_str}: {e_conv}")
                try: return np.nan * self.units(target_units_str)
                except Exception: return np.nan 

        val_to_round = quantity_for_rounding.magnitude
        
        if hasattr(val_to_round, 'item'): 
            val_to_round = val_to_round.item()
        
        if pd.isna(val_to_round): 
            return np.nan
        
        try:
            return round(float(val_to_round), precision)
        except (TypeError, ValueError) as e_round:
            logger.warning(f"Could not round value: {val_to_round} of type {type(val_to_round)}. Error: {e_round}")
            return np.nan

    def _numpy_interp_with_units(self,
                                 target_x_q: Any,
                                 source_x_q: Any, # Expected to be array-like (Quantity or ndarray)
                                 source_y_q: Any, # Expected to be array-like (Quantity or ndarray)
                                 log_x: bool = False) -> Any:
        """
        NumPy-based 1D interpolation that attempts to handle MetPy Quantities.
        Fallback for when MetPy's interpolation is not available or fails.
        Returns a MetPy Quantity (value * original_y_units) or (np.nan * original_y_units).
        """
        if not self.metpy_available or self.units is None: # Should not happen if called as fallback
            logger.error("NumPy interp called but MetPy (and units) supposedly unavailable.")
            return np.nan # Or try to guess units if y had them

        y_units_obj = getattr(source_y_q, 'units', self.units.dimensionless)
        source_x_units = getattr(source_x_q, 'units', self.units.dimensionless)
        
        try:
            # Convert target_x to the units of source_x for consistent interpolation
            if hasattr(target_x_q, 'to'):
                target_x_val_m = target_x_q.to(source_x_units).m
            elif isinstance(target_x_q, (int, float, np.number)):
                target_x_val_m = float(target_x_q) # Assume compatible or dimensionless
            else:
                logger.error(f"Cannot convert target_x for numpy_interp: {target_x_q}")
                return np.nan * y_units_obj

            source_x_vals_m = source_x_q.m if hasattr(source_x_q, 'm') else np.asarray(source_x_q, dtype=float)
            source_y_vals_m = source_y_q.m if hasattr(source_y_q, 'm') else np.asarray(source_y_q, dtype=float)

        except Exception as e_unit_conv: #skipcq: PYL-W0703
            logger.error(f"Error preparing magnitudes for numpy_interp: {e_unit_conv}")
            return np.nan * y_units_obj

        if not isinstance(source_x_vals_m, np.ndarray) or not isinstance(source_y_vals_m, np.ndarray):
            logger.error("Numpy interp source_x_m or source_y_m not ndarray after processing.")
            return np.nan * y_units_obj
        if source_x_vals_m.size <= 1 or source_y_vals_m.size <= 1 or source_x_vals_m.size != source_y_vals_m.size:
            logger.debug(f"Numpy interp: Not enough data or mismatched sizes. x_size: {source_x_vals_m.size}, y_size: {source_y_vals_m.size}")
            return np.nan * y_units_obj
        
        # Handle target_x_val_m being a 0-d array from quantity conversion
        if isinstance(target_x_val_m, np.ndarray) and target_x_val_m.ndim == 0:
            target_x_val_m = target_x_val_m.item()
        
        # Remove NaNs from source arrays (np.interp doesn't like them)
        valid_mask = pd.notna(source_x_vals_m) & pd.notna(source_y_vals_m)
        source_x_vals_m_clean = source_x_vals_m[valid_mask]
        source_y_vals_m_clean = source_y_vals_m[valid_mask]

        if source_x_vals_m_clean.size <= 1:
            logger.debug("Numpy interp: Not enough valid (non-NaN) data points after cleaning.")
            return np.nan * y_units_obj

        if log_x:
            # Check for non-positive values before log transform
            if np.any(target_x_val_m <= 1e-9) or np.any(source_x_vals_m_clean <= 1e-9):
                logger.debug("Numpy interp (log_x): Values <= 0 found, cannot take log.")
                return np.nan * y_units_obj
            target_x_val_m = np.log(target_x_val_m)
            source_x_vals_m_clean = np.log(source_x_vals_m_clean)

        # Sort by source_x_vals and remove duplicates (np.interp requires strictly increasing x)
        sort_indices = np.argsort(source_x_vals_m_clean)
        source_x_vals_sorted = source_x_vals_m_clean[sort_indices]
        source_y_vals_sorted = source_y_vals_m_clean[sort_indices]

        unique_x, unique_indices = np.unique(source_x_vals_sorted, return_index=True)
        if len(unique_x) < len(source_x_vals_sorted):
            source_x_vals_sorted = unique_x
            source_y_vals_sorted = source_y_vals_sorted[unique_indices]
            if source_x_vals_sorted.size <= 1:
                logger.debug("Numpy interp: Not enough unique data points after filtering for duplicates.")
                return np.nan * y_units_obj
        
        try:
            interp_y_val_m = np.interp(target_x_val_m, source_x_vals_sorted, source_y_vals_sorted)
            return interp_y_val_m * y_units_obj # Re-attach original Y units
        except Exception as e_interp: #skipcq: PYL-W0703
            logger.error(f"np.interp failed: {e_interp} for target {target_x_val_m} in {source_x_vals_sorted}")
            return np.nan * y_units_obj

    def interpolate_profile_value(self,
                                  target_coord_q: Any,    # Can be scalar Quantity or number
                                  profile_coord_q: Any,   # Array-like Quantity or ndarray
                                  value_profile_q: Any,   # Array-like Quantity or ndarray
                                  interp_type: str = 'linear') -> Any:
        """
        Interpolates a value from a profile using MetPy if available, otherwise NumPy.
        Returns a MetPy Quantity or (np.nan * units of value_profile_q).
        """
        if not self.metpy_available or self.units is None or self.mpinterp is None:
            logger.warning("MetPy components not available, cannot interpolate profile value sophisticatedly.")
            # Basic check if inputs seem numeric enough for a desperate numpy attempt
            if isinstance(target_coord_q, (int,float,np.number)) and \
               hasattr(profile_coord_q, '__iter__') and hasattr(value_profile_q, '__iter__'):
                try:
                    return self._numpy_interp_with_units(
                        target_coord_q,
                        np.asarray(profile_coord_q), # Attempt to cast
                        np.asarray(value_profile_q), # Attempt to cast
                        log_x=(interp_type.lower() == 'log')
                    )
                except Exception: #skipcq: PYL-W0703
                    return np.nan # Total failure
            return np.nan

        default_value_units = getattr(value_profile_q, 'units', self.units.dimensionless)
        default_coord_units = getattr(profile_coord_q, 'units', self.units.dimensionless)

        # --- Input Validation and Preparation for MetPy ---
        try:
            # Ensure inputs are MetPy Quantities with appropriate units
            target_q = target_coord_q if hasattr(target_coord_q, 'units') else target_coord_q * default_coord_units
            profile_coord_q_arr = profile_coord_q if hasattr(profile_coord_q, 'units') else np.asarray(profile_coord_q) * default_coord_units
            value_profile_q_arr = value_profile_q if hasattr(value_profile_q, 'units') else np.asarray(value_profile_q) * default_value_units
        except Exception as e_unit_prep: #skipcq: PYL-W0703
            logger.error(f"Interpolate: Error preparing quantities: {e_unit_prep}")
            return np.nan * default_value_units
            
        # Check for NaN in target coordinate's magnitude
        if hasattr(target_q, 'magnitude') and pd.isna(target_q.magnitude):
            logger.debug("Interpolate: Target coordinate magnitude is NaN.")
            return np.nan * default_value_units
        
        # Check profile array sizes and content
        if not (hasattr(profile_coord_q_arr, 'size') and hasattr(value_profile_q_arr, 'size')):
             logger.warning("Interpolate: Profile coordinates or values are not array-like with .size")
             return np.nan * default_value_units

        if profile_coord_q_arr.size < 2 or value_profile_q_arr.size < 2 or \
           profile_coord_q_arr.size != value_profile_q_arr.size:
            logger.debug(f"Interpolate: Insufficient data or mismatched array sizes. Profile coord size: {profile_coord_q_arr.size}, Value profile size: {value_profile_q_arr.size}")
            return np.nan * default_value_units

        # MetPy interpolation functions often require sorted coordinates (though some handle it)
        # For interpolate_1d, x must be monotonic. log_interpolate_1d handles sorting.
        # We sort here to be safe and consistent, especially if falling back to NumPy.
        
        # Remove NaNs from profile data before sorting and interpolation
        valid_mask = pd.notna(profile_coord_q_arr.magnitude) & pd.notna(value_profile_q_arr.magnitude)
        if not np.all(valid_mask):
            logger.debug(f"Interpolate: NaN values found in profile. Filtering. Original size: {profile_coord_q_arr.size}")
            profile_coord_q_s_clean = profile_coord_q_arr[valid_mask]
            value_profile_q_s_clean = value_profile_q_arr[valid_mask]
            if profile_coord_q_s_clean.size < 2:
                logger.debug("Interpolate: Less than 2 valid data points after NaN filtering.")
                return np.nan * default_value_units
        else:
            profile_coord_q_s_clean = profile_coord_q_arr
            value_profile_q_s_clean = value_profile_q_arr
            
        # Sort the cleaned data
        # For log interpolation, coordinates must be positive.
        if interp_type.lower() == 'log' and np.any(profile_coord_q_s_clean.magnitude <= 0):
            logger.debug("Interpolate (log): Profile coordinates contain non-positive values. Cannot perform log interpolation.")
            # Try linear as a fallback? Or just fail. For now, fail for log.
            return np.nan * default_value_units
            
        # Sorting by coordinate for MetPy's interpolate_1d and our numpy fallback
        sort_idx_asc = np.argsort(profile_coord_q_s_clean.magnitude)
        profile_coord_q_sorted = profile_coord_q_s_clean[sort_idx_asc]
        value_profile_q_sorted = value_profile_q_s_clean[sort_idx_asc]

        # Check if target is within profile bounds (MetPy can extrapolate, but often not desired)
        min_coord_m = profile_coord_q_sorted.magnitude.min()
        max_coord_m = profile_coord_q_sorted.magnitude.max()
        target_coord_m = target_q.to(profile_coord_q_sorted.units).m # Ensure consistent units for comparison

        if isinstance(target_coord_m, np.ndarray) and target_coord_m.ndim == 0: # if target_q was 0-d Quantity
            target_coord_m = target_coord_m.item()

        if not (min_coord_m <= target_coord_m <= max_coord_m) and \
           not (np.isclose(min_coord_m, target_coord_m) or np.isclose(max_coord_m, target_coord_m)):
            logger.debug(f"Interpolate: Target {target_q} is outside sorted profile bounds ({min_coord_m} to {max_coord_m} {profile_coord_q_sorted.units}). Returning NaN.")
            return np.nan * default_value_units

        # --- Perform Interpolation ---
        try:
            if interp_type.lower() == 'log':
                if self.log_interpolate_1d_func:
                    logger.debug(f"Interpolate: Using MetPy log_interpolate_1d for target {target_q}.")
                    # log_interpolate_1d handles sorting internally, but expects positive x.
                    res = self.log_interpolate_1d_func(target_q, profile_coord_q_s_clean, value_profile_q_s_clean) # Use unsorted but NaN-cleaned
                    return res[0] if isinstance(res, (list, np.ndarray)) and len(res)>0 else res
                else: # Fallback to NumPy log interpolation
                    logger.debug(f"Interpolate: MetPy log_interpolate_1d not found. Using NumPy fallback for target {target_q}.")
                    return self._numpy_interp_with_units(target_q, profile_coord_q_sorted, value_profile_q_sorted, log_x=True)
            
            elif interp_type.lower() == 'linear':
                if self.interpolate_1d_func:
                    logger.debug(f"Interpolate: Using MetPy interpolate_1d for target {target_q}.")
                    # interpolate_1d requires x to be sorted and unique
                    unique_coords, unique_idx = np.unique(profile_coord_q_sorted.magnitude, return_index=True)
                    if len(unique_coords) < profile_coord_q_sorted.size: # Duplicates found
                        prof_coord_unique = profile_coord_q_sorted[unique_idx]
                        val_prof_unique = value_profile_q_sorted[unique_idx]
                        if prof_coord_unique.size < 2: return np.nan * default_value_units
                    else:
                        prof_coord_unique = profile_coord_q_sorted
                        val_prof_unique = value_profile_q_sorted

                    res = self.interpolate_1d_func(target_q, prof_coord_unique, val_prof_unique)
                    return res[0] if isinstance(res, (list, np.ndarray)) and len(res)>0 else res
                else: # Fallback to NumPy linear interpolation
                    logger.debug(f"Interpolate: MetPy interpolate_1d not found. Using NumPy fallback for target {target_q}.")
                    return self._numpy_interp_with_units(target_q, profile_coord_q_sorted, value_profile_q_sorted, log_x=False)
            else:
                logger.error(f"Interpolate: Unknown interpolation type '{interp_type}'.")
                return np.nan * default_value_units

        except Exception as e_interp: #skipcq: PYL-W0703
            logger.error(f"Interpolation execution error for target {target_q} (type: {interp_type}): {e_interp}\n{traceback.format_exc()}")
            return np.nan * default_value_units


    def get_calculator(self):
        """Returns the MetPy calculation module (mpcalc) if available."""
        return self.mpcalc if self.metpy_available else None

    def get_units(self):
        """Returns the MetPy units object if available."""
        return self.units if self.metpy_available else None
    
    def _manual_mean_wind(self,
                          h_prof_q: Any, # Quantity array of heights
                          u_prof_q: Any, # Quantity array of u-component
                          v_prof_q: Any, # Quantity array of v-component
                          depth_q: Optional[Any] = None, # Quantity for depth (e.g., 6000 * m)
                          bottom_agl_m_val: float = 0.0 # float in meters for bottom AGL
                          ) -> Tuple[Any, Any]:
        """
        Manually calculates mean wind. Inputs are expected to be MetPy Quantities.
        Returns (mean_u_q, mean_v_q) as Quantities or (nan*units, nan*units).
        """
        if not self.metpy_available or self.units is None:
            logger.warning("MetPy not available for manual_mean_wind.")
            # Try to return with some unit if possible, else just np.nan
            u_units_fallback = getattr(u_prof_q, 'units', self.units.mps if self.units else "mps_placeholder")
            v_units_fallback = getattr(v_prof_q, 'units', self.units.mps if self.units else "mps_placeholder")
            return np.nan * u_units_fallback, np.nan * v_units_fallback
            
        try:
            h_prof_m = h_prof_q.to(self.units.meter).m
            u_prof_mps = u_prof_q.to(self.units.mps).m
            v_prof_mps = v_prof_q.to(self.units.mps).m
            
            # Store original units for the output
            out_u_units = u_prof_q.units
            out_v_units = v_prof_q.units
        except Exception as e_conv: #skipcq: PYL-W0703
            logger.error(f"Error converting units for manual_mean_wind inputs: {e_conv}")
            return np.nan * self.units.mps, np.nan * self.units.mps # Fallback units

        if not (h_prof_m.size > 1 and u_prof_mps.size == h_prof_m.size and v_prof_mps.size == h_prof_m.size):
            logger.debug("Manual mean wind: Insufficient or mismatched data after unit conversion.")
            return np.nan * out_u_units, np.nan * out_v_units

        top_agl_m_val: float
        if depth_q is not None:
            top_agl_m_val = bottom_agl_m_val + depth_q.to(self.units.meter).m
        else: # If no depth, average over the whole profile above bottom
            # Ensure h_prof_m is not all NaNs before taking nanmax
            if np.all(np.isnan(h_prof_m)):
                logger.debug("Manual mean wind: All heights are NaN when calculating top_agl_m_val.")
                return np.nan * out_u_units, np.nan * out_v_units
            top_agl_m_val = np.nanmax(h_prof_m) 
        
        if pd.isna(top_agl_m_val):
            logger.debug(f"Manual mean wind: Calculated top_agl_m_val is NaN. Bottom: {bottom_agl_m_val}, Depth: {depth_q}")
            return np.nan * out_u_units, np.nan * out_v_units

        # Create mask based on height values in meters
        mask_man = (h_prof_m >= bottom_agl_m_val) & (h_prof_m <= top_agl_m_val)
        
        # Also consider NaNs in u/v components for the mask
        mask_man &= pd.notna(u_prof_mps) & pd.notna(v_prof_mps)
        
        if not np.any(mask_man): # Check if any True values in mask
            logger.debug(f"Manual mean wind: No data points in layer {bottom_agl_m_val}m - {top_agl_m_val}m AGL after NaN check.")
            return np.nan * out_u_units, np.nan * out_v_units

        u_masked_mps = u_prof_mps[mask_man]
        v_masked_mps = v_prof_mps[mask_man]

        # nanmean handles any remaining NaNs if some components were NaN but heights were valid
        mean_u_mps = np.nanmean(u_masked_mps)
        mean_v_mps = np.nanmean(v_masked_mps)

        if pd.isna(mean_u_mps) or pd.isna(mean_v_mps):
            logger.debug("Manual mean wind: Result of nanmean is NaN.")
            return np.nan * out_u_units, np.nan * out_v_units
            
        return mean_u_mps * out_u_units, mean_v_mps * out_v_units # Return with original units

# Create a single instance for the application to use
metpy_manager_instance = MetPyManager()