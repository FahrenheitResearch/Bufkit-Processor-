# sounding_processor/sounding_processor/bufkit_parser.py
"""
Parses BUFKIT text file content into a structured format.
"""
import re
import logging
import pandas as pd
import numpy as np
from io import StringIO # Not strictly needed if parsing line by line
from typing import List, Dict, Tuple, Any, Optional # Added Optional

from .sounding_data import Sounding # Using the dataclass
from . import config as app_config # For BUFKIT_MISSING_VALUES

logger = logging.getLogger(__name__)

class BufkitParser:
    """
    Parses BUFKIT text data into a list of Sounding objects.
    """
    def __init__(self):
        self.soundings_data_list: List[Sounding] = []
        self._current_sounding_dict_raw: Dict[str, Any] = {} # Temp dict for current STID block
        self._reading_level_data: bool = False
        self._reading_surface_summary_header: bool = False
        self._reading_surface_summary_data: bool = False
        
        self._snparm_columns_line1: List[str] = []
        self._snparm_columns_line2: List[str] = []
        self._full_snparm_columns: List[str] = []
        
        self._surface_summary_header_parts: List[str] = []
        self._surface_summary_columns: List[str] = []
        self._surface_summary_data_accumulated_values: List[str] = []
        self._surface_summary_data_lines_raw: List[str] = []

    def _reset_parser_state_for_new_sounding_block(self):
        """Resets variables specific to an individual sounding (STID) block."""
        self._current_sounding_dict_raw = {
            "metadata": {}, "summary_params": {}, "level_data_raw": [], "surface_summary": {}
        }
        self._reading_level_data = False
        self._snparm_columns_line1 = []
        self._snparm_columns_line2 = []
        self._full_snparm_columns = []
        logger.debug("Parser state reset for new sounding block.")

    def _reset_surface_summary_block_state(self):
        """Resets variables for the surface summary block (STN YYMMDD...)."""
        self._reading_surface_summary_header = False
        self._reading_surface_summary_data = False
        self._surface_summary_header_parts = []
        self._surface_summary_columns = []
        self._surface_summary_data_accumulated_values = []
        self._surface_summary_data_lines_raw = []
        logger.debug("Surface summary block state reset.")

    def _process_level_data_for_current_sounding(self):
        """
        Converts raw level data lines in _current_sounding_dict_raw
        into a pandas DataFrame and stores it in 'level_data'.
        """
        raw_lines = self._current_sounding_dict_raw.get('level_data_raw', [])
        if not raw_lines or not self._full_snparm_columns:
            logger.debug("No raw level data or SNPARM columns to process for current sounding.")
            self._current_sounding_dict_raw['level_data'] = pd.DataFrame()
            if 'level_data_raw' in self._current_sounding_dict_raw:
                 del self._current_sounding_dict_raw['level_data_raw']
            return

        parsed_levels = []
        for combined_line_text in raw_lines:
            parts = combined_line_text.split()
            row_dict = {}
            for j, col_name in enumerate(self._full_snparm_columns):
                if j < len(parts):
                    val_str = parts[j]
                    if val_str in app_config.BUFKIT_MISSING_VALUES:
                        row_dict[col_name] = np.nan
                    else:
                        try: row_dict[col_name] = float(val_str)
                        except ValueError: row_dict[col_name] = val_str
                else:
                    row_dict[col_name] = np.nan
            parsed_levels.append(row_dict)

        level_df = pd.DataFrame() # Default empty
        if parsed_levels:
            level_df = pd.DataFrame(parsed_levels)
            cols_to_numeric = [col for col in self._full_snparm_columns if col.upper() not in ['TYPE']]
            for col in cols_to_numeric:
                if col in level_df.columns:
                    level_df[col] = pd.to_numeric(level_df[col], errors='coerce')
        
        self._current_sounding_dict_raw['level_data'] = level_df
        logger.debug(f"Processed level data. DataFrame shape: {level_df.shape}")
        if 'level_data_raw' in self._current_sounding_dict_raw:
            del self._current_sounding_dict_raw['level_data_raw']


    def _finalize_current_sounding_block(self):
        """
        Finalizes the _current_sounding_dict_raw, creates a Sounding object,
        and adds it to self.soundings_data_list.
        """
        if self._current_sounding_dict_raw and self._current_sounding_dict_raw.get('metadata'):
            if 'level_data_raw' in self._current_sounding_dict_raw: # If raw lines still exist
                self._process_level_data_for_current_sounding()
            
            # Create Sounding object using the factory method
            sounding_obj = Sounding.from_parser_output(self._current_sounding_dict_raw)
            self.soundings_data_list.append(sounding_obj)
            logger.info(f"Finalized sounding block for STID: {sounding_obj.metadata.get('STID')}, TIME: {sounding_obj.metadata.get('TIME')}")
        
        # Reset for the next STID block, but keep surface summary stuff if it's being read
        self._reset_parser_state_for_new_sounding_block()


    def _process_and_merge_surface_summary_data(self):
        """
        Processes the accumulated surface summary lines into a DataFrame and
        merges the data with the corresponding Sounding objects in soundings_data_list.
        """
        if not self._surface_summary_data_lines_raw or not self._surface_summary_columns:
            logger.debug("No surface summary data lines or columns to process.")
            self._reset_surface_summary_block_state() # Important to reset even if no data
            return

        try:
            data_for_df = [line_str.split() for line_str in self._surface_summary_data_lines_raw]
            valid_data_for_df = [row for row in data_for_df if len(row) == len(self._surface_summary_columns)]

            if not valid_data_for_df:
                logger.warning("No valid data lines found for surface summary DataFrame.")
                self._reset_surface_summary_block_state()
                return

            surface_summary_df = pd.DataFrame(valid_data_for_df, columns=self._surface_summary_columns)
            for col in surface_summary_df.columns:
                if col not in ['STN', 'YYMMDD/HHMM']:
                    surface_summary_df[col] = pd.to_numeric(surface_summary_df[col], errors='coerce')
            
            surface_summary_df.replace(app_config.SURFACE_SUMMARY_NA_VALS, np.nan, inplace=True)
            logger.info(f"Processed surface summary block. DataFrame shape: {surface_summary_df.shape}")

            for snd_obj in self.soundings_data_list:
                meta = snd_obj.metadata
                stnm_to_match = meta.get('STNM') # STNM is usually like '72357'
                time_to_match = meta.get('TIME') # TIME is 'YYMMDD/HHMM'

                if stnm_to_match and time_to_match and \
                   'STN' in surface_summary_df.columns and \
                   'YYMMDD/HHMM' in surface_summary_df.columns:
                    try:
                        # Ensure types are compatible for comparison (both strings)
                        surface_summary_df['STN_str'] = surface_summary_df['STN'].astype(str)
                        stnm_to_match_str = str(stnm_to_match)

                        matched_rows = surface_summary_df[
                            (surface_summary_df['STN_str'] == stnm_to_match_str) &
                            (surface_summary_df['YYMMDD/HHMM'] == time_to_match)
                        ]
                        if not matched_rows.empty:
                            # Convert row to dict, replacing np.nan with None for easier handling later
                            summary_dict = matched_rows.iloc[0].replace({np.nan: None}).to_dict()
                            if 'STN_str' in summary_dict: del summary_dict['STN_str'] # Remove temp column
                            snd_obj.surface_summary = summary_dict
                            logger.debug(f"Merged surface summary for STNM: {stnm_to_match_str}, TIME: {time_to_match}")
                        # else: # No match is common if summary doesn't cover all STIDs/TIMEs
                        #     logger.debug(f"No surface summary match for STNM: {stnm_to_match_str}, TIME: {time_to_match}")
                    except Exception as e_merge: #skipcq: PYL-W0703
                        logger.error(f"Error merging surface summary for STNM {stnm_to_match}, TIME {time_to_match}: {e_merge}")
            if 'STN_str' in surface_summary_df.columns: # Clean up temp column
                del surface_summary_df['STN_str']

        except Exception as e_proc_surf: #skipcq: PYL-W0703
            logger.error(f"Major error processing surface summary block: {e_proc_surf}\n{traceback.format_exc()}")
        finally:
            self._reset_surface_summary_block_state() # Always reset after processing this block


    def parse(self, bufkit_text_content: str) -> List[Sounding]:
        """Main parsing method."""
        self.soundings_data_list = []
        self._reset_parser_state_for_new_sounding_block()
        self._reset_surface_summary_block_state()

        lines = bufkit_text_content.strip().split('\n')
        line_iterator = iter(lines)
        line_idx = -1

        # Main parsing loop
        while True:
            try:
                line_idx += 1; line_content = next(line_iterator); line = line_content.strip()
            except StopIteration: break # End of file
            if not line: continue # Skip empty lines

            # --- State: Reading Level Data for current STID block ---
            if self._reading_level_data:
                if re.search(r"^-?\d", line) and "=" not in line and \
                   not line.startswith("STID =") and not line.startswith("STN YYMMDD"):
                    data_line1_text = line
                    try:
                        data_line2_text = next(line_iterator).strip(); line_idx +=1
                        self._current_sounding_dict_raw['level_data_raw'].append(data_line1_text + " " + data_line2_text)
                    except StopIteration:
                        self._current_sounding_dict_raw['level_data_raw'].append(data_line1_text); break
                    continue
                else: # Level data block ended
                    self._process_level_data_for_current_sounding()
                    self._reading_level_data = False
                    # Fall through to process current line with other rules

            # --- State: Reading Surface Summary Header (STN YYMMDD block) ---
            if self._reading_surface_summary_header:
                # Surface summary data lines start with STN (can be text) and then date/time
                if not re.match(r"^\s*\S+\s+\d{6}/\d{4}", line): # Adjusted STN regex
                    self._surface_summary_header_parts.append(line)
                else: # This line is the first data line of surface summary
                    self._reading_surface_summary_header = False; self._reading_surface_summary_data = True
                    full_header_str = " ".join(self._surface_summary_header_parts)
                    self._surface_summary_columns = full_header_str.split()
                    self._surface_summary_data_accumulated_values = line.split() # Current line is data
                continue

            # --- State: Reading Surface Summary Data (STN YYMMDD block) ---
            if self._reading_surface_summary_data:
                if re.match(r"^\s*\S+\s+\d{6}/\d{4}", line): # New data line starts
                    if self._surface_summary_data_accumulated_values: # Save previous line
                        self._surface_summary_data_lines_raw.append(" ".join(self._surface_summary_data_accumulated_values))
                    self._surface_summary_data_accumulated_values = line.split() # Start new
                else: # Continuation of a multi-line data entry for one time step
                    self._surface_summary_data_accumulated_values.extend(line.split())
                continue

            # --- Keyword-based parsing for STID blocks & initiation of Surface Summary block ---
            if line.startswith("STID ="):
                if self._current_sounding_dict_raw.get('metadata'): # If a STID block was open
                    self._finalize_current_sounding_block() # Finalize it
                self._reset_parser_state_for_new_sounding_block() # Prepare for new STID block
                parts = line.split()
                try:
                    self._current_sounding_dict_raw['metadata']['STID'] = parts[parts.index("STID") + 2]
                    if "STNM" in parts: self._current_sounding_dict_raw['metadata']['STNM'] = parts[parts.index("STNM") + 2]
                    if "TIME" in parts: self._current_sounding_dict_raw['metadata']['TIME'] = parts[parts.index("TIME") + 2]
                except (IndexError, ValueError) as e: logger.warning(f"STID/STNM/TIME parse error: '{line}': {e}")
                continue

            if line.startswith("STN YYMMDD/HHMM PMSL"): # Start of a new Surface Summary block
                if self._current_sounding_dict_raw.get('metadata'): # Finalize any open STID block
                    self._finalize_current_sounding_block()
                # If we were reading level data (should have been processed by now, but as safeguard)
                if self._reading_level_data: self._process_level_data_for_current_sounding(); self._reading_level_data = False
                
                self._process_and_merge_surface_summary_data() # Process any previous summary block
                self._reset_surface_summary_block_state() # Reset for this new one
                self._reading_surface_summary_header = True
                self._surface_summary_header_parts = [line]
                continue

            # Inside a STID block (self._current_sounding_dict_raw should be active)
            if self._current_sounding_dict_raw:
                if "SLAT =" in line and "SLON =" in line and "SELV =" in line:
                    parts = line.split()
                    try:
                        self._current_sounding_dict_raw['metadata']['SLAT'] = float(parts[parts.index("SLAT") + 2])
                        self._current_sounding_dict_raw['metadata']['SLON'] = float(parts[parts.index("SLON") + 2])
                        self._current_sounding_dict_raw['metadata']['SELV'] = float(parts[parts.index("SELV") + 2])
                    except (IndexError, ValueError) as e: logger.warning(f"SLAT/SLON/SELV parse error: '{line}': {e}")
                    continue
                
                if "STIM =" in line:
                    try: self._current_sounding_dict_raw['metadata']['STIM'] = int(line.split("=")[1].strip())
                    except Exception as e: logger.warning(f"STIM parse error: '{line}': {e}") #skipcq: PYL-W0703
                    continue

                # Level data header lines for current STID block
                if self._current_sounding_dict_raw.get('metadata') and \
                   not line.startswith("SNPARM =") and not line.startswith("STNPRM =") and \
                   "PRES" in line and "TMPC" in line and "OMEG" in line: # First header line
                    snparm_cols_line1_candidate = line.split()
                    try:
                        line2_header_content = next(line_iterator).strip(); line_idx +=1
                        if "CFRL" in line2_header_content and "HGHT" in line2_header_content: # Second header line
                            self._snparm_columns_line1 = snparm_cols_line1_candidate
                            self._snparm_columns_line2 = line2_header_content.split()
                            self._full_snparm_columns = self._snparm_columns_line1 + self._snparm_columns_line2
                            self._reading_level_data = True
                        else: # Not the expected two-line header
                            logger.warning(f"Expected second level data header, got: '{line2_header_content}'. First line: '{line}'")
                            # Logic to "put back" line2_header_content would be complex here.
                            # The original script implies it might then try to parse `line` as summary.
                            # For now, this might lead to line2_header_content being skipped if it's not a new block start.
                    except StopIteration: break # EOF
                    continue

                # Summary parameters for current STID block (e.g., CAPE = 123.45)
                if "=" in line and not self._reading_level_data and \
                   not self._reading_surface_summary_header and \
                   not self._reading_surface_summary_data:
                    summary_param_match = re.findall(r"([A-Z0-9./]+)\s*=\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?|-+[.\d]*|\S+)", line)
                    if summary_param_match:
                        for key, value_str in summary_param_match:
                            key_clean = key.strip().replace("/", "_over_")
                            value_str_clean = value_str.strip()
                            try:
                                if value_str_clean in app_config.BUFKIT_MISSING_VALUES or \
                                   (value_str_clean.startswith("-") and all(c in "-." for c in value_str_clean[1:])):
                                    self._current_sounding_dict_raw['summary_params'][key_clean] = np.nan
                                else:
                                    self._current_sounding_dict_raw['summary_params'][key_clean] = float(value_str_clean)
                            except ValueError:
                                self._current_sounding_dict_raw['summary_params'][key_clean] = value_str_clean
                        continue
            # Line not processed by any specific rule
            # logger.debug(f"Line not processed by major rules: '{line}'")


        # --- End of Loop Processing ---
        if self._reading_level_data: # If EOF hit during level data of last STID block
            self._process_level_data_for_current_sounding()
        if self._current_sounding_dict_raw.get('metadata'): # Finalize any open STID block
            self._finalize_current_sounding_block()

        if self._reading_surface_summary_data and self._surface_summary_data_accumulated_values: # Finalize last line of summary
            self._surface_summary_data_lines_raw.append(" ".join(self._surface_summary_data_accumulated_values))
        if self._surface_summary_data_lines_raw: # Process the entire collected summary block
            self._process_and_merge_surface_summary_data()

        logger.info(f"BUFKIT parsing complete. Found {len(self.soundings_data_list)} soundings.")
        return self.soundings_data_list