# sounding_processor/sounding_data.py
"""
Defines the data structures for holding sounding information.
Primarily, the Sounding dataclass.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List # Added List
import pandas as pd
import numpy as np

@dataclass
class Sounding:
    """
    Represents a single atmospheric sounding.
    """
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary_params: Dict[str, Any] = field(default_factory=dict) # From BUFKIT summary block
    level_data: Optional[pd.DataFrame] = None # Parsed level data as DataFrame
    surface_summary: Dict[str, Any] = field(default_factory=dict) # Parsed surface summary lines

    # Derived parameter sections
    parcel_thermodynamics: Dict[str, Any] = field(default_factory=dict)
    detailed_kinematics: Dict[str, Any] = field(default_factory=dict)
    environmental_thermodynamics: Dict[str, Any] = field(default_factory=dict)
    composite_indices: Dict[str, Any] = field(default_factory=dict)
    inversion_characteristics: Dict[str, Any] = field(default_factory=dict)
    key_levels_data: Dict[str, Any] = field(default_factory=dict)

    # Raw data from parser, potentially useful for debugging or reprocessing
    level_data_raw_lines: List[str] = field(default_factory=list) # Store raw text lines if needed

    # Flag to indicate if derived parameters have been calculated
    derived_params_calculated: bool = False

    def to_dict(self, serialize_df: bool = False) -> Dict[str, Any]:
        """
        Converts the Sounding object to a dictionary.
        If serialize_df is False, DataFrames are replaced by a placeholder.
        """
        # Manually construct dict to handle potential None DataFrames
        data_dict = {
            "metadata": self.metadata,
            "summary_params": self.summary_params,
            "surface_summary": self.surface_summary,
            "parcel_thermodynamics": self.parcel_thermodynamics,
            "detailed_kinematics": self.detailed_kinematics,
            "environmental_thermodynamics": self.environmental_thermodynamics,
            "composite_indices": self.composite_indices,
            "inversion_characteristics": self.inversion_characteristics,
            "key_levels_data": self.key_levels_data,
            "derived_params_calculated": self.derived_params_calculated
            # level_data_raw_lines is not typically part of the final export
        }
        if self.level_data is not None and isinstance(self.level_data, pd.DataFrame):
            if serialize_df:
                # Convert DataFrame to dict for JSON (list of records), handle NaNs
                # Replace np.nan with None for JSON compatibility
                df_for_json = self.level_data.copy()
                for col in df_for_json.select_dtypes(include=[np.number]): # Only for numeric types
                    df_for_json[col] = df_for_json[col].apply(lambda x: None if pd.isna(x) else x)
                data_dict["level_data"] = df_for_json.to_dict(orient='records')
            else:
                data_dict["level_data"] = "DataFrame object - Not Serialized in this export"
        else:
            data_dict["level_data"] = None # Explicitly None if no DataFrame

        return data_dict

    @classmethod
    def from_parser_output(cls, parsed_sounding_dict: Dict[str, Any]) -> 'Sounding':
        """
        Factory method to create a Sounding object from the dictionary
        structure produced by BufkitParser.
        This provides a cleaner way to instantiate from raw parser output.
        """
        # Extract known fields from the parsed_sounding_dict
        metadata = parsed_sounding_dict.get('metadata', {})
        summary_params = parsed_sounding_dict.get('summary_params', {})
        level_data_df = parsed_sounding_dict.get('level_data') # Should be a DataFrame or None
        surface_summary = parsed_sounding_dict.get('surface_summary', {})
        level_data_raw = parsed_sounding_dict.get('level_data_raw', []) # From parser output before DF

        # Ensure level_data is a DataFrame, even if empty, if it's supposed to be one
        if level_data_df is None and 'level_data' in parsed_sounding_dict: # Key exists but is None
            level_data_df = pd.DataFrame()
        elif level_data_df is not None and not isinstance(level_data_df, pd.DataFrame):
            # This case should ideally not happen if parser behaves
            level_data_df = pd.DataFrame()


        return cls(
            metadata=metadata,
            summary_params=summary_params,
            level_data=level_data_df,
            surface_summary=surface_summary,
            level_data_raw_lines=level_data_raw
            # Derived parameters dicts will be default_factory empty dicts
        )