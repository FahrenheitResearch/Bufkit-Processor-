# sounding_processor/config.py
"""
Configuration settings for the sounding processor application.
Paths are relative to the project root by default, or handled by main.py.
"""
import re
import logging
from pathlib import Path

# --- Project Root Path ---
# This assumes config.py is in sounding_processor/ which is one level down from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Logging Configuration ---
LOG_DIRECTORY_NAME = "debug_logs"
LOG_FILENAME = "sounding_processor_debug.txt"
# Full path will be constructed in main.py or a logging setup utility
# e.g., PROJECT_ROOT / "data" / "output" / LOG_DIRECTORY_NAME / LOG_FILENAME

LOG_LEVEL = logging.DEBUG  # Level for file logging
CONSOLE_LOG_LEVEL = logging.INFO # Level for console output

# --- File Paths ---
# Default input/output relative to the 'data' directory within project root
DEFAULT_INPUT_DIRECTORY_NAME = "input"
DEFAULT_OUTPUT_DIRECTORY_NAME = "output"

DEFAULT_INPUT_BUFKIT_FILENAME = "rap_okl.buf.txt" # Make sure this file exists in data/input
DEFAULT_OUTPUT_JSONL_FILENAME = "processed_soundings_default.jsonl"

# --- BUFKIT Specifics ---
BUFKIT_MISSING_VALUES = [
    "-9999.00", "-9999", "9999.00", "9999", "--.--",
    "99999", "99999.0", "-9999.0", "-9999.0000", "nan",
    "99999.00", "999.00"
]
# Used in parser for surface summary specifically, can be merged or kept separate
SURFACE_SUMMARY_NA_VALS = BUFKIT_MISSING_VALUES[:] # Make a copy

# --- Parameter Calculation Constants ---
STANDARD_P_LEVELS_HPA = [925, 850, 700, 500, 300, 250, 200] # For key level snapshots
ML_DEPTH_HPA = 100 # hPa, for Mixed Layer calculations
SRH_CALC_DEPTHS_M = {'0_1km': 1000, '0_3km': 3000} # meters AGL

# Effective inflow layer calculation defaults (if MetPy's is not used or as parameters)
EFFECTIVE_CAPE_THRESHOLD_J_KG = 100 * 1.0 # J/kg # Ensure float
EFFECTIVE_CIN_THRESHOLD_J_KG = -250 * 1.0 # J/kg # Ensure float

# --- MetPy Related ---
# These will be updated by MetPyManager at runtime
METPY_INTERPOLATE_1D_AVAILABLE = False
METPY_LOG_INTERPOLATE_1D_AVAILABLE = False

# Diagnostic settings
EARLY_DIAGNOSTICS_ENABLED = True # For the initial MetPy import check in MetPyManager

# --- Output settings ---
# Whether to include the raw level_data DataFrame in the final JSON (can be large)
# If False, it's processed and then removed from the dict before JSON serialization
SERIALIZE_LEVEL_DATA_DF = False

# sounding_processor/config.py

"""
Configuration for the sounding processor project,
including WFO identifiers, names, and locations.
"""

# Dictionary mapping WFO ID to City/Name
WFO_NAMES = {
    'ABQ': 'Albuquerque', 'ABR': 'Aberdeen', 'AKQ': 'Wakefield', 'ALY': 'Albany',
    'AMA': 'Amarillo', 'APX': 'Gaylord', 'ARX': 'La Crosse', 'BGM': 'Binghamton',
    'BIS': 'Bismarck', 'BMX': 'Birmingham', 'BOI': 'Boise', 'BOU': 'Denver',
    'BOX': 'Boston / Norton', 'BRO': 'Brownsville', 'BTV': 'Burlington', 'BUF': 'Buffalo',
    'BYZ': 'Billings', 'CAE': 'Columbia', 'CAR': 'Caribou', 'CHS': 'Charleston',
    'CLE': 'Cleveland', 'CRP': 'Corpus Christi', 'CTP': 'State College', 'CYS': 'Cheyenne',
    'DDC': 'Dodge City', 'DLH': 'Duluth', 'DMX': 'Des Moines', 'DTX': 'Detroit',
    'DVN': 'Quad Cities IA IL', 'EAX': 'Kansas City/Pleasant Hill', 'EKA': 'Eureka', 'EPZ': 'El Paso',
    'EWX': 'Austin/San Antonio', 'FFC': 'Peachtree City', 'FGF': 'Grand Forks', 'FGZ': 'Flagstaff',
    'FSD': 'Sioux Falls', 'FWD': 'Dallas/Fort Worth', 'GGW': 'Glasgow', 'GID': 'Hastings',
    'GJT': 'Grand Junction', 'GLD': 'Goodland', 'GRB': 'Green Bay', 'GRR': 'Grand Rapids',
    'GSP': 'Greenville/Spartanburg', 'GYX': 'Gray', 'HGX': 'Houston/Galveston', 'HNX': 'San Joaquin Valley/Hanford',
    'HUN': 'Huntsville', 'ICT': 'Wichita', 'ILM': 'Wilmington', 'ILN': 'Wilmington', # OH
    'ILX': 'Lincoln', 'IND': 'Indianapolis', 'IWX': 'Northern Indiana', 'JAN': 'Jackson',
    'JAX': 'Jacksonville', 'JKL': 'Jackson', # KY
    'KEY': 'Key West', 'LBF': 'North Platte', 'LCH': 'Lake Charles', 'LIX': 'New Orleans',
    'LKN': 'Elko', 'LMK': 'Louisville', 'LOT': 'Chicago', 'LOX': 'Los Angeles/Oxnard',
    'LSX': 'St Louis', 'LUB': 'Lubbock', 'LWX': 'Baltimore/Washington', 'LZK': 'Little Rock',
    'MAF': 'Midland/Odessa', 'MEG': 'Memphis', 'MFL': 'Miami', 'MFR': 'Medford',
    'MHX': 'Newport/Morehead City', 'MKX': 'Milwaukee/Sullivan', 'MLB': 'Melbourne', 'MOB': 'Mobile',
    'MPX': 'Twin Cities/Chanhassen', 'MQT': 'Marquette', 'MRX': 'Morristown', 'MSO': 'Missoula',
    'MTR': 'San Francisco', 'OAX': 'Omaha / Valley', 'OHX': 'Nashville', 'OKX': 'New York',
    'OTX': 'Spokane', 'OUN': 'Norman', 'PAFC': 'Anchorage', 'PAFG': 'Fairbanks',
    'PAH': 'Paducah', 'PAJK': 'Juneau', 'PBZ': 'Pittsburgh', 'PDT': 'Pendleton',
    'PGUM': 'Guam', 'PHFO': 'Honolulu', 'PHI': 'Mount Holly', 'PIH': 'Pocatello/Idaho Falls',
    'PQR': 'Portland', 'PSR': 'Phoenix', 'PUB': 'Pueblo', 'RAH': 'Raleigh',
    'REV': 'Reno', 'RIW': 'Riverton', 'RLX': 'Charleston', # WV
    'RNK': 'Blacksburg', 'SEW': 'Seattle', 'SGF': 'Springfield', 'SGX': 'San Diego',
    'SHV': 'Shreveport', 'SJT': 'San Angelo', 'SLC': 'Salt Lake City', 'STO': 'Sacramento',
    'TAE': 'Tallahassee', 'TBW': 'Tampa Bay Area / Ruskin', 'TFX': 'Great Falls', 'TJSJ': 'San Juan',
    'TOP': 'Topeka', 'TSA': 'Tulsa', 'TWC': 'Tucson', 'UNR': 'Rapid City',
    'VEF': 'Las Vegas'
}

# Dictionary mapping WFO ID to approximate Lat/Lon
# !! Verify these coordinates if precision is critical !!
WFO_LOCATIONS = {
    'ABQ': (35.04, -106.62), 'ABR': (45.45, -98.42), 'AKQ': (36.71, -76.99), 'ALY': (42.75, -73.80),
    'AMA': (35.23, -101.71), 'APX': (44.91, -84.70), 'ARX': (43.82, -91.19), 'BGM': (42.21, -75.98),
    'BIS': (46.77, -100.75), 'BMX': (33.37, -86.75), 'BOI': (43.57, -116.21), 'BOU': (39.99, -105.17),
    'BOX': (41.95, -71.12), 'BRO': (25.91, -97.42), 'BTV': (44.47, -73.15), 'BUF': (42.94, -78.73),
    'BYZ': (45.80, -108.54), 'CAE': (33.94, -81.12), 'CAR': (46.87, -68.02), 'CHS': (32.90, -80.04),
    'CLE': (41.41, -81.86), 'CRP': (27.77, -97.50), 'CTP': (40.85, -77.94), 'CYS': (41.15, -104.81),
    'DDC': (37.76, -100.02), 'DLH': (46.84, -92.21), 'DMX': (41.73, -93.75), 'DTX': (42.49, -83.47),
    'DVN': (41.61, -90.58), 'EAX': (38.81, -94.36), 'EKA': (40.98, -124.10), 'EPZ': (31.91, -106.78),
    'EWX': (29.72, -98.03), 'FFC': (33.36, -84.57), 'FGF': (47.96, -97.18), 'FGZ': (35.33, -111.67),
    'FSD': (43.73, -96.73), 'FWD': (32.82, -97.30), 'GGW': (48.21, -106.62), 'GID': (40.57, -98.32),
    'GJT': (39.12, -108.53), 'GLD': (39.37, -101.69), 'GRB': (44.49, -88.11), 'GRR': (42.89, -85.52),
    'GSP': (34.89, -82.22), 'GYX': (43.89, -70.25), 'HGX': (29.47, -95.08), 'HNX': (36.31, -119.63),
    'HUN': (34.65, -86.78), 'ICT': (37.65, -97.44), 'ILM': (34.27, -77.90), 'ILN': (39.42, -83.82), # OH
    'ILX': (39.84, -89.67), 'IND': (39.71, -86.28), 'IWX': (41.10, -85.70), 'JAN': (32.33, -90.08),
    'JAX': (30.50, -81.70), 'JKL': (37.59, -83.31), # KY
    'KEY': (24.56, -81.75), 'LBF': (41.13, -100.68), 'LCH': (30.12, -93.22), 'LIX': (30.34, -89.83),
    'LKN': (40.89, -115.72), 'LMK': (38.17, -85.74), 'LOT': (41.61, -88.09), 'LOX': (34.30, -119.12),
    'LSX': (38.76, -90.72), 'LUB': (33.65, -101.81), 'LWX': (39.08, -77.46), 'LZK': (34.83, -92.25),
    'MAF': (31.94, -102.19), 'MEG': (35.06, -89.98), 'MFL': (25.68, -80.41), 'MFR': (42.37, -122.87),
    'MHX': (34.77, -76.87), 'MKX': (42.95, -88.55), 'MLB': (28.11, -80.65), 'MOB': (30.68, -88.24),
    'MPX': (44.85, -93.56), 'MQT': (46.53, -87.55), 'MRX': (36.02, -83.40), 'MSO': (46.92, -114.08),
    'MTR': (36.61, -121.76), 'OAX': (41.32, -96.37), 'OHX': (36.25, -86.56), 'OKX': (40.86, -72.86),
    'OTX': (47.68, -117.63), 'OUN': (35.24, -97.47), 'PAFC': (61.17, -150.02), 'PAFG': (64.80, -147.87),
    'PAH': (37.06, -88.77), 'PAJK': (58.42, -134.60), 'PBZ': (40.53, -80.23), 'PDT': (45.69, -118.85),
    'PGUM': (13.48, 144.78), 'PHFO': (21.32, -157.93), 'PHI': (39.86, -74.81), 'PIH': (42.91, -112.59),
    'PQR': (45.59, -122.96), 'PSR': (33.43, -112.01), 'PUB': (38.29, -104.52), 'RAH': (35.87, -78.79),
    'REV': (39.57, -119.80), 'RIW': (43.06, -108.48), 'RLX': (38.31, -81.72), # WV
    'RNK': (37.20, -80.21), 'SEW': (47.45, -122.30), 'SGF': (37.23, -93.39), 'SGX': (32.83, -117.03),
    'SHV': (32.46, -93.83), 'SJT': (31.37, -100.49), 'SLC': (40.77, -111.96), 'STO': (38.61, -121.34),
    'TAE': (30.39, -84.36), 'TBW': (27.69, -82.40), 'TFX': (47.45, -111.38), 'TJSJ': (18.43, -66.00),
    'TOP': (39.07, -95.62), 'TSA': (36.19, -95.78), 'TWC': (32.12, -110.94), 'UNR': (44.14, -103.10),
    'VEF': (36.08, -115.15)
}

# List of AFOS Product PILs to fetch generically
AFOS_PILS_GENERIC = ["AFD", "LSR", "SPS", "SVR", "TOR", "FFW", "WSW"]

# Other constants
DEFAULT_MODEL = "RAP"
BUFKT_SAVE_EXTENSION = ".buf.txt" # Desired final extension
BUFKT_API_ENDPOINT = "/api/1/nws/bufkit.txt" # Use .txt format
AFOS_API_ENDPOINT = "/cgi-bin/afos/retrieve.py"
IEM_BASE_URL = "https://mesonet.agron.iastate.edu"

# Regex to find WMO header timestamp (e.g., FOUS43 KDMX 120555 -> 120555)
WMO_TIMESTAMP_REGEX = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+(\d{6})")