"""
Generates XRD patterns from CIF structures and creates condition vectors for ML training.
"""

import argparse
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from tqdm import tqdm

# Configuration constants
MAX_ENTRIES_TO_PROCESS = None  # Set to integer to limit processing, None for all
TOP_K_PEAKS = 20
THETA_MIN, THETA_MAX = 0, 90
INTENSITY_MIN, INTENSITY_MAX = 0, 100
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() - 1)  # Leave 1 CPU core free
CHUNK_SIZE = 50  # Smaller chunks for better load balancing

# Suppress pymatgen warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

def generate_xrd_pattern_worker(cif_str):
    """Worker function to generate XRD pattern for a single CIF string."""
    try:
        structure = Structure.from_str(cif_str, fmt='cif')
        xrd_calc = XRDCalculator(wavelength="CuKa")
        pattern = xrd_calc.get_pattern(structure)
        
        xrd_data = []
        for two_theta, intensity, hkls_list, d_hkl in zip(pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
            for hkl_dict in hkls_list:
                hkl = hkl_dict['hkl']
                xrd_data.append({
                    "two_theta": float(two_theta),
                    "intensity": float(intensity),
                    "hkl": tuple(hkl),
                    "d_hkl": float(d_hkl)
                })
        return xrd_data if xrd_data else None
    except Exception:
        return None


def process_chunk(cif_strings):
    """Process a chunk of CIF strings to compute XRD patterns."""
    return [generate_xrd_pattern_worker(cif_str) for cif_str in cif_strings]


def add_xrd_columns(df, num_workers, column_name='CIF'):
    """Adds XRD patterns to DataFrame using parallel processing."""
    # Initialize XRD column
    df['XRD'] = None
    df['XRD'] = df['XRD'].astype(object)

    # Determine how many entries to process
    total_rows = len(df)
    if MAX_ENTRIES_TO_PROCESS is not None:
        total_rows = min(total_rows, MAX_ENTRIES_TO_PROCESS)

    rows_to_process = df.index[:total_rows]
    cif_strings = [df.at[idx, column_name] for idx in rows_to_process]

    # Create smaller, more manageable chunks for better load balancing
    chunks = [cif_strings[i:i + CHUNK_SIZE] for i in range(0, len(cif_strings), CHUNK_SIZE)]
    
    # print(f"Processing {len(cif_strings)} structures in {len(chunks)} chunks using {num_workers} workers")

    with tqdm(total=total_rows, desc="Generating XRD patterns", unit="structures") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = []
            for result_chunk in executor.map(process_chunk, chunks):
                results.extend(result_chunk)
                pbar.update(len(result_chunk))

    # Assign results back to DataFrame
    for idx, xrd_pattern in zip(rows_to_process, results):
        df.at[idx, 'XRD'] = xrd_pattern

    return df

def parse_condition_vector_string(vector_str):
    """Parse condition vector string back to list of floats."""
    if vector_str is None or vector_str == "None":
        return None
    
    # Remove brackets and split by comma
    vector_str = str(vector_str).strip()
    if vector_str.startswith('[') and vector_str.endswith(']'):
        vector_str = vector_str[1:-1]
    
    try:
        return [float(x.strip()) for x in vector_str.split(',')]
    except (ValueError, AttributeError):
        return None

def compute_one_condition_vector_as_list(xrd):
    """Computes a single condition vector from an XRD pattern and returns as list."""
    if xrd is None:
        # Handle failed XRD computations
        thetas = [-100] * TOP_K_PEAKS
        ints = [-100] * TOP_K_PEAKS
    else:
        peaks = sorted(xrd, key=lambda d: d['intensity'], reverse=True)[:TOP_K_PEAKS]
        thetas = [d['two_theta'] for d in peaks]
        ints = [d['intensity'] for d in peaks]
        # Pad with -100 if fewer than TOP_K_PEAKS
        thetas += [-100] * (TOP_K_PEAKS - len(thetas))
        ints += [-100] * (TOP_K_PEAKS - len(ints))

    # Normalize and create condition vector
    # Scale theta values
    scaled_theta = [(t - THETA_MIN) / (THETA_MAX - THETA_MIN) if t != -100 else -100 for t in thetas]
    scaled_theta = [round(t, 3) for t in scaled_theta]

    # Scale intensity values
    scaled_int = [(i - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN) if i != -100 else -100 for i in ints]
    scaled_int = [round(i, 3) for i in scaled_int]

    # Combine into vector list
    vec = scaled_theta + scaled_int
    return vec

def compute_one_condition_vector(xrd):
    """Computes a single condition vector from an XRD pattern."""
    vec = compute_one_condition_vector_as_list(xrd)
    vector_str = "[" + ",".join(map(str, vec)) + "]"
    return vector_str

def compute_condition_vector(df):
    """Computes condition vectors from XRD patterns for ML training."""
    print("Computing condition vectors from XRD patterns")
    
    vector_strs = []
    for xrd in tqdm(df['XRD'], desc="Computing condition vectors", unit="structures"):
        vector_str = compute_one_condition_vector(xrd)
        vector_strs.append(vector_str)
    
    df['condition_vector'] = vector_strs
    print(f"Theta range: {THETA_MIN}-{THETA_MAX}, Intensity range: {INTENSITY_MIN}-{INTENSITY_MAX}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate XRD patterns from CIF structures")
    parser.add_argument("--input_parquet", type=str, required=True, 
                       help="Path to input database file (.pkl or .parquet)")
    parser.add_argument("--output_parquet", type=str, required=True, 
                       help="Path to save the processed database")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, 
                       help="Number of parallel workers")
    parser.add_argument("--column_name", type=str, default="CIF", 
                       help="Name of the column containing CIF strings")
    args = parser.parse_args()

    # Load database
    print(f"Loading database from {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} entries")
    if os.path.dirname(args.output_parquet):
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    # Process XRD patterns
    print("Processing XRD patterns for the entries")
    df = add_xrd_columns(df, args.num_workers, args.column_name)
    
    # Compute condition vectors
    df = compute_condition_vector(df)
    
    # Clean up - remove intermediate XRD column
    df = df.drop(columns=['XRD'])
    
    # Save results
    print(f"Saving results to {args.output_parquet}")
    # if 
    df.to_parquet(args.output_parquet, index=False)
    print("done")

if __name__ == "__main__":
    main()
