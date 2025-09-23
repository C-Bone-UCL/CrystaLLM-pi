#!/usr/bin/env python3
"""
Computes E_hull and formation energy from DFT energies using MP reference data with optional novelty analysis.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    get_novelty, 
    extract_generated_formulas, 
    load_and_filter_training_data,
    safe_filename,
    get_density,
    download_mp_data,
    MPDataProvider
)

# E_hull clipping range in eV
EHULL_CLIP_RANGE = (-10.0, 10.0)

# Column names
CIF_COLUMN = "opt_cif"  # CSV column optimized structures
ENERGY_COLUMN = "Energy_eV"  # CSV column with total cell energy in eV from optimized structure
RESCALE_FROM_RAW = True  # Rescale energy based on raw_cif atom count when opt structure was reduced to a smaller primitive cell

# Structure matching tolerances for novelty
LTOL = 0.2  
STOL = 0.3
ANGLE_TOL = 5.0

# MP data file (where to load MP data from / will be downloaded if not there)
MP_DATA_FILE = "mp_computed_structure_entries.json.gz" 

def process_energy_rescaling(row):
    """Rescale DFT energy when optimized structure was reduced to primitive cell.
    
    The energy is always from the optimized structure, but sometimes the opt structure
    gets reduced to a primitive cell with fewer atoms. In those cases, we need to rescale
    the energy based on the original (raw) cell size to get the correct total energy.
    Returns parsed structure, rescaled energy, and atom count or None values if parsing fails.
    """
    energy_val = row[ENERGY_COLUMN]
    cif_str = row[CIF_COLUMN]
    
    # Basic validation
    if pd.isna(cif_str) or not str(cif_str).strip():
        return None, None, None
    
    try:
        energy_tot = float(energy_val)
        structure_opt = Structure.from_str(str(cif_str), fmt="cif")
        n_opt = len(structure_opt)
    except Exception:
        return None, None, None
    
    # Determine source atom count for potential rescaling
    n_src = n_opt
    if RESCALE_FROM_RAW:
        raw_cif_str = row.get("raw_cif", None)
        if raw_cif_str and isinstance(raw_cif_str, str) and raw_cif_str.strip():
            try:
                structure_raw = Structure.from_str(raw_cif_str, fmt="cif")
                n_src = len(structure_raw)
            except Exception:
                pass  # Fall back to n_opt if raw parsing fails
    
    # Skip if we can't determine atom counts
    if n_opt is None or n_src is None or n_src == 0:
        return None, None, None
    
    # Rescale energy if opt structure was reduced to primitive cell
    energy_rescaled = energy_tot * (n_opt / n_src)
     
    return structure_opt, energy_rescaled, n_opt


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_csv", required=True, help="Input CSV with columns: opt_cif, Energy_eV (plus any others).")
    parser.add_argument("--output_parquet", required=True, help="Output .parquet path.")
    
    parser.add_argument("--huggingface_dataset", type=str, default=None,
                        help="Hugging Face dataset path (e.g., 'org/name'). If provided, compute 'is_novel'.")
    parser.add_argument("--train_cif_column", type=str, default="CIF",
                        help="Column name in the HF training dataset that contains CIF strings (default: 'CIF').")
    parser.add_argument("--load_processed_data", type=str, default=None,
                        help="Path to preprocessed Hugging Face dataset (Parquet) to load instead of processing anew.")
    
    parser.add_argument("--density", action="store_true", default=False,
                        help="If set, compute density from CIFs and add to output DataFrame under 'Density (g/cm^3)' column.")
    parser.add_argument("--output_cif_dir", type=str, default=None,
                        help="If provided, write each row's opt_cif to this directory as individual .cif files.")
    
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for processing the Hugging Face training dataset.")

    args = parser.parse_args()

    # Download MP data if it doesn't exist
    download_mp_data(MP_DATA_FILE)
    
    # Create efficient MP data provider (only builds phase diagrams as needed)
    print("Setting up MP data provider...")
    mp_provider = MPDataProvider(MP_DATA_FILE)

    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # Validate required columns
    if CIF_COLUMN not in df.columns:
        raise KeyError(f"Missing CIF column '{CIF_COLUMN}' in CSV.")
    if ENERGY_COLUMN not in df.columns:
        raise KeyError(f"Missing energy column '{ENERGY_COLUMN}' in CSV.")

    # Initialize result storage
    ehull_values = []
    formation_energies = []
    structures = []  # Cache parsed structures

    # Process structures and compute thermodynamic properties
    print("Processing structures and computing E_hull and formation energy per atom...")
    for idx, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
        # Process energy rescaling and structure parsing
        structure, energy_rescaled, n_atoms = process_energy_rescaling(row)
        
        if structure is None:
            # Failed to process - append NaN values
            ehull_values.append(np.nan)
            formation_energies.append(np.nan)
            structures.append(None)
            continue

        # Compute hull and formation energy using MP references
        try:
            ehull_val, eform_pa = mp_provider.compute_ehull_and_eform(structure, energy_rescaled)
        except Exception:
            ehull_val, eform_pa = np.nan, np.nan

        # Clip E_hull to reasonable range
        if np.isfinite(ehull_val):
            ehull_val = float(np.clip(ehull_val, *EHULL_CLIP_RANGE))
            if ehull_val in EHULL_CLIP_RANGE:
                struct_name = df.at[idx, "Structure"] if "Structure" in df.columns else f"Entry {idx}"
                print(f"Clipped E_hull to: {ehull_val:.6f} eV for {struct_name}")

        ehull_values.append(ehull_val)
        formation_energies.append(eform_pa if np.isfinite(eform_pa) else np.nan)
        structures.append(structure)

    # Add computed columns to DataFrame
    df["dft_ehull"] = ehull_values
    df["dft_formation_energy_per_atom"] = formation_energies

    # Novelty calculation if comparison train set provided
    if args.huggingface_dataset:
        print("\nComputing Novelty vs Training Dataset")
        print(f"Loading training dataset from Hugging Face: {args.huggingface_dataset} (train split)")

        # Extract unique reduced formulas from generated structures
        gen_formulas = extract_generated_formulas(structures)
        
        # Load and filter training data (same approach as VUN_metrics.py)
        base_comps = load_and_filter_training_data(
            args.huggingface_dataset, 
            args.load_processed_data, 
            args.num_workers, 
            gen_formulas
        )

        # Compute novelty using the VUN_metrics approach
        df = get_novelty(
            df_gen=df,
            base_comps=base_comps,
            ref_cif_column=CIF_COLUMN, 
            ltol=LTOL,
            stol=STOL,
            angle_tol=ANGLE_TOL,
            structures=structures,
            workers=args.num_workers,
            is_unique_filter=False  # Don't filter by uniqueness since we don't have that column
        )

    # Density calculation (if flag)
    if args.density:
        print("\nComputing Density from CIFs")
        df['Density (g/cm^3)'] = df[CIF_COLUMN].apply(get_density)
        print("Density column 'Density (g/cm^3)' added to DataFrame.")

    # Print summary statistics
    finite_ehull = pd.to_numeric(df["dft_ehull"], errors="coerce")
    mask_stable = (finite_ehull <= 0.15) & (finite_ehull != EHULL_CLIP_RANGE[0])
    
    print(f"\nResults Summary:")
    print(f"Number of stable entries (E_hull <= 0.15 eV): {int(mask_stable.sum())}")
    
    valid_ehull = finite_ehull[(finite_ehull != EHULL_CLIP_RANGE[0]) & (finite_ehull != EHULL_CLIP_RANGE[1])]
    if not valid_ehull.empty:
        print("Summary statistics for E_hull (excluding clipped values):")
        print(valid_ehull.describe())

    # Export CIF files (if flag)
    if args.output_cif_dir:
        print(f"\nWriting CIF files to: {args.output_cif_dir}")
        os.makedirs(args.output_cif_dir, exist_ok=True)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Export CIFs", dynamic_ncols=True):
            cif_str = row[CIF_COLUMN]
            if pd.isna(cif_str) or not str(cif_str).strip():
                continue
            base_name = row["Structure"] if "Structure" in df.columns else f"entry_{idx}"
            fname = safe_filename(str(base_name)) + f"__{idx}.cif"
            out_path = os.path.join(args.output_cif_dir, fname)
            try:
                with open(out_path, "w") as f:
                    f.write(str(cif_str).rstrip() + "\n")
            except Exception:
                continue

    # Write output file
    out_dir = os.path.dirname(os.path.abspath(args.output_parquet))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(df), args.output_parquet)
    print(f"\nWrote Parquet file to: {args.output_parquet}")


if __name__ == "__main__":
    main()
