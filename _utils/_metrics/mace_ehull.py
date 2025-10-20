"""
Script to calculate energy above hull using MACE and MP2020 corrections,
inspired from https://github.com/facebookresearch/crystal-text-llm/blob/2b5d56ec95caf82e854e67274c027bffc2358542/e_above_hull.py
"""

import argparse
import tempfile
import numpy as np
import pandas as pd
import torch
import os
import requests
import gzip
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
import warnings
import sys

from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    MPDataProvider,
    download_mp_data,
)

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

def generate_CSE(structure, mace_energy):
    """Generate ComputedStructureEntry exactly like the reference script"""
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": mace_energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        clean=True,
    )

    # Return the final CSE
    return cse

def get_mace_energy(cif_str, calculator):
    """Get MACE energy for a CIF string using pre-loaded calculator"""
    try:
        # Convert CIF to structure to atoms
        structure = Structure.from_str(cif_str, fmt="cif")
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        # Use the pre-loaded calculator
        atoms.calc = calculator
        
        # Calculate energy
        energy = atoms.get_potential_energy()
        return energy, structure
    except Exception:
        return None, None

def process_batch(cif_batch, mp_provider, device_id):
    """Process a batch of CIF strings"""
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    # Load MACE calculator once per batch
    calculator = mace_mp(default_dtype="float32", device=device)
    
    results = []
    
    for cif_str in cif_batch:
        try:
            # Get MACE energy using the pre-loaded calculator
            energy, structure = get_mace_energy(cif_str, calculator)
            if energy is None or structure is None:
                results.append(np.nan)
                continue
            
            # Use the same logic as mace_ehull_copy.py
            eh, _ = mp_provider.compute_ehull_and_eform(structure, float(energy))
            results.append(eh)
            
        except Exception:
            results.append(np.nan)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate energy above hull using MACE")
    parser.add_argument("--post_parquet", required=True, 
                       help="Path to input .parquet file with CIF column")
    parser.add_argument("--output_parquet", required=True, 
                       help="Path to output .parquet file")
    parser.add_argument("--mp_data", default="mp_computed_structure_entries.json.gz",
                       help="Path to MP computed structure entries JSON.gz file (will download if missing)")
    parser.add_argument("--cif_column", default="Generated CIF", 
                       help="Name of CIF column in parquet file")
    parser.add_argument("--num_workers", type=int, default=2, 
                       help="Number of parallel processes")
    parser.add_argument("--batch_size", type=int, default=100, 
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Download MP data if it doesn't exist
    download_mp_data(args.mp_data)
    
    # Create efficient MP data provider (only builds phase diagrams as needed)
    print("Setting up MP data provider...")
    mp_provider = MPDataProvider(args.mp_data)
    
    # Load input data
    print("Loading input data...")
    df = pd.read_parquet(args.post_parquet)
    
    # Filter valid structures if column exists
    if "is_valid" in df.columns:
        valid_mask = df["is_valid"].astype(bool)
        df_valid = df[valid_mask].copy()
        print(f"Processing {len(df_valid)} valid structures out of {len(df)} total")
    else:
        df_valid = df.copy()
        print(f"Processing all {len(df_valid)} structures")
    
    cif_strings = df_valid[args.cif_column].tolist()
    
    # Pre-analyze chemical systems to give user an idea of what will be processed
    print("Analyzing chemical systems in input data...")
    unique_chemsys = set()
    for cif_str in tqdm(cif_strings[:100], desc="Sampling chemical systems"):  # Sample first 100
        try:
            structure = Structure.from_str(cif_str, fmt="cif")
            elements = tuple(sorted(str(el) for el in structure.composition.elements))
            unique_chemsys.add(elements)
        except Exception:
            continue
    
    print(f"Found {len(unique_chemsys)} unique chemical systems in sample.")
    
    # Split into batches
    batches = [cif_strings[i:i + args.batch_size] 
               for i in range(0, len(cif_strings), args.batch_size)]
    
    print(f"Processing {len(batches)} batches with {args.num_workers} workers...")
    
    # Process in parallel
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    results = Parallel(n_jobs=args.num_workers, prefer="processes")(
        delayed(process_batch)(batch, mp_provider, i % gpu_count)
        for i, batch in enumerate(tqdm(batches, desc="Processing batches"))
    )
    
    # Flatten results
    ehull_values = []
    for batch_results in results:
        ehull_values.extend(batch_results)
    
    # Add results to dataframe
    df["ehull_mace_mp"] = np.nan
    if "is_valid" in df.columns:
        df.loc[valid_mask, "ehull_mace_mp"] = ehull_values
    else:
        df["ehull_mace_mp"] = ehull_values
    
    # Print statistics
    valid_ehull = df["ehull_mace_mp"].dropna()
    print(f"\nResults for {len(valid_ehull)} valid calculations:")
    print(f"Mean e_above_hull: {df['ehull_mace_mp'].mean():.4f} eV/atom")
    print(f"Median e_above_hull: {df['ehull_mace_mp'].median():.4f} eV/atom")
    print(f"Min e_above_hull: {df['ehull_mace_mp'].min():.4f} eV/atom")
    print(f"Stable structures (e_above_hull <= 0.157 (0.1 + MAE of MACE)): {(valid_ehull <= 0.157).sum()}")
    
    # Save results
    df.to_parquet(args.output_parquet)
    print(f"Results saved to {args.output_parquet}")

if __name__ == "__main__":
    main()