"""
Sequential debugging tool to isolate pipeline stalls in MACE/MP2020 calculations.
Executes each step in an isolated process with timeouts, loads MP data once, saves failing CIFs, and suppresses verbose third-party logs.
"""

import argparse
import multiprocessing
import pandas as pd
import numpy as np
import os
import sys
import warnings
import logging
import contextlib

# Suppress specific e3nn PyTorch warnings in the main process
warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")
# Silence verbose info logs from mace and cuequivariance in the main process
logging.getLogger("mace").setLevel(logging.WARNING)

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import MPDataProvider, download_mp_data

def run_with_timeout(func, args, timeout):
    # Runs a function in a separate process to safely kill C-level stalls
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(func, args)
    
    try:
        result = async_result.get(timeout=timeout)
        pool.close()
        pool.join()
        return True, result, None
    except multiprocessing.context.TimeoutError:
        pool.terminate()
        pool.join()
        return False, None, "Timeout"
    except Exception as e:
        pool.terminate()
        pool.join()
        return False, None, str(e)

def save_timeout_cif(cif_str, filename, out_dir):
    # Helper to save unphysical structures for later analysis
    filepath = os.path.join(out_dir, filename)
    try:
        with open(filepath, "w") as f:
            f.write(cif_str)
    except Exception as e:
        logging.error(f"Failed to save CIF to {filepath}: {e}")

# Isolated Pipeline Steps

def step_parse_pymatgen(cif_str):
    return Structure.from_str(cif_str, fmt="cif")

def step_parse_ase(structure):
    adaptor = AseAtomsAdaptor()
    return adaptor.get_atoms(structure)

def step_mace_energy(atoms):
    # Re-import warnings and os for the isolated child process environment
    import warnings
    import os
    import contextlib
    
    # Suppress PyTorch unpickling warnings in this specific worker
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Swallow MACE's hardcoded print statements by redirecting stdout to devnull
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        calc = mace_mp(default_dtype="float32", device="cpu")
        
    atoms.calc = calc
    return atoms.get_potential_energy()

def step_phase_diagram(structure, energy, mp_provider):
    # Calculate ehull using the pre-loaded provider object
    eh, eform_pa = mp_provider.compute_ehull_and_eform(structure, float(energy))
    return eh

def main():
    parser = argparse.ArgumentParser(description="Debug stalls in ehull pipeline")
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--cif_column", default="Generated CIF")
    parser.add_argument("--mp_data", default="data/mp_computed_structure_entries.json.gz")
    parser.add_argument("--test_limit", type=int, default=500, help="Number of structures to test")
    parser.add_argument("--log_file", default="__logs__/debug_stalls.log", help="File to save log outputs")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for each step")
    parser.add_argument("--timeout_dir", default="__logs__/timeout_cifs", help="Directory to save CIFs that cause stalls")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.timeout_dir, exist_ok=True)

    # Setup logger to write to both console and file simultaneously
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Downloading and organizing MP data provider. This may take a minute.")
    download_mp_data(args.mp_data)
    mp_provider = MPDataProvider(args.mp_data)
    logger.info("MP data provider loaded successfully.\n")

    df = pd.read_parquet(args.input_parquet)
    cifs_to_test = df[args.cif_column].head(args.test_limit).tolist()
    
    logger.info(f"Starting debug run on {len(cifs_to_test)} structures with {args.timeout}s timeout per step")
    logger.info(f"Failing structures will be saved to: {args.timeout_dir}\n")

    for idx, cif_str in enumerate(cifs_to_test):
        logger.info(f"Testing Structure {idx + 1}/{len(cifs_to_test)}")
        
        # Step 1: Pymatgen
        success, structure, err = run_with_timeout(step_parse_pymatgen, (cif_str,), args.timeout)
        if not success:
            logger.error(f"  [FAILED] Pymatgen Parsing: {err}\n")
            if err == "Timeout":
                save_timeout_cif(cif_str, f"timeout_struct_{idx}_pymatgen.cif", args.timeout_dir)
            continue
        logger.info("  [OK] Pymatgen Parsing")

        # Step 2: ASE
        success, atoms, err = run_with_timeout(step_parse_ase, (structure,), args.timeout)
        if not success:
            logger.error(f"  [FAILED] ASE Conversion: {err}\n")
            if err == "Timeout":
                save_timeout_cif(cif_str, f"timeout_struct_{idx}_ase.cif", args.timeout_dir)
            continue
        logger.info("  [OK] ASE Conversion")

        # Step 3: MACE
        success, energy, err = run_with_timeout(step_mace_energy, (atoms,), args.timeout)
        if not success:
            logger.error(f"  [FAILED] MACE Energy: {err}\n")
            if err == "Timeout":
                save_timeout_cif(cif_str, f"timeout_struct_{idx}_mace.cif", args.timeout_dir)
            continue
        logger.info(f"  [OK] MACE Energy: {energy:.3f} eV")

        # Step 4: Phase Diagram & e_hull
        success, ehull, err = run_with_timeout(step_phase_diagram, (structure, energy, mp_provider), args.timeout)
        if not success:
            logger.error(f"  [FAILED] Phase Diagram: {err}\n")
            if err == "Timeout":
                save_timeout_cif(cif_str, f"timeout_struct_{idx}_phasediagram.cif", args.timeout_dir)
            continue
        logger.info(f"  [OK] Phase Diagram ehull: {ehull:.3f}\n")

if __name__ == "__main__":
    # Required for safe multiprocessing behavior across OS platforms
    multiprocessing.set_start_method('spawn')
    main()