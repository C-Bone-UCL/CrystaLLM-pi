"""
Evaluates generated CIF structures for validity, consistency, and crystallographic properties.
Feed in generated cifs directly (dont need to postprocess)
Option to save the valid ones to a new parquet file
"""

import argparse
import tarfile
import queue
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Global configuration constants
LENGTH_LO = 0.5      # Smallest cell length for sensibility check
LENGTH_HI = 1000.0   # Largest cell length for sensibility check  
ANGLE_LO = 10.0      # Smallest cell angle for sensibility check
ANGLE_HI = 170.0     # Largest cell angle for sensibility check
DEFAULT_TOKENIZER_DIR = "HF-cif-tokenizer"  # Default tokenizer directory

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _utils import (
    is_sensible,
    extract_space_group_symbol,
    replace_symmetry_operators,
    is_atom_site_multiplicity_consistent,
    is_space_group_consistent,
    bond_length_reasonableness_score,
    extract_numeric_property,
    get_unit_cell_volume,
    extract_volume,
    extract_data_formula,
    is_valid,
)

warnings.filterwarnings("ignore")

def read_generated_cifs(input_path):
    """Read CIFs from parquet or tar.gz files."""
    if input_path.endswith(".tar.gz"):
        generated_cifs = []
        with tarfile.open(input_path, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting generated CIFs..."):
                f = tar.extractfile(member)
                if f is not None:
                    cif = f.read().decode("utf-8")
                    generated_cifs.append(cif)
        df = pd.DataFrame(generated_cifs, columns=["Generated CIF"])
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    return df

def progress_listener(queue, n):
    """Track progress of CIF evaluation."""
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break

def eval_cif(progress_queue, task_queue, result_queue, tokenizer):
    """Evaluate a CIF for validity and other properties."""
    # Initialize counters and result containers
    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []
    is_valid_and_len = []
    valid_cifs = []

    while not task_queue.empty():
        try:
            i, cif = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            # Check if the CIF is sensible using global constants
            if not is_sensible(cif, LENGTH_LO, LENGTH_HI, ANGLE_LO, ANGLE_HI):
                raise Exception("CIF not sensible")

            # Tokenize the CIF and get its length
            gen_len = len(tokenizer.encode(cif))

            # Extract and replace symmetry operators if necessary
            space_group_symbol = extract_space_group_symbol(cif)
            if DEBUG:
                print(f"Space group symbol: {space_group_symbol}")
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)

            # Check atom site multiplicity consistency
            if is_atom_site_multiplicity_consistent(cif):
                n_atom_site_multiplicity_consistent += 1
                if DEBUG:
                    print("Atom site multiplicity consistent")
            else:
                if DEBUG:
                    print("Atom site multiplicity NOT consistent")

            # Check space group consistency
            if is_space_group_consistent(cif):
                n_space_group_consistent += 1
                if DEBUG:
                    print("Space group consistent")
            else:
                if DEBUG:
                    print("Space group NOT consistent")

            # Calculate bond length reasonableness score
            score = bond_length_reasonableness_score(cif)
            bond_length_reasonableness_scores.append(score)
            if DEBUG:
                print(f"Bond length reasonableness score: {score}")

            # Extract cell parameters and calculate volumes
            cell_params = ['_cell_length_a', '_cell_length_b', '_cell_length_c', 
                          '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']
            a, b, c, alpha, beta, gamma = [extract_numeric_property(cif, param) for param in cell_params]
            implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
            gen_vol = extract_volume(cif)
            data_formula = extract_data_formula(cif)

            # Check if the CIF is valid
            valid = is_valid(cif, bond_length_acceptability_cutoff=1.0, debug=False)
            if DEBUG:
                print(f"Is valid: {valid}")

            # Record the results
            is_valid_and_len.append((data_formula, space_group_symbol, valid, gen_len, implied_vol, gen_vol))

            # If valid, store for later saving
            if valid:
                valid_cifs.append((i, cif))

        except Exception as e:
            if DEBUG:
                print(f"ERROR in CIF {i}: {e}")
                print(f"Line: {sys.exc_info()[-1].tb_lineno}")
            
        progress_queue.put(1)

    # Return aggregated results
    result = (
        n_atom_site_multiplicity_consistent,
        n_space_group_consistent,
        bond_length_reasonableness_scores,
        is_valid_and_len,
        valid_cifs
    )
    result_queue.put(result)

def aggregate_results(result_queue):
    """Aggregate results from all worker processes."""
    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []
    is_valid_and_lens = []
    valid_cif_indices = []

    while not result_queue.empty():
        (n_atom_site_occ, n_space_group, scores, is_valid_and_len, valid_cifs) = result_queue.get()
        n_atom_site_multiplicity_consistent += n_atom_site_occ
        n_space_group_consistent += n_space_group
        bond_length_reasonableness_scores.extend(scores)
        is_valid_and_lens.extend(is_valid_and_len)
        valid_cif_indices.extend(valid_cifs)
    
    return (n_atom_site_multiplicity_consistent, n_space_group_consistent, 
            bond_length_reasonableness_scores, is_valid_and_lens, valid_cif_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated structures.")
    parser.add_argument("--input_parquet",
                        help="Path to parquet file containing generated CIF structures")
    parser.add_argument("--metrics_out", "-o", action="store",
                        required=False,
                        help="Output path for metrics parquet file")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of parallel workers")
    parser.add_argument("--save_valid_parquet", required=False, default=None,
                        help="Save valid CIFs to this parquet file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output during evaluation (see whats wrong with the generated CIFs)")

    args = parser.parse_args()
    DEBUG = args.debug

    # Load and process CIFs
    df = read_generated_cifs(args.input_parquet)
    cifs = df["Generated CIF"].tolist()
    print(f"Loaded {len(cifs)} CIFs from {args.input_parquet}")

    # Initialize tokenizer once
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=DEFAULT_TOKENIZER_DIR,
        pad_token="<pad>"
    )

    # Setup multiprocessing
    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Populate task queue
    n = len(cifs)
    for i, cif in enumerate(cifs):
        task_queue.put((i, cif))

    # Start progress watcher and worker processes
    watcher = mp.Process(target=progress_listener, args=(progress_queue, n,))
    processes = [
        mp.Process(
            target=eval_cif,
            args=(progress_queue, task_queue, result_queue, tokenizer)
        ) for _ in range(args.num_workers)
    ]
    processes.append(watcher)

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    finally:
        # Ensure all processes are cleaned up
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

    # Aggregate results from all workers
    (n_atom_site_multiplicity_consistent, n_space_group_consistent, 
     bond_length_reasonableness_scores, is_valid_and_lens, valid_cif_indices) = aggregate_results(result_queue)

    # Process results and generate summary statistics
    n_valid = 0
    valid_gen_lens = []
    results_data = {
        "comp": [],
        "sg": [],
        "is_valid": [],
        "gen_len": [],
        "implied_vol": [],
        "gen_vol": [],
    }

    for comp, sg, valid, gen_len, implied_vol, gen_vol in is_valid_and_lens:
        if valid:
            n_valid += 1
            valid_gen_lens.append(gen_len)
        results_data["comp"].append(comp)
        results_data["sg"].append(sg)
        results_data["is_valid"].append(valid)
        results_data["gen_len"].append(gen_len)
        results_data["implied_vol"].append(implied_vol)
        results_data["gen_vol"].append(gen_vol)

    # Print summary statistics
    print(f"\nspace group consistent: {n_space_group_consistent}/{n} ({n_space_group_consistent / n:.3f})")
    print(f"atom site multiplicity consistent: {n_atom_site_multiplicity_consistent}/{n} ({n_atom_site_multiplicity_consistent / n:.3f})")
    
    if bond_length_reasonableness_scores:
        print(f"avg. bond length reasonableness score: {np.mean(bond_length_reasonableness_scores):.4f} ± {np.std(bond_length_reasonableness_scores):.4f}")
        print(f"bond lengths reasonable: {bond_length_reasonableness_scores.count(1.0)}/{n} ({bond_length_reasonableness_scores.count(1.0) / n:.3f})")
    
    print(f"\nnum valid: {n_valid}/{n} ({n_valid / n:.2f})")
    
    if valid_gen_lens:
        print(f"longest valid generated length: {np.max(valid_gen_lens):,}")
        print(f"avg. valid generated length: {np.mean(valid_gen_lens):.3f} ± {np.std(valid_gen_lens):.3f}")
    else:
        print("No valid CIFs found")

    # Save results to parquet file if specified
    if args.metrics_out is not None:
        results_df = pd.DataFrame(results_data)
        results_df.to_parquet(args.metrics_out, index=False)

    # Optionally save valid CIFs as a new DataFrame
    if args.save_valid_parquet is not None:
        valid_indices = [idx for idx, _ in valid_cif_indices]
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        valid_df.to_parquet(args.save_valid_parquet, index=False)
        print(f"Valid CIFs have been saved to {args.save_valid_parquet}")