"""
CIF preprocessing script for cleaning and normalizing crystalline structure data.

Processes CIF files by applying transformations like ordering disordered structures,
normalizing property columns, and adding atomic properties blocks.
"""

import argparse
import ast
import os
import warnings
from io import StringIO
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

from pymatgen.core import Structure, Composition
from pymatgen.io.cif import CifParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    extract_formula_units,
    replace_data_formula_with_nonreduced_formula,
    semisymmetrize_cif,
    add_atomic_props_block,
    round_numbers,
    remove_comments,
    order_or_round_cif,
    add_variable_brackets_to_cif,
    normalize_property_column
)

warnings.filterwarnings("ignore")

# Configuration constants
ROUND_TOL = 0.10
MANDATORY_ELEMENTS = True
CHUNK_SIZE = 1000
DECIMAL_PLACES = 4
OXI_DEFAULT = False  # Default value for oxidation states if not provided

def progress_listener(progress_queue, total):
    pbar = tqdm(total=total)
    processed_count = 0
    while True:
        message = progress_queue.get()
        if message is None:
            break
        processed_count += message
        pbar.update(message)
        if processed_count >= total:
            break
    pbar.close()


def augment_cif_chunk(chunk, oxi, progress_queue, make_ordered=False):
    """Process a chunk of CIF strings, applying various transformations."""
    results = []
    for (idx, cif_str) in chunk:
        try:
            if make_ordered:
                cif_str = order_or_round_cif(cif_str, 
                                             MANDATORY_ELEMENTS=MANDATORY_ELEMENTS, 
                                             ROUND_TOL=ROUND_TOL)
            
            if cif_str is None:
                continue

            formula_units = extract_formula_units(cif_str)
            if formula_units == 0:
                raise Exception("Formula units extraction failed")

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            cif_str = round_numbers(cif_str, decimal_places=DECIMAL_PLACES)
            cif_str = remove_comments(cif_str)
            cif_str = add_variable_brackets_to_cif(cif_str)
            results.append((idx, cif_str))
        except Exception:
            pass
        finally:
            progress_queue.put(1)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("--input_parquet", type=str,
                        help="Path to the input file. (Parquet format)")
    parser.add_argument("--output_parquet", "-o", action="store",
                        required=True,
                        help="Path to the output file. (Parquet format)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers to use for processing. Adding too many workers may slow down processing due to overhead.")
    parser.add_argument("--make_disordered_ordered", action="store_true",
                        help="Attempt to convert disordered structures to ordered ones before preprocessing (for COD XRD experiment).")
    parser.add_argument("--property_columns", type=str, default="[]",
                        help="List of property columns to normalize, e.g., \"['Bandgap (eV)', 'ehull']\". Default is empty list.")
    parser.add_argument("--property1_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the first property column.")
    parser.add_argument("--property2_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the second property column.")
    parser.add_argument("--property3_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the third property column.")

    args = parser.parse_args()

    input_fname = args.input_parquet
    output_fname = args.output_parquet
    num_workers = args.num_workers
    make_ordered = args.make_disordered_ordered

    print(f"Loading data from {input_fname} as Parquet with zstd compression...")
    dataframe = pd.read_parquet(input_fname)

    required_columns = {'CIF'}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(f"The input dataframe must contain the columns: {required_columns}")

    try:
        property_columns = ast.literal_eval(args.property_columns)
        if not isinstance(property_columns, list):
            raise ValueError
    except Exception:
        raise ValueError("property_columns must be a valid list, e.g., \"['Bandgap (eV)', 'ehull']\"")

    # make above more concise:
    normalisers = [
        args.property1_normaliser if i == 0 else
        args.property2_normaliser if i == 1 else
        args.property3_normaliser if i == 2 else
        "None"
        for i in range(len(property_columns))
    ]

    # Validate property columns exist in dataframe
    missing_props = [prop for prop in property_columns if prop not in dataframe.columns]
    if missing_props:
        print(f"Warning: Property columns not found in dataframe: {missing_props}")

    if property_columns:
        print("\nNormalizing property columns")
        for i, prop in enumerate(property_columns):
            norm_method = normalisers[i]
            if norm_method != "None":
                dataframe = normalize_property_column(dataframe, prop, norm_method)

    print("\nLets augment the CIFs now (parallelizing sometimes takes a min before speeding up")

    cifs = list(dataframe[['CIF']].itertuples(index=True, name=None))
    chunks = [cifs[i:i + CHUNK_SIZE] for i in range(0, len(cifs), CHUNK_SIZE)]
    total_cifs = len(cifs)

    print(f"Number of CIFs before preprocessing: {total_cifs}")

    manager = mp.Manager()
    progress_queue = manager.Queue()

    listener = mp.Process(target=progress_listener, args=(progress_queue, total_cifs))
    listener.start()

    print(f"Number of workers: {num_workers}")
    with mp.Pool(processes=num_workers) as pool:
        chunked_results = pool.starmap(
            augment_cif_chunk,
            [(chunk, OXI_DEFAULT, progress_queue, make_ordered) for chunk in chunks]
        )

    progress_queue.put(None)
    listener.join()

    processed_results = [res for sublist in chunked_results for res in sublist]

    for idx, cif_str in processed_results:
        dataframe.at[idx, 'CIF'] = cif_str

    print("Number of CIFs before filtering out bad ones: ", len(dataframe))
    dataframe = dataframe[dataframe['CIF'].str.startswith("data_", na=False)]
    print(f"Number of CIFs after filtering: {len(dataframe)}")

    dataframe.reset_index(drop=True, inplace=True)

    if os.path.dirname(output_fname) != "":
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    print(f"\nSaving updated dataframe to {output_fname} as Parquet with zstd compression...")
    dataframe.to_parquet(output_fname, compression='zstd')

    print("Preprocessing completed successfully.")
