r"""Extract and symmetrise CIFs from tarballs into a parquet dataset.

The CIF tarballs are converted to the parquet layout expected by the package. Split labels are
derived from the tarball filenames after removing their extensions, so filenames determine the
resulting split names.

From original repo: https://github.com/lantunes/CrystaLLM/blob/main/ARTIFACTS.md

Usage:
    ```bash
    python _utils/_preprocessing/cifs_zip_to_parquet.py --input_tarballs train.tar.gz val.tar.gz \
        --output_parquet cifs.parquet --num_workers 8
    ```
"""

import argparse
import concurrent.futures
import os
import tarfile
import warnings

# Suppress noisy warnings from pymatgen before importing it
warnings.filterwarnings("ignore")

import pandas as pd
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def _process_single_cif(payload):
    """Parse and attempt to symmetrise one CIF.

    The single-argument interface allows the function to be mapped across a
    ``ProcessPool``. It returns processed data or an error string when processing
    fails completely.
    """
    cif_string, material_id, current_split = payload
    
    try:
        parser = CifParser.from_str(cif_string)
        struct = parser.parse_structures()[0]
        
        # Use the parsed structure if spatial standardisation fails.
        try:
            sga = SpacegroupAnalyzer(struct)
            symm_struct = sga.get_symmetrized_structure()
            final_cif_str = str(CifWriter(symm_struct, symprec=0.1))
        except Exception:
            # Fallback to standard cif if symmetrization fails
            final_cif_str = struct.to(fmt="cif")
            
        formula = struct.composition.reduced_formula
        
        return {
            "Material ID": material_id,
            "Reduced Formula": formula,
            "CIF": final_cif_str,
            "Split": current_split,
            "error": None
        }
    except Exception as e:
        # Catch parsing errors so we don't crash the worker pool
        return {
            "Material ID": material_id,
            "error": str(e)
        }


def load_cifs(tarball_paths, database_name, num_workers):
    """Extract CIFs from tar archives and process them using a multiprocessing pool."""
    payloads = []
    
    # Process I/O sequentially: reading from a single tarball isn't process-safe.
    # We load strings into memory first to avoid heavy IPC bottlenecks later.
    for full_tar_path in tarball_paths:
        filename = os.path.basename(full_tar_path)
        current_split = filename.split('.')[0]
        
        print(f"\nUnpacking {full_tar_path} into memory...")
        with tarfile.open(full_tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    cif_bytes = tar.extractfile(member).read()
                    cif_string = cif_bytes.decode("utf-8")
                    material_id = os.path.splitext(os.path.basename(member.name))[0]
                    
                    payloads.append((cif_string, material_id, current_split))
                    
    results = []
    
    print(f"\nDistributing {len(payloads)} structures across {num_workers} workers...")
    
    # Map the CPU-bound tasks across the process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(_process_single_cif, payloads), total=len(payloads), desc="Processing CIFs"):
            if res.get("error"):
                # Skip files that completely fail to parse
                pass 
            else:
                results.append({
                    "Database": str(database_name), 
                    "Material ID": res["Material ID"], 
                    "Reduced Formula": res["Reduced Formula"], 
                    "CIF": res["CIF"], 
                    "Split": res["Split"]
                })
                
    df = pd.DataFrame(results)
    return df
    

def main():
    parser = argparse.ArgumentParser(description="Process Benchmark CIFs and save to df")
    parser.add_argument("--input_tarballs", type=str, nargs='+', required=True, help="Path(s) to the specific input tar.gz file(s). eg. --input_tarballs /data/train.tar.gz /data/val.tar.gz")
    parser.add_argument("--output_parquet", type=str, required=True, help="Path to save the updated dataframe.")
    parser.add_argument("--database_name", type=str, required=True, help="Name of the database to populate the Database column.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for CPU-bound processing.")

    args = parser.parse_args()

    # Load the database
    print("\nProcessing CIFs for the entries...")
    df = load_cifs(args.input_tarballs, args.database_name, args.num_workers)

    # Save the database to parquet format
    df.to_parquet(args.output_parquet, engine='pyarrow', compression='zstd')

    print(f"\nDataframe saved to {args.output_parquet}")
    print(f"\nTotal entries processed: {len(df)}")
    # print(f"\nSplit distribution:\n{df['Split'].value_counts()}\n")


if __name__ == "__main__":
    main()