r"""Download stratified CHILI-100K as a symmetrised CIF parquet.

Source: https://github.com/UlrikFriisJensen/CHILI.
Uses the splits from `c-bone/chili100k_strat`.
Test: 500 seen, 500 structurally novel and 500 compositionally novel structures.
Validation: 1,500 seen structures. Novelty is measured against LeMaterial (Alex-MP-20 is in LeMaterial).

Split construction:
https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/X_XRD_chili100k.ipynb

Usage:
    ```bash
    python _utils/_datasets/chili.py \
        --output_parquet chili100k.parquet --num_workers 64
    ```
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import pandas as pd
from datasets import load_dataset
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils._datasets.common import _process_single_cif, fetch, pmap, save

CHILI_URL = "https://erda.ku.dk/archives/c9d91863f89c3a7e87201c175ff4b213/Nanostructure_Data/Data_for_MachineLearning/DatasetPaper/CHILI-100K.zip"

SPLIT_DATASET = "c-bone/chili100k_strat"
SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def chili_cif(path: str) -> dict:
    """Convert a CHILI HDF5 structure to a symmetrised CIF."""
    material_id = os.path.basename(path).removesuffix(".h5")
    try:
        with h5py.File(path, "r") as h5f:
            # Atomic numbers are in NodeFeatures column 0.
            struct = Structure(Lattice.from_parameters(*h5f["GlobalLabels"]["CellParameters"][:]),
                               [int(row[0]) for row in h5f["UnitCellGraph"]["NodeFeatures"][:]],
                               h5f["UnitCellGraph"]["FractionalCoordinates"][:])
    except Exception:
        return {"material_id": material_id, "CIF": None}

    result = _process_single_cif((struct.to(fmt="cif"), material_id, None))
    return {"material_id": material_id, "reduced_formula": result.get("Reduced Formula"), "CIF": result.get("CIF")}


def main() -> None:
    """Build the CHILI-100K CIF parquet."""
    parser = argparse.ArgumentParser(description="Download stratified CHILI-100K as a symmetrised CIF parquet.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Where downloads are cached")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    strat = load_dataset(SPLIT_DATASET, cache_dir=os.path.join(args.raw_dir, "chili"))
    split_of = {mid: SPLIT_NAMES[s] for s in strat for mid in strat[s]["Material ID"]}

    paths = glob.glob(os.path.join(fetch(CHILI_URL, os.path.join(args.raw_dir, "chili")), "**", "*.h5"), recursive=True)
    paths = sorted(p for p in paths if os.path.basename(p).removesuffix(".h5") in split_of)
    print(f"{len(paths)} of {len(split_of)} split structures found in the archive")

    df = pd.DataFrame(pmap(chili_cif, paths, len(paths), args.num_workers, "Symmetrising CIFs"))
    df["Split"] = df["material_id"].map(split_of)
    save(df, args.output_parquet)


if __name__ == "__main__":
    main()
