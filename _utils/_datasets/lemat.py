r"""Download LeMat-BulkUnique PBE as a symmetrised CIF parquet.

Source: https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique.
Uses a seeded random 5% validation split.

Usage:
    ```bash
    python _utils/_datasets/lemat.py \
        --output_parquet lematerial.parquet --num_workers 64
    ```
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import load_dataset
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils._datasets.common import _process_single_cif, pmap, save

LEMAT_REVISION = "99c34041e8cc4451bbad647de152c76073c6eb84"
LEMAT_COLUMNS = ["immutable_id", "lattice_vectors", "species_at_sites", "cartesian_site_positions",
                 "chemical_formula_reduced"]


def lemat_cif(row: dict) -> dict:
    """Convert a LeMat row to a symmetrised CIF."""
    struct = Structure(Lattice(row["lattice_vectors"]), row["species_at_sites"], row["cartesian_site_positions"],
                       coords_are_cartesian=True)
    return _process_single_cif((struct.to(fmt="cif"), row["immutable_id"], None))


def main() -> None:
    """Build the LeMat CIF parquet."""
    parser = argparse.ArgumentParser(description="Download LeMat-BulkUnique PBE as a symmetrised CIF parquet.")
    parser.add_argument("--revision", type=str, default=LEMAT_REVISION, help="LeMat-BulkUnique Hub commit")
    parser.add_argument("--val_percent", type=int, default=5, help="Percent of rows in the validation split")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the validation sample")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Where downloads are cached")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    lemat = load_dataset("LeMaterial/LeMat-BulkUnique", "unique_pbe", split="train", revision=args.revision,
                         cache_dir=os.path.join(args.raw_dir, "lemat"))
    lemat = lemat.select_columns(LEMAT_COLUMNS)

    results = pmap(lemat_cif, lemat, len(lemat), args.num_workers, "Symmetrising CIFs")
    df = pd.DataFrame({"material_id": lemat["immutable_id"], "reduced_formula": lemat["chemical_formula_reduced"],
                       "CIF": [r.get("CIF") for r in results]})

    # Rows come in the order of the pinned revision, so the seed fixes the split.
    df["Split"] = "train"
    df.loc[df.sample(frac=args.val_percent / 100, random_state=args.seed).index, "Split"] = "val"

    save(df, args.output_parquet)


if __name__ == "__main__":
    main()
