r"""Download LeMat-BulkUnique PBE as a symmetrised CIF parquet.

Source: https://huggingface.co/datasets/LeMaterial/LeMat-BulkUnique.
Excludes Alex-MP-20 validation materials by ID and BAWL fingerprint.
Uses a fixed 5% validation split based on material IDs.
Install the `metrics` extra with `uv sync --extra metrics` for BAWL fingerprint matching.

Usage:
    ```bash
    python _utils/_datasets/lemat.py \
        --output_parquet lematerial.parquet --num_workers 64
    ```
"""

import argparse
import hashlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import load_dataset
from pymatgen.core import Lattice, Structure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils._datasets.common import _process_single_cif, pmap, save
from _utils._datasets.mattergen import load_mattergen

LEMAT_REVISION = "99c34041e8cc4451bbad647de152c76073c6eb84"
LEMAT_COLUMNS = ["immutable_id", "lattice_vectors", "species_at_sites", "cartesian_site_positions",
                 "chemical_formula_reduced", "entalpic_fingerprint"]


def fingerprint(cif: str) -> str:
    """Return the BAWL fingerprint stored as `entalpic_fingerprint`."""
    warnings.filterwarnings("ignore")
    from material_hasher.hasher.bawl import BAWLHasher
    return BAWLHasher().get_material_hash(Structure.from_str(cif, fmt="cif"))


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
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Where downloads are cached")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    # Fingerprints catch materials stored under different database IDs.
    held = load_mattergen("alex_mp_20", args.raw_dir).query("Split == 'val'")
    held_ids = set(held["material_id"].str.replace(r"^alex<(.*)>$", r"\1", regex=True))
    held_fps = set(pmap(fingerprint, held["cif"], len(held), args.num_workers, "Fingerprinting held-out"))

    lemat = load_dataset("LeMaterial/LeMat-BulkUnique", "unique_pbe", split="train", revision=args.revision,
                         cache_dir=os.path.join(args.raw_dir, "lemat"))
    lemat = lemat.select_columns(LEMAT_COLUMNS)
    by_id = [i in held_ids for i in lemat["immutable_id"]]
    by_fp = [fp in held_fps for fp in lemat["entalpic_fingerprint"]]
    keep = [i for i, (a, b) in enumerate(zip(by_id, by_fp)) if not (a or b)]
    print(f"LeMat rows {len(lemat)}, excluded by ID {sum(by_id)}, by fingerprint only {sum(b and not a for a, b in zip(by_id, by_fp))}")
    lemat = lemat.select(keep)

    results = pmap(lemat_cif, lemat, len(lemat), args.num_workers, "Symmetrising CIFs")
    df = pd.DataFrame({"material_id": lemat["immutable_id"], "reduced_formula": lemat["chemical_formula_reduced"],
                       "CIF": [r.get("CIF") for r in results]})

    # ID hashing keeps the validation split independent of row order.
    df["Split"] = ["val" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 100 < args.val_percent else "train"
                   for i in df["material_id"]]

    save(df, args.output_parquet)


if __name__ == "__main__":
    main()
