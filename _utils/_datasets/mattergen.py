r"""Download MatterGen's Alex-MP-20 or MP-20 as a symmetrised CIF parquet.

Source: https://github.com/microsoft/mattergen (`data-release/`).
Alex-MP-20 keeps its train/val splits. MP-20 also keeps its test split.

Usage:
    ```bash
    python _utils/_datasets/mattergen.py --dataset alex_mp_20 \
        --output_parquet alex_mp_20.parquet --num_workers 64
    ```
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils._datasets.common import _process_single_cif, fetch, pmap, save

MATTERGEN_URL = "https://media.githubusercontent.com/media/microsoft/mattergen/main/data-release/{}.zip"
DATASETS = {"alex_mp_20": "alex-mp/alex_mp_20", "mp_20": "mp-20/mp_20"}


def load_mattergen(name: str, raw_dir: str) -> pd.DataFrame:
    """Download dataset rows with their original splits."""
    folder = os.path.join(fetch(MATTERGEN_URL.format(DATASETS[name]), os.path.join(raw_dir, name)), name)
    splits = [s for s in ("train", "val", "test") if os.path.exists(os.path.join(folder, f"{s}.csv"))]
    return pd.concat([pd.read_csv(os.path.join(folder, f"{s}.csv")).assign(Split=s) for s in splits], ignore_index=True)


def main() -> None:
    """Build the MatterGen CIF parquet."""
    parser = argparse.ArgumentParser(description="Download Alex-MP-20 or MP-20 as a symmetrised CIF parquet.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True, help="Which MatterGen dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Where downloads are cached")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    df = load_mattergen(args.dataset, args.raw_dir)
    print(f"Loaded {len(df)} rows: {df['Split'].value_counts().to_dict()}")

    results = pmap(_process_single_cif, zip(df["cif"], df["material_id"], df["Split"]), len(df), args.num_workers, "Symmetrising CIFs")
    df["CIF"] = [r.get("CIF") for r in results]

    # MP-20 lacks reduced_formula, and NaN formulas are blank in the source CSV.
    if "reduced_formula" not in df:
        df["reduced_formula"] = None
    df["reduced_formula"] = df["reduced_formula"].fillna(pd.Series([r.get("Reduced Formula") for r in results]))

    save(df, args.output_parquet)


if __name__ == "__main__":
    main()
