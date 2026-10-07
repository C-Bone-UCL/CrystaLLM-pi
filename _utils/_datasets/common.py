"""Shared download, parallel processing and saving helpers."""

import concurrent.futures
import os
import zipfile

import pandas as pd
import requests
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


def fetch(url: str, raw_dir: str) -> str:
    """Download and extract the archive once. Return `raw_dir`."""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, os.path.basename(url))

    # A .part file prevents reusing incomplete downloads.
    if not os.path.exists(path):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            bar = tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(url)}", dynamic_ncols=True)
            with open(path + ".part", "wb") as f, bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    bar.update(f.write(chunk))
        os.replace(path + ".part", path)

    if not os.path.exists(path + ".extracted"):
        with zipfile.ZipFile(path) as z:
            z.extractall(raw_dir)
        open(path + ".extracted", "w").close()
    return raw_dir


def pmap(fn, items, total: int, num_workers: int, desc: str) -> list:
    """Apply `fn` in worker processes, preserving item order."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(tqdm(executor.map(fn, items, chunksize=64), total=total, desc=desc, dynamic_ncols=True))


def save(df: pd.DataFrame, output_parquet: str) -> None:
    """Save successful CIF rows and log failed IDs beside the output."""
    failed = df["CIF"].isna()
    with open(f"{output_parquet}.failed.txt", "w") as f:
        f.write("\n".join(df.loc[failed, "material_id"].astype(str)))

    df = df.loc[~failed, ["material_id", "reduced_formula", "CIF", "Split"]]
    df = df.rename(columns={"material_id": "Material ID", "reduced_formula": "Reduced Formula"})
    df.to_parquet(output_parquet, compression="zstd")
    print(f"Saved {len(df)} rows to {output_parquet} ({failed.sum()} failed): {df['Split'].value_counts().to_dict()}")
