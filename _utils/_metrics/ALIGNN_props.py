#!/usr/bin/env python

"""
ALIGNN bandgap prediction script with parallel CIF parsing for bulk structure analysis.
Inspired by: https://github.com/usnistgov/alignn/blob/develop/alignn/pretrained.py
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignn.dataset import get_torch_dataset
from alignn.graphs import Graph
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from jarvis.core.atoms import Atoms

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

tqdm.pandas()

# Suppress warnings
warnings.filterwarnings("ignore")

# These are the models available, with figshare links
# There are more models available at the ALIGNN github
all_models = {
    "mp_e_form_alignn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "jv_ehull_alignn": [
        "https://figshare.com/ndownloader/files/31458658",
        1,
    ],
}

def get_all_models():
    """Return the figshare links for models."""
    return all_models

# Determine device
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

def get_figshare_model(model_name="mp_gappbe_alignn"):
    """Get ALIGNN torch models from figshare."""
    tmp = all_models[model_name]
    url = tmp[0]
    zfile = "alignn_model_ckpts/"+ model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KiB
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    cfg = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            chks.append(i)
        if "config.json" in i:
            cfg = i
        if "best_model.pt" in i:
            chks.append(i)

    if len(chks) == 0:
        raise ValueError("No checkpoint file found in the downloaded zip.")
    chosen_chk = chks[-1]  # Typically picks "best_model.pt" if present

    config = json.loads(zipfile.ZipFile(path).read(cfg))
    data = zipfile.ZipFile(path).read(chosen_chk)
    model = ALIGNN(ALIGNNConfig(**config["model"]))

    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)
    return model

def get_prediction(
    model_name="mp_gappbe_alignn",
    atoms=None,
    cutoff=8,
    max_neighbors=12,
):
    """Get model prediction on a single structure."""
    model = get_figshare_model(model_name)
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=max_neighbors,
    )
    lat = torch.tensor(atoms.lattice_mat)
    out_data = (
        model([g.to(device), lg.to(device), lat.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data

def get_multiple_predictions(
    atoms_array=[],
    jids=[],
    cutoff=8,
    neighbor_strategy="k-nearest",
    max_neighbors=12,
    use_canonize=True,
    target="prop",
    atom_features="cgcnn",
    line_graph=True,
    workers=0,
    include_atoms=True,
    pin_memory=False,
    output_features=1,
    batch_size=1,
    model=None,
    model_name="mp_gappbe_alignn",
    print_freq=100,
):
    """
    Use a pretrained model on a number of structures, returning predictions in-memory.
    Returns a dict mapping row index (jid) to predicted float value.
    """
    # Prepare data
    mem = []
    for i, at in enumerate(atoms_array):
        info = {}
        if isinstance(at, Atoms):
            info["atoms"] = at.to_dict()
        else:
            info["atoms"] = at
        info["prop"] = -9999  # placeholder
        info["jid"] = jids[i]
        mem.append(info)

    if model is None:
        model = get_figshare_model(model_name)

    # Convert dataset to the needed form
    test_data = get_torch_dataset(
        dataset=mem,
        target=target,
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )
    collate_fn = test_data.collate_line_graph

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    # Inference
    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids

        for dat, data_id in tqdm(zip(test_loader, ids), total=len(test_loader), desc="Predicting"):
            g, lg, lat, target_vals = dat
            out_data = model([g.to(device), lg.to(device), lat.to(device)])
            out_data = out_data.cpu().numpy().flatten().tolist()
            results.append({"id": data_id, "pred": out_data[0]})

    # Build a dict: row index -> predicted value
    pred_dict = {}
    for item in results:
        pred_dict[item["id"]] = item["pred"]

    return pred_dict

def _parse_single_cif(args):
    """
    Helper for parallel CIF parsing. Returns (index, Atoms or None).
    Tries with get_primitive_atoms=True, then falls back to False.
    """
    idx, cif_str = args
    try:
        atoms_tmp = Atoms.from_cif(from_string=cif_str, get_primitive_atoms=True, use_cif2cell=False)
    except Exception as e:
        try:
            atoms_tmp = Atoms.from_cif(from_string=cif_str, get_primitive_atoms=False, use_cif2cell=False)
        except Exception as e:
            print(f"Error parsing CIF at index {idx}: {e}, unable to parse CIF.")
            return (idx, None)
        
    return (idx, atoms_tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALIGNN multi-prediction for bandgap (mp_gappbe_alignn) and e-hull (jv_ehull_alignn)"
        " on a dataframe of CIF strings, using ProcessPoolExecutor for parallel CIF parsing and no JSON output."
    )
    parser.add_argument(
        "--input_parquet",
        required=True,
        help="Path to input .parquet file with 'Generated CIF' column at least.",
    )
    parser.add_argument(
        "--output_parquet",
        required=True,
        help="Path to output .parquet file with predictions.",
    )
    parser.add_argument(
        "--cutoff",
        default=8,
        type=float,
        help="Distance cutoff for graph construction",
    )
    parser.add_argument(
        "--max_neighbors",
        default=12,
        type=int,
        help="Maximum neighbors for graph construction",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size for DataLoader",
    )
    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--chunk_size",
        default=10000,
        type=int,
        help="Chunk size for processing rows in parallel",
    )

    args = parser.parse_args()

    # Load the input dataframe
    df = pd.read_parquet(args.input_parquet)
    df.reset_index(drop=True, inplace=True)
    if "Generated CIF" not in df.columns:
        print("Error: 'CIF' column not found.")
        sys.exit(1)

    # Prepare data for parallel parsing
    data_to_parse = [(idx, row["Generated CIF"]) for idx, row in df.iterrows()]

    # Parallel parse with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        parsed_iter = executor.map(_parse_single_cif, data_to_parse)
        parsed_results = list(tqdm(parsed_iter, total=len(data_to_parse), desc="Parsing CIFs in parallel"))

    # Collect valid structures
    atoms_list = []
    jids_list = []
    for (idx, atoms_obj) in parsed_results:
        if atoms_obj is not None:
            atoms_list.append(atoms_obj)
            jids_list.append(idx)

    # Get bandgap predictions (mp_gappbe_alignn)
    bg_results = get_multiple_predictions(
        atoms_array=atoms_list,
        jids=jids_list,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        model_name="mp_gappbe_alignn",
        batch_size=args.batch_size,
        workers=args.num_workers,
    )

    # Create columns in df
    df["ALIGNN_bg (eV)"] = np.nan

    # Map predictions back to dataframe rows
    for idx, val in bg_results.items():
        df.loc[idx, "ALIGNN_bg (eV)"] = val

    print(df.head())
    df.to_parquet(args.output_parquet, index=False)
    print("Saved predictions to", args.output_parquet)
