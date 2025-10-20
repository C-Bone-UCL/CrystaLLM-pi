"""
Evaluates generated CIF structures against challenge set materials for validity and structural matching.
We can see how many generated structures are valid and how many match the true structure from the challenge set. Including info on whether the material was seen in training or not.
"""

import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import is_sensible, is_valid

# These are the materials in the training dataset according to CrystaLLM paper
# https://www.nature.com/articles/s41467-024-54639-7

TRAIN_MATERIAL_IDS = [
    "Ba2MnCr",
    "Ca10(PO4)6(OH)2",
    "CH3NH3PbI3",
    "Co2CO3(OH)2",
    "CsCuTePt",
    "Cu2C1O5H2",
    "Cu3(CO3)2(OH)2",
    "K2AgMoI6",
    "MgF2",
    "Mn4(PO4)3",
    "PbCu(OH)2SO4",
    "Sm2BO4",
]

# Global sensibility check parameters
LENGTH_LO = 0.5
LENGTH_HI = 1000.0
ANGLE_LO = 10.0
ANGLE_HI = 170.0

# Structure matcher parameters
STOL = 0.3
ANGLE_TOL = 5
LTOL = 0.2

def _parallel_parse(args):
    """Parse CIF string to pymatgen Structure with sensibility checks."""
    cif, length_lo, length_hi, angle_lo, angle_hi = args
    try:
        if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
            return None
        return Structure.from_str(cif, fmt="cif")
    except Exception:
        return None

def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi, num_workers):
    """Convert CIF strings to pymatgen Structures for generated and true materials."""
    gen_structs, true_structs, material_ids = [], [], []
    all_work_args = []
    material_cif_mapping = []
    
    # First pass: collect all valid materials and prepare work args
    for material_id, cifs in id_to_gen_cifs.items():
        if material_id not in id_to_true_cifs:
            continue
        try:
            true_struct = Structure.from_str(id_to_true_cifs[material_id], fmt="cif")
        except Exception:
            continue
        
        subset = cifs if n_gens is None else cifs[:n_gens]
        start_idx = len(all_work_args)
        for cif in subset:
            all_work_args.append((cif, length_lo, length_hi, angle_lo, angle_hi))
        end_idx = len(all_work_args)
        
        material_cif_mapping.append((material_id, true_struct, subset, start_idx, end_idx))
    
    # Process all CIFs in a single executor
    all_parsed = []
    if all_work_args:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            all_parsed = list(tqdm(
                executor.map(_parallel_parse, all_work_args),
                total=len(all_work_args),
                desc="converting CIFs to Structures..."
            ))
    
    # Second pass: group results back by material
    for material_id, true_struct, subset, start_idx, end_idx in material_cif_mapping:
        parsed_batch = all_parsed[start_idx:end_idx]
        parsed = [(cif, st) for cif, st in zip(subset, parsed_batch) if st is not None]
        if not parsed:
            continue
        material_ids.append(material_id)
        true_structs.append(true_struct)
        gen_structs.append(parsed)
    
    return gen_structs, true_structs, material_ids

def _parallel_evaluate(args):
    """Evaluate structural match and validity for a generated CIF against true structure."""
    material_id, true_cif, gen_cif, stol, angle_tol, ltol = args
    try:
        true_struct = Structure.from_str(true_cif, fmt="cif")
        gen_struct = Structure.from_str(gen_cif, fmt="cif")
    except Exception:
        return material_id, true_cif, gen_cif, False, False
    
    try:
        matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol, scale=True)
        is_match = matcher.fit(gen_struct, true_struct)
    except Exception:
        is_match = False

    try:
        is_valid_flag = is_valid(gen_cif)
    except Exception:
        is_valid_flag = False
    return material_id, true_cif, gen_cif, is_match, is_valid_flag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="Input .parquet with columns: Material ID, Generated CIF")
    parser.add_argument("--path_to_db", required=True, help="Path to .parquet with columns: Material ID, CIF or True CIF, (and optionally Split)")
    parser.add_argument("--num_gens", type=int, default=0, help="Number of generations to consider per material ID (0 for all)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallel processing")
    parser.add_argument("--output_parquet", required=True, help="Output .parquet path")
    args = parser.parse_args()

    df_gen = pd.read_parquet(args.input_parquet)
    id_to_gen_cifs = df_gen.groupby("Material ID")["Generated CIF"].apply(list).to_dict()

    df_true = pd.read_parquet(args.path_to_db)
    if "Split" in df_true.columns:
        df_true = df_true[df_true["Split"] == "test"].reset_index(drop=True)
    try:
        id_to_true_cifs = df_true.groupby("Material ID")["CIF"].first().to_dict()
    except KeyError:
        id_to_true_cifs = df_true.groupby("Material ID")["True CIF"].first().to_dict()

    n_gens = None if args.num_gens == 0 else args.num_gens
    gen_structs, true_structs, material_ids = get_structs(
        id_to_gen_cifs,
        id_to_true_cifs,
        n_gens,
        LENGTH_LO,
        LENGTH_HI,
        ANGLE_LO,
        ANGLE_HI,
        args.num_workers,
    )

    tasks = []
    for material_id, true_struct, gen_list in zip(material_ids, true_structs, gen_structs):
        true_cif = id_to_true_cifs[material_id]
        for gen_cif, _ in gen_list:
            tasks.append((material_id, true_cif, gen_cif, STOL, ANGLE_TOL, LTOL))

    rows = []
    match_set, valid_set = set(), set()
    match_set_train, match_set_not_train = set(), set()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for material_id, true_cif, gen_cif, is_match, is_valid_flag in tqdm(
            executor.map(_parallel_evaluate, tasks),
            total=len(tasks),
            desc="evaluating structures...",
        ):
            in_train = material_id in TRAIN_MATERIAL_IDS
            rows.append(
                {
                    "Material ID": material_id,
                    "True Struct": true_cif,
                    "Gen Struct": gen_cif,
                    "is_match": is_match,
                    "is_valid": is_valid_flag,
                    "is_in_train": in_train,
                }
            )
            if is_match:
                match_set.add(material_id)
                if in_train:
                    match_set_train.add(material_id)
                else:
                    match_set_not_train.add(material_id)
            if is_valid_flag:
                valid_set.add(material_id)

    pd.DataFrame(rows).to_parquet(args.output_parquet)
    
    print(f"Number of CIFs with at least one Valid Generation: {len(valid_set)}")
    print(f"Number of CIFs with at least one match in materials seen in training: {len(match_set_train)}")
    print(f"Number of CIFs with at least one match in materials not seen in training: {len(match_set_not_train)}")
    
    # also the percentages of each of these 3
    print(f"Percentage of Valid Generations: {len(valid_set) / len(material_ids) * 100:.2f}% ({len(valid_set)} / {len(material_ids)})")
    # percent for match set train is compared to total number of materials in training set
    print(f"Percentage of matches in training set: {len(match_set_train) / len(TRAIN_MATERIAL_IDS) * 100:.2f}% ({len(match_set_train)} / {len(TRAIN_MATERIAL_IDS)})")
    print(f"Percentage of matches in not training set: {len(match_set_not_train) / (len(material_ids) - len(TRAIN_MATERIAL_IDS)) * 100:.2f}% ({len(match_set_not_train)} / {len(material_ids) - len(TRAIN_MATERIAL_IDS)})")
    
    print(f"Total number of materials: {len(material_ids)}")

if __name__ == "__main__":
    main()