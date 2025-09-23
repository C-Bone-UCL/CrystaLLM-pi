"""
Compute lattice parameter metrics (match rate, RMS distance, lattice parameter differences)
between generated and true crystal structures
"""

import os
import argparse
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import is_sensible

import warnings
warnings.filterwarnings("ignore")

# Global sensibility check parameters
LENGTH_LO = 0.5
LENGTH_HI = 1000.0
ANGLE_LO = 10.0
ANGLE_HI = 170.0


def _symmetrize_cif(struct):
    """Convert structure to symmetrized CIF format with error handling."""
    try:
        sga = SpacegroupAnalyzer(struct)
        symm_struct = sga.get_symmetrized_structure()
        return str(CifWriter(symm_struct, symprec=0.1))
    except Exception:
        return struct.to(fmt="cif")


def _find_best_lattice_match(gen_structs, true_struct):
    """Find best generated structure based on lattice parameter similarity."""
    best_gen, best_score = None, float("inf")
    best_a = best_b = best_c = None
    true_a, true_b, true_c = true_struct.lattice.abc
    
    for gen in gen_structs:
        try:
            gen_a, gen_b, gen_c = gen.lattice.abc
            score = abs(true_a - gen_a) + abs(true_b - gen_b) + abs(true_c - gen_c)
            if score < best_score:
                best_score, best_gen = score, gen
                best_a = abs(true_a - gen_a)
                best_b = abs(true_b - gen_b)
                best_c = abs(true_c - gen_c)
        except Exception:
            pass
    
    return best_gen, best_a, best_b, best_c


def _create_empty_row(true_struct, valid_num):
    """Create row for cases where structure processing fails."""
    try:
        true_cif = true_struct.to(fmt="cif")
    except Exception:
        true_cif = None
    
    return {
        "True Struct": true_cif, "Gen Struct": None, "RMS-d": None,
        "True a": None, "True b": None, "True c": None, "True volume": None,
        "Gen a": None, "Gen b": None, "Gen c": None, "Gen volume": None,
        "Valid Num": valid_num
    }


def _create_result_row(true_struct, best_gen, rms_dist, valid_num):
    """Create result row for DataFrame output."""
    true_a, true_b, true_c = true_struct.lattice.abc
    true_vol = true_struct.volume
    
    if best_gen is not None:
        gen_a, gen_b, gen_c = best_gen.lattice.abc
        gen_vol = best_gen.volume
        gen_cif = _symmetrize_cif(best_gen)
    else:
        gen_a = gen_b = gen_c = gen_vol = gen_cif = None
    
    return {
        "True Struct": _symmetrize_cif(true_struct),
        "Gen Struct": gen_cif,
        "RMS-d": rms_dist if rms_dist not in (9999.0,) else None,
        "True a": true_a, "True b": true_b, "True c": true_c, "True volume": true_vol,
        "Gen a": gen_a, "Gen b": gen_b, "Gen c": gen_c, "Gen volume": gen_vol,
        "Valid Num": valid_num
    }


def _calculate_metrics(rms_dists, a_diffs, b_diffs, c_diffs, gen_structs):
    """Calculate overall metrics from collected differences."""
    rms_array = np.array(rms_dists, dtype=object)
    match_rate = np.sum(rms_array != None) / len(gen_structs)
    valid_rms = rms_array[rms_array != None]
    mean_rms_dist = valid_rms.mean() if len(valid_rms) > 0 else None
    
    def calc_mean_diff(diffs):
        arr = np.array(diffs, dtype=object)
        valid = arr[arr != None]
        return valid.mean() if len(valid) > 0 else None
    
    return {
        "match_rate": match_rate,
        "rms_dist": mean_rms_dist,
        "a_diff": calc_mean_diff(a_diffs),
        "b_diff": calc_mean_diff(b_diffs),
        "c_diff": calc_mean_diff(c_diffs)
    }


def get_match_rate_and_rms(gen_structs, true_structs, matcher):
    """Compute lattice parameter metrics between generated and true structures."""
    rms_dists, a_diffs, b_diffs, c_diffs, rows = [], [], [], [], []

    for i in tqdm(range(len(gen_structs)), desc="Comparing structures"):
        valid_num = len(gen_structs[i])
        tmp_rms_dists = 9999.0
        tmp_a_diffs = tmp_b_diffs = tmp_c_diffs = 9999.0
        tmp_best_gen = None
        current_true_struct = true_structs[i]
        
        # Calculate normalization factor for RMS distance
        try:
            norm_factor = (current_true_struct.volume / current_true_struct.num_sites) ** (1 / 3) if current_true_struct.num_sites > 0 else 1.0
        except Exception:
            # Handle structure parsing errors
            rms_dists.append(None)
            a_diffs.extend([None] * 3)
            rows.append(_create_empty_row(current_true_struct, valid_num))
            continue

        # Find best matching generated structure using structure matcher
        for gen in gen_structs[i]:
            try:
                rms_info = matcher.get_rms_dist(gen, true_structs[i])
                if rms_info is not None:
                    rms_val = rms_info[0] / norm_factor
                    if rms_val < tmp_rms_dists:
                        tmp_rms_dists = rms_val
                        tmp_best_gen = gen
                        true_a, true_b, true_c = true_structs[i].lattice.abc
                        gen_a, gen_b, gen_c = gen.lattice.abc
                        tmp_a_diffs = abs(true_a - gen_a)
                        tmp_b_diffs = abs(true_b - gen_b)
                        tmp_c_diffs = abs(true_c - gen_c)
            except Exception:
                pass

        # Fallback to lattice parameter similarity if no structure match
        if tmp_rms_dists == 9999.0:
            tmp_best_gen, tmp_a_diffs, tmp_b_diffs, tmp_c_diffs = _find_best_lattice_match(
                gen_structs[i], current_true_struct)
            tmp_rms_dists = None if tmp_best_gen else 9999.0

        # Store results
        rms_dists.append(tmp_rms_dists if tmp_rms_dists != 9999.0 else None)
        a_diffs.append(tmp_a_diffs if tmp_a_diffs != 9999.0 else None)
        b_diffs.append(tmp_b_diffs if tmp_b_diffs != 9999.0 else None)
        c_diffs.append(tmp_c_diffs if tmp_c_diffs != 9999.0 else None)
        
        rows.append(_create_result_row(current_true_struct, tmp_best_gen, tmp_rms_dists, valid_num))

    return _calculate_metrics(rms_dists, a_diffs, b_diffs, c_diffs, gen_structs), pd.DataFrame(rows)


def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, num_workers):
    """Process generated and true CIF structures in parallel."""
    from concurrent.futures import ProcessPoolExecutor

    true_structs = []
    valid_material_ids = []
    
    for mid, cifs in tqdm(id_to_gen_cifs.items(), desc="Validating true CIFs"):
        if mid not in id_to_true_cifs:
            continue
        try:
            true_struct = Structure.from_str(id_to_true_cifs[mid], fmt="cif")
            true_structs.append(true_struct)
            valid_material_ids.append(mid)
        except Exception:
            continue

    # Prepare work for parallel processing
    all_work_args = []
    material_boundaries = []
    current_idx = 0
    
    for mid in valid_material_ids:
        cifs = id_to_gen_cifs[mid]
        subset = cifs if n_gens is None else cifs[:n_gens]
        work_args = [(cif, LENGTH_LO, LENGTH_HI, ANGLE_LO, ANGLE_HI) for cif in subset]
        all_work_args.extend(work_args)
        material_boundaries.append((current_idx, current_idx + len(work_args)))
        current_idx += len(work_args)

    print(f"Processing {len(all_work_args)} CIFs across {len(valid_material_ids)} materials")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_results = list(tqdm(
            executor.map(_parallel_convert_generated_cif, all_work_args),
            total=len(all_work_args),
            desc="Converting CIFs"
        ))

    # Group results by material
    gen_structs = []
    for start_idx, end_idx in material_boundaries:
        material_results = all_results[start_idx:end_idx]
        valid_structures = [st for st in material_results if st is not None]
        gen_structs.append(valid_structures)

    print(f"Materials processed: {len(gen_structs)}")
    print(f"Materials with sensible structures: {sum(1 for g in gen_structs if len(g) > 0)}")
    
    assert len(gen_structs) == len(true_structs), f"Mismatch: {len(gen_structs)} gen vs {len(true_structs)} true"
    
    return gen_structs, true_structs


def _parallel_convert_generated_cif(args):
    """Helper function for parallel processing of generated CIFs"""
    cif, length_lo, length_hi, angle_lo, angle_hi = args
    try:
        # Check if CIF passes sensibility tests before trying to parse
        if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
            return None
        return Structure.from_str(cif, fmt="cif")
    except Exception:
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True, help="Path to .parquet file with generated structures")
    parser.add_argument("--num_gens", default=0, type=int, help="Max generations per structure (0=all)")
    parser.add_argument("--num_workers", type=int, default=1, help="Workers for parallel processing")
    parser.add_argument("--output_parquet", required=True, type=str, help="Output path for statistics")
    parser.add_argument("--path_to_db", default=None, type=str, help="Override true CIFs from this DB")

    args = parser.parse_args()

    n_gens = args.num_gens if args.num_gens > 0 else None
    print(f"Using {'all' if n_gens is None else n_gens} generation(s) per compound")

    # Structure matcher with loose tolerances for lattice parameters (as per pxrd and jarvis (not spec?) benchmark)
    struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3, scale=True)

    # Load generated CIFs
    df = pd.read_parquet(args.input_parquet)
    id_to_gen_cifs = {mid: group["Generated CIF"].tolist() for mid, group in df.groupby("Material ID")}
    print(f"Loaded {len(id_to_gen_cifs)} materials from {args.input_parquet}")

    # Load true CIFs
    if args.path_to_db:
        df_testdb = pd.read_parquet(args.path_to_db)
        if "Split" in df_testdb.columns:
            df_testdb = df_testdb[df_testdb["Split"] == "test"]
        
        id_to_true_cifs = {}
        for mid, group in df_testdb.groupby("Material ID"):
            cif_col = "CIF" if "CIF" in group.columns else "True CIF"
            id_to_true_cifs[mid] = group[cif_col].iloc[0]
        
        # Filter to intersection
        intersection_ids = set(id_to_gen_cifs.keys()) & set(id_to_true_cifs.keys())
        id_to_gen_cifs = {k: v for k, v in id_to_gen_cifs.items() if k in intersection_ids}
        id_to_true_cifs = {k: v for k, v in id_to_true_cifs.items() if k in intersection_ids}
        print(f"Using {len(intersection_ids)} matched materials from test DB")
    else:
        id_to_true_cifs = {mid: group["True CIF"].iloc[0] for mid, group in df.groupby("Material ID")}
        print(f"Using true CIFs from input parquet")

    # Process structures and compute metrics
    gen_structs, true_structs = get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, args.num_workers)
    metrics, stats_df = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher)

    # Save and display results
    stats_df.to_parquet(args.output_parquet, index=False)
    
    print(f"\nResults saved to: {args.output_parquet}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
