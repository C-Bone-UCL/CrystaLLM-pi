"""
Compute crystal structure matching metrics using DiffCSP-compliant validity checks.
Evaluates match rate and RMS distance between generated and true structures,
plus additional lattice parameter analysis for XRD benchmarking.

Core metrics:
- match_rate: fraction of structures with valid StructureMatcher matches
- rms_dist: mean RMS distance for matched structures

Additional XRD metrics:
- lattice parameter differences (a_diff, b_diff, c_diff)
- match count (n_matched)
- detailed structure comparison table
"""

import os
import argparse
import sys
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import Counter
import itertools
import smact
from smact.screening import pauling_test

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import is_sensible, is_valid

import warnings
warnings.filterwarnings("ignore")

# Global sensibility check parameters
LENGTH_LO = 0.5
LENGTH_HI = 1000.0
ANGLE_LO = 10.0
ANGLE_HI = 170.0


# DiffCSP smact validity check
# adapted from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([Element.from_Z(elem).symbol for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    # Early exit for computational efficiency
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
        
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False

# DiffCSP structure validity check
# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with large number to ignore self-distances
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    # Three validity criteria: atom distances, volume, lattice size
    if dist_mat.min() < cutoff or crystal.volume < 0.1 or max(crystal.lattice.abc) > 40:
        return False
    else:
        return True

def is_valid_bench(struct):
    """Validity check combining composition and structure validity."""
    # Convert pymatgen structure to comp/count format
    elem_counter = Counter([specie.Z for specie in struct.species])
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))
    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())
    
    comp_valid = smact_validity(comp, count)
    struct_valid = structure_validity(struct)
    return comp_valid and struct_valid

# def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
#     """Check if CIF has reasonable lattice parameters."""
#     cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
#     cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

#     cell_lengths = cell_length_pattern.findall(cif_str)
#     for length_str in cell_lengths:
#         length = float(length_str)
#         if length < length_lo or length > length_hi:
#             return False

#     cell_angles = cell_angle_pattern.findall(cif_str)
#     for _, angle_str in cell_angles:
#         angle = float(angle_str)
#         if angle < angle_lo or angle > angle_hi:
#             return False

#     return True




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


def _create_empty_row(true_struct, valid_num, score=None):
    """Create row for cases where structure processing fails."""
    try:
        true_cif = true_struct.to(fmt="cif")
    except Exception:
        true_cif = None
    
    row = {
        "True Struct": true_cif, "Gen Struct": None, "RMS-d": None,
        "True a": None, "True b": None, "True c": None, "True volume": None,
        "Gen a": None, "Gen b": None, "Gen c": None, "Gen volume": None,
        "Sensibles Num": valid_num
    }
    
    if score is not None:
        row["Score"] = score
    
    return row


def _create_result_row(true_struct, best_gen, rms_dist, valid_num, score=None):
    """Create result row for DataFrame output."""
    true_a, true_b, true_c = true_struct.lattice.abc
    true_vol = true_struct.volume
    
    if best_gen is not None:
        gen_a, gen_b, gen_c = best_gen.lattice.abc
        gen_vol = best_gen.volume
        gen_cif = _symmetrize_cif(best_gen)
    else:
        gen_a = gen_b = gen_c = gen_vol = gen_cif = None
    
    row = {
        "True Struct": _symmetrize_cif(true_struct),
        "Gen Struct": gen_cif,
        "RMS-d": rms_dist if rms_dist not in (9999.0,) else None,
        "True a": true_a, "True b": true_b, "True c": true_c, "True volume": true_vol,
        "Gen a": gen_a, "Gen b": gen_b, "Gen c": gen_c, "Gen volume": gen_vol,
        "Sensible Num": valid_num
    }
    
    if score is not None:
        row["Score"] = score
    
    return row


def _process_material_comparison(args_tuple):
    """Process structure comparison for a single material (for parallel execution)."""
    (
        i, gen_structs_i, true_struct_i, material_id, scores,
        validity_check, matcher_params
    ) = args_tuple
    
    try:
        # Recreate StructureMatcher in worker process
        from pymatgen.analysis.structure_matcher import StructureMatcher
        matcher = StructureMatcher(**matcher_params)
        
        valid_num = len(gen_structs_i)
        
        # Handle true structure parsing errors
        try:
            true_struct_i.volume
        except Exception:
            best_score = scores[0] if scores else None
            return (i, None, None, None, None, _create_empty_row(true_struct_i, valid_num, best_score))
        
        tmp_rms_dists = []
        valid_gen_structs = []
        valid_gen_scores = []
        
        # Process each generated structure
        for j, gen in enumerate(gen_structs_i):
            try:
                if validity_check == "crystallm":
                    struct_valid = is_valid(gen)
                elif validity_check == "diffcsp":
                    struct_valid = is_valid_bench(gen)
                else:
                    struct_valid = True
                
                if struct_valid:
                    try:
                        rms_dist = matcher.get_rms_dist(gen, true_struct_i)
                        rms_dist = None if rms_dist is None else rms_dist[0]
                        if rms_dist is not None:
                            tmp_rms_dists.append(rms_dist)
                            valid_gen_structs.append(gen)
                            gen_score = scores[j] if j < len(scores) else None
                            valid_gen_scores.append(gen_score)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Process results
        tmp_best_gen = None
        tmp_a_diffs = tmp_b_diffs = tmp_c_diffs = None
        final_rms_dist = None
        best_score = None
        
        if len(tmp_rms_dists) == 0:
            # No valid StructureMatcher matches
            final_rms_dist = None
            # Lattice fallback for additional XRD analysis
            tmp_best_gen, tmp_a_diffs, tmp_b_diffs, tmp_c_diffs = _find_best_lattice_match(
                gen_structs_i, true_struct_i)
            if tmp_a_diffs == 9999.0:
                tmp_a_diffs = tmp_b_diffs = tmp_c_diffs = None
            best_score = scores[0] if scores else None
        else:
            # Take minimum RMS distance among valid matches
            min_idx = np.argmin(tmp_rms_dists)
            final_rms_dist = tmp_rms_dists[min_idx]
            tmp_best_gen = valid_gen_structs[min_idx]
            best_score = valid_gen_scores[min_idx] if min_idx < len(valid_gen_scores) else None
            
            # Calculate lattice parameter differences for best RMS match
            true_a, true_b, true_c = true_struct_i.lattice.abc
            gen_a, gen_b, gen_c = tmp_best_gen.lattice.abc
            tmp_a_diffs = abs(true_a - gen_a)
            tmp_b_diffs = abs(true_b - gen_b)
            tmp_c_diffs = abs(true_c - gen_c)
        
        row = _create_result_row(true_struct_i, tmp_best_gen, final_rms_dist, valid_num, best_score)
        
        return (i, final_rms_dist, tmp_a_diffs, tmp_b_diffs, tmp_c_diffs, row)
        
    except Exception as e:
        # Return empty result for failed processing
        return (i, None, None, None, None, None)


def _calculate_metrics(rms_dists, a_diffs, b_diffs, c_diffs, gen_structs):
    """Calculate overall metrics from collected differences."""
    rms_array = np.array(rms_dists, dtype=object)
    match_rate = np.sum(rms_array != None) / len(gen_structs)
    valid_rms = rms_array[rms_array != None]
    mean_rms_dist = valid_rms.mean() if len(valid_rms) > 0 else None
    n_matched = np.sum(rms_array != None)
    
    def calc_mean_diff(diffs):
        arr = np.array(diffs, dtype=object)
        valid = arr[arr != None]
        return valid.mean() if len(valid) > 0 else None
    
    return {
        "match_rate": match_rate,
        "rms_dist": mean_rms_dist,
        "n_matched": n_matched,
        "a_diff": calc_mean_diff(a_diffs),
        "b_diff": calc_mean_diff(b_diffs),
        "c_diff": calc_mean_diff(c_diffs)
    }


def get_match_rate_and_rms(gen_structs, true_structs, matcher, args, score_data=None, num_workers=None):
    """Compute match rate and RMS distance plus additional XRD metrics.
    
    - Only valid structures (smact + structure validity) are considered
    - Takes minimum RMS distance among all valid matches per material
    - Match rate = fraction of materials with at least one valid StructureMatcher match
    - Lattice parameter fallback for analysis only (doesn't count toward match rate)
    
    Args:
        score_data: Dict mapping material_id -> list of scores corresponding to gen_structs
        num_workers: Number of parallel workers (defaults to CPU count // 2)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    # Prepare data for parallel processing
    material_ids = list(score_data.keys()) if score_data else []
    matcher_params = {
        'stol': matcher.stol,
        'angle_tol': matcher.angle_tol, 
        'ltol': matcher.ltol
    }
    
    # Build work items
    work_items = []
    for i in range(len(gen_structs)):
        current_material_id = material_ids[i] if i < len(material_ids) else None
        current_scores = score_data.get(current_material_id, []) if score_data and current_material_id else []
        
        work_items.append((
            i, gen_structs[i], true_structs[i], current_material_id, current_scores,
            args.validity_check, matcher_params
        ))
    
    # Process in parallel
    results = []
    if num_workers == 1:
        # Serial processing for debugging or single-core systems
        for work_item in tqdm(work_items, desc="Comparing structures"):
            results.append(_process_material_comparison(work_item))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_process_material_comparison, work_items),
                total=len(work_items),
                desc="Comparing structures"
            ))
    
    # Collect results in original order
    rms_dists = [None] * len(gen_structs)
    a_diffs = [None] * len(gen_structs)
    b_diffs = [None] * len(gen_structs)
    c_diffs = [None] * len(gen_structs)
    rows = [None] * len(gen_structs)
    
    for result in results:
        if result and len(result) == 6:
            i, rms_dist, a_diff, b_diff, c_diff, row = result
            if i is not None and 0 <= i < len(gen_structs):
                rms_dists[i] = rms_dist
                a_diffs[i] = a_diff
                b_diffs[i] = b_diff
                c_diffs[i] = c_diff
                rows[i] = row
    
    # Handle any failed results by creating empty rows
    for i in range(len(gen_structs)):
        if rows[i] is None:
            current_material_id = material_ids[i] if i < len(material_ids) else None
            current_scores = score_data.get(current_material_id, []) if score_data and current_material_id else []
            best_score = current_scores[0] if current_scores else None
            rows[i] = _create_empty_row(true_structs[i], len(gen_structs[i]), best_score)
    
    return _calculate_metrics(rms_dists, a_diffs, b_diffs, c_diffs, gen_structs), pd.DataFrame(rows)


def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, num_workers, df=None, has_rank_column=False, id_to_scores=None):
    """Process generated and true CIF structures in parallel.
    
    Args:
        id_to_scores: Dict mapping material_id -> list of scores corresponding to CIFs
    """
    from concurrent.futures import ProcessPoolExecutor

    true_structs = []
    valid_material_ids = []
    
    for mid, cifs in tqdm(id_to_gen_cifs.items(), desc="Parsing true CIFs"):
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
        # For rank-based filtering when num_gens=1, the CIFs are already filtered in the main function
        # For other cases, apply the original logic
        if n_gens == 1 and has_rank_column:
            # CIFs are already filtered to rank=1, so use all of them
            subset = cifs
        else:
            # Original logic: take first n_gens or all if None
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
            desc="Parsing and sensible check for gen CIFs"
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
    
    # Filter score data to match processed materials
    filtered_score_data = {}
    if id_to_scores:
        for mid in valid_material_ids:
            if mid in id_to_scores:
                filtered_score_data[mid] = id_to_scores[mid]
    
    return gen_structs, true_structs, filtered_score_data


def _parallel_convert_generated_cif(args):
    """Convert CIF string to pymatgen Structure with sensibility pre-filtering."""
    cif, length_lo, length_hi, angle_lo, angle_hi = args
    try:
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
    parser.add_argument("--ref_parquet", default=None, type=str, help="Override true CIFs from this DB")
    parser.add_argument("--validity_check", type=str, choices=["diffcsp", "crystallm", "none"], default="diffcsp", help="Validity check to use: 'diffcsp' for smact+structure (benchmark), 'crystallm' for to avoid overly strict checks")
    parser.add_argument("--sort_gens", type=str, choices=["rank", "random", "first"], default="rank", help="Sorting method when num_gens=1: 'rank' to use rank=1, 'random' to randomly select one generation, 'first' to use the first generation")

    args = parser.parse_args()

    n_gens = args.num_gens if args.num_gens > 0 else None
    print(f"Using {'all' if n_gens is None else n_gens} generation(s) per compound")

    struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3, primitive_cell=True)


    # Load generated CIFs
    df = pd.read_parquet(args.input_parquet)

    if len(df) > 100000 and args.num_workers != 1:
        args.num_workers = max(1, multiprocessing.cpu_count() // 6)
    else:
        args.num_workers = max(1, multiprocessing.cpu_count() // 3)

    print(f"Using {args.num_workers} workers for parallel processing (based on input size)")
    
    # Check if score column exists
    has_score_column = "score" in df.columns
    has_rank_column = "rank" in df.columns

    if n_gens == 1 and has_rank_column and args.sort_gens == 'rank':  # Fixed condition
        print("Using rank=1 rows for num_gens=1 (rank column detected)")
        # Filter to only rank=1 rows for each material, then take the CIF and score
        id_to_gen_cifs = {}
        id_to_scores = {} if has_score_column else None
        for mid, group in df.groupby("Material ID"):
            rank_1_rows = group[group["rank"] == 1]
            if len(rank_1_rows) > 0:
                id_to_gen_cifs[mid] = rank_1_rows["Generated CIF"].tolist()
                if has_score_column:
                    id_to_scores[mid] = rank_1_rows["score"].tolist()
            else:
                id_to_gen_cifs[mid] = [group["Generated CIF"].iloc[0]]
                if has_score_column:
                    id_to_scores[mid] = [group["score"].iloc[0]]
    elif n_gens == 1 and args.sort_gens == 'random':
        print("Randomly selecting one generation per material for num_gens=1")
        id_to_gen_cifs = {}
        id_to_scores = {} if has_score_column else None
        for mid, group in df.groupby("Material ID"):
            selected_row = group.sample(n=1, random_state=1)
            id_to_gen_cifs[mid] = selected_row["Generated CIF"].tolist()
            if has_score_column:
                id_to_scores[mid] = selected_row["score"].tolist()
    elif n_gens == 1 and args.sort_gens == 'first':
        print("Selecting the first generation per material for num_gens=1")
        id_to_gen_cifs = {}
        id_to_scores = {} if has_score_column else None
        for mid, group in df.groupby("Material ID"):
            first_row = group.iloc[0]
            id_to_gen_cifs[mid] = [first_row["Generated CIF"]]
            if has_score_column:
                id_to_scores[mid] = [first_row["score"]]
    else:
        # Original logic: take all CIFs for each material
        id_to_gen_cifs = {mid: group["Generated CIF"].tolist() for mid, group in df.groupby("Material ID")}
        id_to_scores = {mid: group["score"].tolist() for mid, group in df.groupby("Material ID")} if has_score_column else None
    
    print(f"Loaded {len(id_to_gen_cifs)} materials from {args.input_parquet}")

    # Load true CIFs
    if args.ref_parquet:
        df_testdb = pd.read_parquet(args.ref_parquet)
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
    # Conservative parallelization to avoid memory issues with large structures
    max_workers = max(1, multiprocessing.cpu_count() // 4)
    args.num_workers = min(args.num_workers, max_workers)
    gen_structs, true_structs, score_data = get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, args.num_workers, df, has_rank_column, id_to_scores)
    
    # Use half the workers for structure comparison to balance load
    comparison_workers = max(1, args.num_workers // 2)
    metrics, stats_df = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher, args, score_data, comparison_workers)

    # Save and display results
    stats_df.to_parquet(args.output_parquet, index=False)
    
    print(f"\nResults saved to: {args.output_parquet}")
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")
