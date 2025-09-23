import os
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import is_sensible, is_valid

# TRAIN_MATERIAL_IDS = [
#     "Ba2MnCr",
#     "Ca10(PO4)6(OH)2",
#     "CH3NH3PbI3",
#     "Co2CO3(OH)2",
#     "CsCuTePt",
#     "Cu2C1O5H2",
#     "Cu3(CO3)2(OH)2",
#     "K2AgMoI6",
#     "MgF2",
#     "Mn4(PO4)3",
#     "PbCu(OH)2SO4",
#     "Sm2BO4",
# ]
TRAIN_MATERIAL_IDS = []

def _parallel_parse(args):
    cif, length_lo, length_hi, angle_lo, angle_hi = args
    try:
        if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
            return None
        return Structure.from_str(cif, fmt="cif")
    except Exception:
        return None

def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi, num_workers, group_by_condition=False):
    # First, validate all true structures and collect valid groups
    valid_groups = []
    true_structs = []
    
    for group_key, cifs in tqdm(id_to_gen_cifs.items(), desc="validating true CIFs..."):
        if group_by_condition:
            material_id = group_key[0]
        else:
            material_id = group_key
            
        if material_id not in id_to_true_cifs:
            continue
            
        try:
            true_struct = Structure.from_str(id_to_true_cifs[material_id], fmt="cif")
            valid_groups.append(group_key)
            true_structs.append(true_struct)
        except Exception:
            continue
    
    # Flatten all generated CIFs for parallel processing
    all_work_args = []
    group_boundaries = []
    current_idx = 0
    
    for group_key in valid_groups:
        cifs = id_to_gen_cifs[group_key]
        subset = cifs if n_gens is None else cifs[:min(len(cifs), n_gens)]
        work_args = [(cif, length_lo, length_hi, angle_lo, angle_hi) for cif in subset]
        all_work_args.extend(work_args)
        group_boundaries.append((current_idx, current_idx + len(work_args), subset))
        current_idx += len(work_args)
    
    print(f"Processing {len(all_work_args)} CIFs across {len(valid_groups)} groups...")
    
    # Process all CIFs in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        all_results = list(tqdm(
            ex.map(_parallel_parse, all_work_args),
            total=len(all_work_args),
            desc="converting CIFs to Structures..."
        ))
    
    # Group results back by material/group
    gen_structs = []
    sort_ids = []
    
    for i, (start_idx, end_idx, original_cifs) in enumerate(group_boundaries):
        group_results = all_results[start_idx:end_idx]
        parsed = [(cif, st) for cif, st in zip(original_cifs, group_results) if st is not None]
        
        if parsed:  # Only include groups with at least one valid structure
            gen_structs.append(parsed)
            sort_ids.append(valid_groups[i])
    
    # Filter true_structs to match the final sort_ids
    final_true_structs = []
    for sort_id in sort_ids:
        idx = valid_groups.index(sort_id)
        final_true_structs.append(true_structs[idx])
    
    return gen_structs, final_true_structs, sort_ids

def _parallel_evaluate(args):
    mid, true_cif, gen_cif, stol, angle_tol, ltol = args
    try:
        true_struct = Structure.from_str(true_cif, fmt="cif")
        gen_struct = Structure.from_str(gen_cif, fmt="cif")
    except Exception:
        return mid, true_cif, gen_cif, False, float('nan'), False
    
    try:
        matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol, scale=True)

        is_match = matcher.fit(gen_struct, true_struct)

        rms_info = matcher.get_rms_dist(gen_struct, true_struct)
        norm_factor = (true_struct.volume / true_struct.num_sites) ** (1 / 3)

        rms_d = rms_info[0] / norm_factor

    except Exception as e:
        is_match = False
        rms_d = float('nan')

    try:
        is_valid_flag = is_valid(gen_cif, bond_length_acceptability_cutoff=0.75, debug=False)
    except Exception:
        is_valid_flag = False
    return mid, true_cif, gen_cif, is_match, rms_d, is_valid_flag

def _parallel_evaluate_extended(args):
    group_key, material_id, condition, true_cif, gen_cif, stol, angle_tol, ltol = args
    mid, true_cif_out, gen_cif_out, is_match, rms_d, is_valid_flag = _parallel_evaluate((material_id, true_cif, gen_cif, stol, angle_tol, ltol))
    return group_key, material_id, condition, true_cif_out, gen_cif_out, is_match, rms_d, is_valid_flag
def create_condition_level_mapping(df_gen, sort_by_column):
    """Create mapping from condition_vector to condition level based on ranking within each material"""
    condition_mapping = {}
    
    for material_id in df_gen[sort_by_column].unique():
        material_data = df_gen[df_gen[sort_by_column] == material_id]
        
        # Extract second value from condition vector and collect unique values
        condition_values = []
        seen_second_values = set()
        
        for _, row in material_data.iterrows():
            condition_str = row['condition_vector']
            try:
                # Split the string and convert to float, handling potential errors
                second_value = float(condition_str.split(', ')[2])
                condition_values.append((condition_str, second_value))
                seen_second_values.add(second_value)
            except (ValueError, IndexError) as e:
                # If parsing fails, skip this condition
                print(f"Error parsing condition_vector '{condition_str}' for material {material_id}: {e}")
                continue
        
        # Sort unique second values and create level mapping
        unique_second_values = sorted(seen_second_values)
        second_value_to_level = {val: f"Level_{i+1}" for i, val in enumerate(unique_second_values)}
        
        # Assign levels to all condition strings based on their second value
        for condition_str, second_value in condition_values:
            condition_mapping[condition_str] = second_value_to_level[second_value]
        
        # print(f"Material ID: {material_id}")
        # print(f"Unique second values: {unique_second_values}")
        # print(f"Number of unique conditions for {material_id}: {len(unique_second_values)}")
        # print(f"Level mapping: {second_value_to_level}")
    
    return condition_mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet")
    parser.add_argument("--path_to_db")
    parser.add_argument("--num_gens", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--output_parquet")
    parser.add_argument("--length_lo", type=float, default=0.5)
    parser.add_argument("--length_hi", type=float, default=1000.0)
    parser.add_argument("--angle_lo", type=float, default=10.0)
    parser.add_argument("--angle_hi", type=float, default=170.0)
    parser.add_argument("--sort_by", type=str, default="Material ID")
    parser.add_argument("--group_by_condition", action="store_true", help="Group by Material ID + condition level combination")
    args = parser.parse_args()

    df_gen = pd.read_parquet(args.input_parquet)
    
    if args.group_by_condition:
        # Create condition level mapping
        condition_mapping = create_condition_level_mapping(df_gen, args.sort_by)
        df_gen['condition_level'] = df_gen['condition_vector'].map(condition_mapping)
        
        
        # Group by Material ID and condition level
        id_to_gen_cifs = df_gen.groupby([args.sort_by, "condition_level"])["Generated CIF"].apply(list).to_dict()
        print(f"Number of (Material ID, Condition Level) groups: {len(id_to_gen_cifs)}")
    else:
        id_to_gen_cifs = df_gen.groupby(args.sort_by)["Generated CIF"].apply(list).to_dict()

    df_true = pd.read_parquet(args.path_to_db)

    # df_true = df_true.head(12)
    
    if "Split" in df_true.columns:
        df_true = df_true[df_true["Split"] == "test"].reset_index(drop=True)
    try:
        id_to_true_cifs = df_true.groupby(args.sort_by)["CIF"].first().to_dict()
    except KeyError:
        id_to_true_cifs = df_true.groupby(args.sort_by)["True CIF"].first().to_dict()

    
    # Check overlap between generated and true materials
    gen_materials = set(df_gen[args.sort_by].unique())
    true_materials = set(id_to_true_cifs.keys())
    overlap = gen_materials.intersection(true_materials)
    print(f"Materials in both generated and true data: {len(overlap)}")
    
    if len(overlap) == 0:
        print("ERROR: No materials overlap between generated and true data!")
        print(f"Sample generated materials: {list(gen_materials)[:5]}")
        print(f"Sample true materials: {list(true_materials)[:5]}")
        return

    n_gens = None if args.num_gens == 0 else args.num_gens
    gen_structs, true_structs, sort_ids = get_structs(
        id_to_gen_cifs,
        id_to_true_cifs,
        n_gens,
        args.length_lo,
        args.length_hi,
        args.angle_lo,
        args.angle_hi,
        args.num_workers,
        args.group_by_condition,
    )


    tasks = []
    for group_key, true_struct, gen_list in zip(sort_ids, true_structs, gen_structs):
        if args.group_by_condition:
            material_id = group_key[0]
            condition = group_key[1]
        else:
            material_id = group_key
            condition = None
            
        true_cif = id_to_true_cifs[material_id]
        for gen_cif, _ in gen_list:
            tasks.append((group_key, material_id, condition, true_cif, gen_cif, 0.3, 5, 0.2))

    rows = []
    match_set, valid_set = set(), set()
    match_set_train, match_set_not_train = set(), set()
    condition_stats = defaultdict(lambda: {'matches': set(), 'valids': set(), 'total': 0})

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for group_key, material_id, condition, true_cif, gen_cif, is_match, rms_d, is_valid_flag in tqdm(
            ex.map(_parallel_evaluate_extended, tasks),
            total=len(tasks),
            desc="evaluating structures...",
        ):
            in_train = material_id in TRAIN_MATERIAL_IDS
            rows.append(
                {
                    "ID": material_id,
                    "Condition": condition if condition else "N/A",
                    "Group_Key": str(group_key),
                    "True Struct": true_cif,
                    "Gen Struct": gen_cif,
                    "is_match": is_match,
                    "rms_distance": rms_d,
                    "is_valid": is_valid_flag,
                    "is_in_train": in_train,
                }
            )
            
            if args.group_by_condition:
                condition_stats[condition]['total'] += 1
                if is_match:
                    condition_stats[condition]['matches'].add(material_id)
                if is_valid_flag:
                    condition_stats[condition]['valids'].add(material_id)
            
            if is_match:
                match_set.add(group_key)
                if in_train:
                    match_set_train.add(group_key)
                else:
                    match_set_not_train.add(group_key)
            if is_valid_flag:
                valid_set.add(group_key)

    pd.DataFrame(rows).to_parquet(args.output_parquet)
    
    total_unique_groups = len(sort_ids)
    print(f"Number of groups with at least one Valid Generation: {len(valid_set)}")
    print(f"Number of groups with at least one match in materials seen in training: {len(match_set_train)}")
    print(f"Number of groups with at least one match in materials not seen in training: {len(match_set_not_train)}")
    print(f"Percentage of Valid Generations: {len(valid_set) / total_unique_groups * 100:.2f}% ({len(valid_set)} / {total_unique_groups})")
    print(f"Total number of groups: {total_unique_groups}")
    
    if args.group_by_condition:
        print("\n=== Results by Condition Level ===")
        # Sort condition levels naturally (Level_1, Level_2, etc.)
        sorted_conditions = sorted(condition_stats.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
        
        for condition in sorted_conditions:
            stats = condition_stats[condition]
            total_materials_this_condition = len(set(material_id for group_key, material_id, cond, _, _, _, _, _ in tasks if cond == condition))
            match_rate = len(stats['matches']) / total_materials_this_condition * 100 if total_materials_this_condition > 0 else 0
            valid_rate = len(stats['valids']) / total_materials_this_condition * 100 if total_materials_this_condition > 0 else 0
            print(f"Condition {condition}:")
            print(f"  Materials with matches: {len(stats['matches'])}/{total_materials_this_condition} ({match_rate:.1f}%)")
            print(f"  Materials with valid structures: {len(stats['valids'])}/{total_materials_this_condition} ({valid_rate:.1f}%)")
            print(f"  Total generations: {stats['total']}")

if __name__ == "__main__":
    main()
