"""Clean CIFs and optionally build offline train-only CIF augmentations."""

import argparse
import ast
import os
import warnings
import multiprocessing as mp
from functools import partial

import pandas as pd
import sys
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter  # <-- ADDED for debug writing

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    assign_split_labels,
    normalize_property_column,
)
from _utils._preprocessing_utils import (
    get_train_augmentation_mask,
    run_parallel_chunks as _run_parallel_chunks,
    apply_variant_dedup_and_thresholds as _apply_variant_dedup_and_thresholds,
    prepare_variant_row_for_counting as _prepare_variant_row_for_counting,
    init_tokenizer_worker as _init_tokenizer_worker,
    count_chunk_token_lengths,
    parse_structure,
    get_supercell_candidate_sequence,
    structure_to_augmented_cif,
    quick_token_count,
    _is_equivalent_cell,
)

warnings.filterwarnings("ignore")

CHUNK_SIZE = 1000
DECIMAL_PLACES = 4
OXI_DEFAULT = False
DEFAULT_TOKENIZER_DIR = "HF-cif-tokenizer"
DEFAULT_TOKEN_BATCH_SIZE = 256


def process_cif_chunk(
    chunk,
    oxi,
    supercell_seed,
    context_length,
    tokenizer_batch_size,
    decimal_places=4,
    debug_dir=None,
):
    """Parse CIFs into structures, produce normalized base CIF and train-row augmented variants with optional token counts."""
    results = []
    variant_rows = []  # only populated when context_length is not None

    def _write_debug_cif(cif_text, r_key, aug_type):
        """Helper to write debug CIFs to disk (stripping ML brackets for VESTA)."""
        if debug_dir and cif_text and str(cif_text).strip() != "":
            # Strip the ML tokenization brackets so VESTA can parse the file
            clean_text = str(cif_text).replace("[", "").replace("]", "")
            
            # Sanitize row_key just in case it contains slashes
            safe_key = str(r_key).replace("/", "_").replace("\\", "_")
            filepath = os.path.join(debug_dir, f"{safe_key}_{aug_type}.cif")
            with open(filepath, "w") as f:
                f.write(clean_text)

    for idx, raw_cif, row_key, is_train in chunk:
        base_cif = None
        niggli_cif = ""
        primitive_cif = ""
        supercell_cif = ""
        sampled_params = []
        structure = None

        try:
            structure = parse_structure(str(raw_cif))
            
            # 1. Generate the ML-ready base CIF (semi-symmetrized with brackets)
            base_cif = structure_to_augmented_cif(structure, oxi=oxi, decimal_places=decimal_places, symmetrize=True)
            
            # 2. Write the Debug base CIF (Fully symmetrized, NO semi-symmetrization)
            if debug_dir:
                try:
                    sga = SpacegroupAnalyzer(structure, symprec=0.1)
                    sym_struct = sga.get_symmetrized_structure()
                    # Generate a pure physical CIF, bypassing ML normalization completely
                    debug_base_text = str(CifWriter(sym_struct, symprec=0.1))
                    _write_debug_cif(debug_base_text, row_key, "base")
                except Exception:
                    # Fallback just in case standard CifWriter fails
                    _write_debug_cif(base_cif, row_key, "base")
                    
        except Exception:
            pass

        if base_cif is None:
            variant_rows.append(("", "", "", ""))
            results.append((idx, None, "", "", "", []))
            continue

        if is_train:
            try:
                # Niggli: mathematically minimal cell. Symmetrize=False to keep exact bounds.
                try:
                    niggli_structure = structure.get_reduced_structure(reduction_algo="niggli")
                    c = structure_to_augmented_cif(niggli_structure, oxi=oxi, decimal_places=decimal_places, symmetrize=False)
                    if c:
                        niggli_cif = c
                        _write_debug_cif(niggli_cif, row_key, "niggli")
                except Exception:
                    pass

                # Primitive: reduced primitive cell (smaller than conventional for centred lattices).
                # Symmetrize=False to avoid pymatgen re-expanding back to conventional.
                try:
                    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
                    prim_structure = analyzer.get_primitive_standard_structure()
                    if not _is_equivalent_cell(structure, prim_structure):
                        c = structure_to_augmented_cif(prim_structure, oxi=oxi, decimal_places=decimal_places, symmetrize=False)
                        if c:
                            primitive_cif = c
                            _write_debug_cif(primitive_cif, row_key, "primitive")
                except Exception:
                    pass

                # Supercell: expanded periodic cell. Symmetrize=False to prevent collapse.
                # If context_length is set, skip candidates whose token count already exceeds
                # the threshold so smaller fallback candidates are tried.
                for supercell_candidate in get_supercell_candidate_sequence(row_key=row_key, seed=supercell_seed):
                    try:
                        supercell = structure.copy()
                        supercell.make_supercell(tuple(supercell_candidate))
                        c = structure_to_augmented_cif(supercell, oxi=oxi, decimal_places=decimal_places, symmetrize=False)
                        if c:
                            if context_length is not None:
                                token_count = quick_token_count(c)
                                if token_count is not None and token_count > context_length:
                                    continue
                            supercell_cif = c
                            sampled_params = list(supercell_candidate)
                            
                            # Write debug file with parameter dimensions in the name
                            param_str = "x".join(map(str, sampled_params))
                            _write_debug_cif(supercell_cif, row_key, f"supercell_{param_str}")
                            break
                    except Exception:
                        continue

            except Exception:
                pass

        variant_rows.append(
            _prepare_variant_row_for_counting(base_cif, niggli_cif, primitive_cif, supercell_cif)
        )
        results.append((idx, base_cif, niggli_cif, primitive_cif, supercell_cif, sampled_params))

    counts = count_chunk_token_lengths(variant_rows, batch_size=tokenizer_batch_size)

    final_results = []
    for row_i, (idx, base_cif, niggli_cif, primitive_cif, supercell_cif, sampled_params) in enumerate(results):
        if base_cif is None:
            final_results.append((idx, None, "", "", "", [], [0, 0, 0, 0]))
            continue

        if counts is not None:
            row_counts = counts[row_i].tolist()
            normalized_niggli, normalized_primitive, normalized_supercell, normalized_counts = (
                _apply_variant_dedup_and_thresholds(
                    base_cif=base_cif,
                    niggli_cif=niggli_cif,
                    primitive_cif=primitive_cif,
                    supercell_cif=supercell_cif,
                    token_counts=row_counts,
                    max_augmented_tokens=context_length,
                )
            )
            final_results.append((
                idx, base_cif,
                normalized_niggli, normalized_primitive, normalized_supercell,
                sampled_params, normalized_counts,
            ))
        else:
            niggli_cif = "" if (not niggli_cif or niggli_cif == base_cif) else niggli_cif
            primitive_cif = "" if (not primitive_cif or primitive_cif == base_cif) else primitive_cif
            supercell_cif = "" if (not supercell_cif or supercell_cif == base_cif) else supercell_cif
            final_results.append((idx, base_cif, niggli_cif, primitive_cif, supercell_cif, sampled_params, []))

    return final_results, len(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("--input_parquet", type=str,
                        help="Path to the input file. (Parquet format)")
    parser.add_argument("--output_parquet", "-o", action="store",
                        required=True,
                        help="Path to the output file. (Parquet format)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers to use for processing.")
    parser.add_argument("--property_columns", type=str, default="[]",
                        help="List of property columns to normalize, e.g., \"['Bandgap (eV)', 'ehull']\".")
    parser.add_argument("--property1_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the first property column.")
    parser.add_argument("--property2_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the second property column.")
    parser.add_argument("--property3_normaliser", type=str, choices=["power_log", "linear", "signed_log", "log10", "None"], default="None",
                        help="Normalization method for the third property column.")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Context length used as token threshold: blanks overlong augmented variants and records token counts. Does not remove rows.")
    parser.add_argument("--add_split_column", action="store_true",
                        help="Add a Split column using train/val/test assignment logic.")
    parser.add_argument("--test_size", type=float, default=0.0,
                        help="Test split fraction used with --add_split_column.")
    parser.add_argument("--valid_size", type=float, default=0.0,
                        help="Validation split fraction used with --add_split_column.")
    parser.add_argument("--duplicates", action="store_true",
                        help="Split by Material ID to avoid leakage when duplicates exist.")
    parser.add_argument("--split_seed", type=int, default=1,
                        help="Seed for deterministic split assignment.")
    parser.add_argument("--supercell_seed", type=int, default=1,
                        help="Seed for deterministic supercell parameter assignment.")
    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_TOKENIZER_DIR,
                        help="Tokenizer directory used for token counting.")
    parser.add_argument("--tokenizer_batch_size", type=int, default=DEFAULT_TOKEN_BATCH_SIZE,
                        help="Batch size for token counting pass.")
    parser.add_argument("--xtra_augment", action="store_true",
                        help="Build Niggli, primitive, and supercell CIF augmentation columns for train rows.")
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Optional directory to save generated CIFs for visual debugging before they are tokenized.")

    args = parser.parse_args()

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        print(f"Debug mode enabled: Saving CIF variants to '{args.debug_dir}'\n")

    print(f"Loading data from {args.input_parquet} as Parquet with zstd compression...\n")
    dataframe = pd.read_parquet(args.input_parquet)

    if "CIF" not in dataframe.columns:
        raise ValueError("The input dataframe must contain the 'CIF' column.")

    try:
        property_columns = ast.literal_eval(args.property_columns)
        if not isinstance(property_columns, list):
            raise ValueError
    except Exception:
        raise ValueError("property_columns must be a valid list, e.g., \"['Bandgap (eV)', 'ehull']\"")

    normalisers = [
        args.property1_normaliser if i == 0 else
        args.property2_normaliser if i == 1 else
        args.property3_normaliser if i == 2 else
        "None"
        for i in range(len(property_columns))
    ]

    missing_props = [prop for prop in property_columns if prop not in dataframe.columns]
    if missing_props:
        print(f"Warning: Property columns not found in dataframe: {missing_props}\n")

    if property_columns:
        print("Normalizing property columns\n")
        for i, prop in enumerate(property_columns):
            if normalisers[i] != "None":
                dataframe = normalize_property_column(dataframe, prop, normalisers[i])

    if args.add_split_column:
        print("Assigning Split column\n")
        dataframe["Split"] = assign_split_labels(
            dataframe=dataframe,
            test_size=args.test_size,
            valid_size=args.valid_size,
            duplicates_mode=args.duplicates,
            seed=args.split_seed,
        )

    dataframe.reset_index(drop=True, inplace=True)

    if "Material ID" in dataframe.columns:
        row_keys = dataframe["Material ID"].astype(str).tolist()
    else:
        row_keys = [str(i) for i in range(len(dataframe))]

    train_mask = get_train_augmentation_mask(dataframe) if args.xtra_augment else pd.Series([False] * len(dataframe))

    all_rows = [
        (idx, dataframe.at[idx, "CIF"], row_keys[idx], bool(train_mask.iloc[idx]))
        for idx in range(len(dataframe))
    ]
    chunks = [all_rows[i:i + CHUNK_SIZE] for i in range(0, len(all_rows), CHUNK_SIZE)]

    print(f"Processing {len(dataframe)} CIFs with {args.num_workers} workers\n")
    if args.context_length is not None:
        print(f"Context length: {args.context_length} (token counting enabled, variants above threshold will be blanked)\n")
    if args.xtra_augment:
        print("Building augmentation variants for train rows\n")

    pool_kwargs = {
        "initializer": _init_tokenizer_worker,
        "initargs": (args.tokenizer_dir,),
    }

    worker = partial(
        process_cif_chunk,
        oxi=OXI_DEFAULT,
        supercell_seed=args.supercell_seed,
        context_length=args.context_length,
        tokenizer_batch_size=args.tokenizer_batch_size,
        decimal_places=DECIMAL_PLACES,
        debug_dir=args.debug_dir,
    )

    with mp.Pool(processes=args.num_workers, **pool_kwargs) as pool:
        all_results = _run_parallel_chunks(pool, worker, chunks, len(all_rows))

    # Build results lookup keyed by original (reset) positional index
    results_dict = {
        idx: (base_cif, niggli, primitive, supercell, params, counts)
        for idx, base_cif, niggli, primitive, supercell, params, counts in all_results
    }

    dataframe["CIF"] = [results_dict[idx][0] for idx in dataframe.index]

    if args.xtra_augment:
        dataframe["CIF_NIGGLI"] = [results_dict[idx][1] for idx in dataframe.index]
        dataframe["CIF_PRIMITIVE"] = [results_dict[idx][2] for idx in dataframe.index]
        dataframe["CIF_SUPERCELL"] = [results_dict[idx][3] for idx in dataframe.index]
        dataframe["supercell_params"] = pd.Series(
            [results_dict[idx][4] for idx in dataframe.index], dtype=object
        )
        dataframe["token_count_by_cif_variant"] = pd.Series(
                [results_dict[idx][5] for idx in dataframe.index], dtype=object
            )
    else:
        dataframe["token_count"] = [
            results_dict[idx][5][0] if results_dict[idx][5] else 0
            for idx in dataframe.index
        ]

    print(f"\nNumber of CIFs before filtering out bad ones: {len(dataframe)}\n")
    dataframe = dataframe[dataframe["CIF"].str.startswith("data_", na=False)]
    print(f"Number of CIFs after filtering: {len(dataframe)}\n")
    dataframe.reset_index(drop=True, inplace=True)

    if os.path.dirname(args.output_parquet) != "":
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    print(f"Saving updated dataframe with {len(dataframe)} rows to {args.output_parquet}...\n")
    dataframe.to_parquet(args.output_parquet, compression='zstd')

    print("Preprocessing completed successfully.\n")