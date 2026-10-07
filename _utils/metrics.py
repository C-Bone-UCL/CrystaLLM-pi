"""VUN metrics and property calculations for CrystaLLM-pi."""

import argparse
import os
import re
import signal
import sys
import warnings
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from typing import Generator
import json
import concurrent.futures

import pandas as pd

from _utils.validity import (
    _configure_pymatgen_warning_filters,
    bond_length_reasonableness_score,
    get_density,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    is_sensible,
    is_space_group_consistent,
    is_valid,
    _validity_worker,
)
from _utils.mp_data import MPDataProvider, download_mp_data


from datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN

from _utils import (
    cif_parser,
    extract_volume,
    extract_formula_units,
    extract_data_formula,
)
from _utils._generating.postprocess import process_dataframe

logger = logging.getLogger(__name__)
_configure_pymatgen_warning_filters()

# Generated data processing
def load_and_process_generated_data(gen_data_path: str, num_workers: int) -> pd.DataFrame:
    """Load generated CIFs and run processing pipeline."""
    print("\nLoading & Processing Generated CIFs")
    print(f"Loading generated data from {gen_data_path}...")
    
    gen_df = pd.read_parquet(gen_data_path)
    
    # Add the condition column so downstream sorting has it
    if 'condition_vector' not in gen_df.columns and 'Condition Vector' not in gen_df.columns:
        print("No condition column found. Creating 'condition_vector' with value -100.")
        gen_df['condition_vector'] = -100
    
    if "Generated CIF" not in gen_df.columns:
        if "CIF" in gen_df.columns:
            gen_df.rename(columns={"CIF": "Generated CIF"}, inplace=True)
        else:
            raise ValueError("Input DataFrame must contain 'Generated CIF' column.")

    print("Processing the generated CIFs...")
    return process_dataframe(gen_df, num_workers=num_workers, column_name='Generated CIF')


def build_generated_structures(df_proc: pd.DataFrame) -> list:
    """Convert valid CIFs to pymatgen structures."""
    structures = [None] * len(df_proc)
    
    # This part is fast, so no need to parallelize
    for idx in tqdm(df_proc.index, desc="Building generated structures"):
        if df_proc.at[idx, "is_valid"]:
            try:
                cif_str = df_proc.at[idx, "Generated CIF"]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    structures[idx] = cif_parser(cif_str).parse_structures(primitive=False)[0]
            except Exception:
                # If structure building fails here, it's not valid
                df_proc.at[idx, "is_valid"] = False
    
    return structures


def extract_generated_formulas(structures: list) -> list:
    """Get unique formulas from structures, using normalized reduced formulas."""
    print("Extracting unique reduced formulas from generated structures...")
    
    formulas = set()
    for struct in tqdm(structures, desc="Extracting reduced formulas"):
        if struct is not None:
            try:
                # Normalize through Composition so formulas compare consistently
                normalized_formula = Composition(struct.composition).reduced_formula
                formulas.add(normalized_formula)
            except Exception:
                continue
    
    print(f"Found {len(formulas)} unique reduced formulas in generated set")
    return formulas


# VUN metrics

def get_valid(df_proc: pd.DataFrame, num_workers: int, bond_length_acceptability_cutoff: float=1.0, allow_stated_p1_mismatch: bool=False) -> tuple:
    """Check CIF validity in parallel."""
    print("\nValidity Metrics")
    
    max_workers_recommended = 16
    max_workers = min(num_workers, max_workers_recommended)
    cifs_to_check = df_proc["Generated CIF"].tolist()
    
    worker_func = partial(
        _validity_worker,
        bond_length_acceptability_cutoff=bond_length_acceptability_cutoff,
        allow_stated_p1_mismatch=allow_stated_p1_mismatch,
    )
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(worker_func, cif): i for i, cif in enumerate(cifs_to_check)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(future_to_idx), desc="Checking validity"):
            results[future_to_idx[future]] = future.result()

    df_proc["is_valid"] = [results[i] for i in range(len(cifs_to_check))]
    valid_count = df_proc['is_valid'].sum()
    
    print(f"{valid_count} valid CIFs out of {len(df_proc)} total")
    print(f"Validity rate: {valid_count / len(df_proc) * 100:.2f}%")
    
    return df_proc


def _uniqueness_worker(args_tuple: tuple[int, str, float]) -> tuple[int, str, float]:
    """Worker to compute BAWL hash and get a metric for uniqueness selection."""
    _configure_pymatgen_warning_filters()
    from material_hasher.hasher.bawl import BAWLHasher
    idx, cif_str, ehull_val = args_tuple
    
    try:
        struct = cif_parser(cif_str).parse_structures(primitive=False)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*No oxidation states specified on sites!.*",
                category=UserWarning,
            )
            hash_val = BAWLHasher().get_material_hash(struct)
        
        # Use ehull if available, otherwise fallback to volume per formula unit
        if ehull_val is not None and not np.isnan(ehull_val):
            selection_metric = ehull_val
        else:
            formula_units = extract_formula_units(cif_str) or 1
            selection_metric = extract_volume(cif_str) / formula_units
            
        return idx, hash_val, selection_metric
    except Exception:
        return idx, None, None

def get_unique(df_gen: pd.DataFrame, workers: int) -> tuple:
    """Find unique structures using BAWL hashing."""
    print("\nUniqueness Metrics")
    max_workers_recommended = 32
    max_workers = min(workers, max_workers_recommended)

    ehull_column = next((col for col in df_gen.columns if 'ehull' in col.lower()), None)
    if ehull_column:
        print(f"Using '{ehull_column}' to select best among duplicates.")
    else:
        print("No ehull column found. Using volume per formula unit to select best among duplicates.")

    df_valid = df_gen[df_gen["is_valid"]].copy()
    if df_valid.empty:
        print("No valid structures to check for uniqueness.")
        df_gen["is_unique"] = False
        return df_gen

    ehull_values = df_valid[ehull_column].tolist() if ehull_column else [None] * len(df_valid)
    cifs_to_check = df_valid["Generated CIF"].tolist()
    
    # Process valid CIFs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = list(zip(df_valid.index, cifs_to_check, ehull_values))
        futures = [executor.submit(_uniqueness_worker, task) for task in tasks]
        
        # Process results to find unique structures
        is_unique = pd.Series(False, index=df_gen.index)
        best_indices, best_metrics = {}, {}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), desc="Finding best unique structures"):
            idx, hash_val, metric_val = future.result()
            if hash_val is None:
                continue
            
            # Keep structure with best metric value (lowest for both ehull and volume per formula unit)
            if hash_val not in best_indices or metric_val < best_metrics[hash_val]:
                if hash_val in best_indices:
                    is_unique.at[best_indices[hash_val]] = False
                best_indices[hash_val] = idx
                best_metrics[hash_val] = metric_val
                is_unique.at[idx] = True

    df_gen["is_unique"] = is_unique
    
    unique_count = df_gen['is_unique'].sum()
    total_valid = len(df_valid)
    
    print(f"{unique_count} unique CIFs out of {total_valid} valid structures.")
    if total_valid > 0:
        print(f"Uniqueness rate among valid: {unique_count / total_valid * 100:.2f}%")
        
    return df_gen


def _novelty_worker(args_tuple: tuple) -> tuple:
    """Worker to check if a single generated structure is novel."""
    _configure_pymatgen_warning_filters()
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher

    gen_struct, comp_key, base_comps, ltol, stol, angle_tol = args_tuple

    if gen_struct is None or not comp_key:
        return False

    # If no reference structures with this composition exist, it's novel by definition
    ref_cifs = base_comps.get(comp_key, [])
    if not ref_cifs:
        return True

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    
    for ref_cif_str in ref_cifs:
        try:
            ref_struct = cif_parser(ref_cif_str).parse_structures(primitive=False)[0]
            if matcher.fit(gen_struct, ref_struct):
                return False  # Found a match, so it's not novel
        except Exception:
            continue  # Ignore faulty reference CIFs

    return True # No match found after checking all references

def get_novelty(df_gen: pd.DataFrame, base_comps: set, ltol: float, stol: float, angle_tol: float, structures: list, workers: int) -> tuple:
    """Check if structures are novel vs training set."""
    print("\nNovelty Metrics")
    max_workers_recommended = 32
    max_workers = min(workers, max_workers_recommended)

    df_to_check = df_gen[df_gen["is_unique"]].copy()
    if df_to_check.empty:
        print("No unique structures to check for novelty.")
        df_gen["is_novel"] = False
        return df_gen
        
    tasks = []
    for idx, row in df_to_check.iterrows():
        struct = structures[df_gen.index.get_loc(idx)]
        # Reduce to the smallest whole-number formula so Si2O4 and SiO2 share a key
        if struct:
            try:
                comp_key = Composition(struct.composition).reduced_formula
            except Exception:
                comp_key = None
        else:
            comp_key = None
        tasks.append((struct, comp_key, base_comps, ltol, stol, angle_tol))

    # Run novelty checks in parallel
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_novelty_worker, task): i for i, task in enumerate(tasks)}
        for future in tqdm(concurrent.futures.as_completed(future_to_idx),
                           total=len(future_to_idx), desc="Checking novelty"):
            results[future_to_idx[future]] = future.result()

    df_gen["is_novel"] = False
    df_gen.loc[df_to_check.index, "is_novel"] = [results[i] for i in range(len(tasks))]
    
    novel_count = df_gen['is_novel'].sum()
    print(f"Found {novel_count} novel CIFs.")
    return df_gen

# Novelty helpers
def load_and_filter_training_data(hf_dataset: str, processed_data_path: str, num_workers: int, gen_formulas: list) -> pd.DataFrame:
    """Load training data and filter to relevant compositions."""
    print("\nLoading Training Dataset (for novelty check)")

    if processed_data_path and os.path.exists(processed_data_path):
        print(f"Loading pre-processed training data from {processed_data_path}.")
        proc_train_df = pd.read_parquet(processed_data_path)
    else:
        print(f"Loading training data from Hugging Face: {hf_dataset}")
        train_dataset = load_dataset(hf_dataset, split="train").to_pandas()
        proc_train_df = process_dataframe(train_dataset, num_workers=num_workers, column_name='CIF')
        if processed_data_path:
            proc_train_df.to_parquet(processed_data_path)
            print(f"Saved processed training data to {processed_data_path}.")

    return build_reference_compositions(proc_train_df, gen_formulas)

def build_reference_compositions(proc_train_df: pd.DataFrame, gen_formulas: list) -> set:
    """Group training CIFs by composition, using normalized formula keys."""
    print("Filtering training dataset to compositions present in generated set...")
    
    base_comps = defaultdict(list)
    
    # Normalize generated formulas for matching
    gen_formulas_normalized = set()
    for formula in gen_formulas:
        try:
            normalized = Composition(formula).reduced_formula
            gen_formulas_normalized.add(normalized)
        except Exception:
            continue
    
    # Build normalized lookup for training data
    for _, row in tqdm(proc_train_df.iterrows(), total=len(proc_train_df), desc="Grouping training CIFs"):
        cif_string = row.get('CIF')
        comp_key = row.get('Reduced Formula')
        
        if cif_string and comp_key:
            try:
                # Normalize the composition key to handle different formula representations
                normalized_key = Composition(comp_key).reduced_formula
                
                # Only include if matches a generated composition
                if normalized_key in gen_formulas_normalized:
                    base_comps[normalized_key].append(cif_string)
            except Exception:
                continue
            
    total_refs = sum(len(v) for v in base_comps.values())
    print(f"Filtered training set to {total_refs} CIFs across {len(base_comps)} compositions.")
    return base_comps

# Compositional novelty
def get_comp_novelty(df_gen: pd.DataFrame, base_comps: set, structures: list) -> tuple:
    """Check if compositions are novel vs training set."""
    print("\nCompositional Novelty Metrics")
    
    # Only check structures that are both valid AND unique
    df_to_check = df_gen[df_gen["is_unique"]].copy()
    if df_to_check.empty:
        print("No unique structures to check for compositional novelty.")
        df_gen["is_comp_novel"] = False
        return df_gen
    
    is_comp_novel_list = []
    for idx in tqdm(df_to_check.index, desc="Checking compositional novelty"):
        struct = structures[df_gen.index.get_loc(idx)]
        if struct is None:
            is_comp_novel_list.append(False)
            continue
        
        # Reduce to the smallest whole-number formula so Si2O4 and SiO2 share a key
        try:
            comp_key = Composition(struct.composition).reduced_formula
        except Exception:
            is_comp_novel_list.append(False)
            continue
            
        # If this composition doesn't exist in training data, it's compositionally novel
        is_novel = comp_key not in base_comps or len(base_comps[comp_key]) == 0
        is_comp_novel_list.append(is_novel)
    
    df_gen["is_comp_novel"] = False
    df_gen.loc[df_to_check.index, "is_comp_novel"] = is_comp_novel_list
    
    comp_novel_count = df_gen['is_comp_novel'].sum()
    print(f"Found {comp_novel_count} compositionally novel structures.")
    return df_gen

# MP data for ehull calculations


# Property calculations
    
### Property prediction main function
def predict_properties(gen_df_proc: pd.DataFrame, property_targets: list, num_workers: int) -> pd.DataFrame:
    """Run property predictions (density via pymatgen)."""
    df_valid = gen_df_proc[gen_df_proc["is_valid"]].copy()

    for prop in property_targets:
        if 'density' in prop.lower() or 'den' in prop.lower():
            print("Getting density predictions...")
            gen_df_proc['gen_density (g/cm3)'] = df_valid["Generated CIF"].apply(get_density)

    return gen_df_proc


# CIF validation functions (adapted from CrystaLLM)










