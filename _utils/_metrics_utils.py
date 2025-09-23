"""
Utility functions for computing Validity, Uniqueness, Novelty (VUN), and other
metrics for CrystaLLMv2. Optimized for parallel processing.
"""
import argparse
import os
import re
import signal
import sys
import warnings
import argparse
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import sys
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from typing import Dict, Generator, Tuple
import os
import json
import concurrent.futures

import pandas as pd
from datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN

from _utils import (
    extract_volume,
    extract_formula_units,
    extract_data_formula,
)
from _utils._generating.postprocess import process_dataframe

logger = logging.getLogger(__name__)

#####################################
# Loading and processing Generated Data
#####################################

def load_and_process_generated_data(gen_data_path, num_workers):
    """Load and process generated CIF data."""
    print("\nLoading & Processing Generated CIFs")
    print(f"Loading generated data from {gen_data_path}...")
    
    gen_df = pd.read_parquet(gen_data_path)
    
    # Ensure condition column exists for downstream sorting
    if 'condition_vector' not in gen_df.columns and 'Condition Vector' not in gen_df.columns:
        print("No condition column found. Creating 'condition_vector' with value -100.")
        gen_df['condition_vector'] = -100
    
    if "Generated CIF" not in gen_df.columns:
        raise ValueError("Input DataFrame must contain 'Generated CIF' column.")

    print("Processing the generated CIFs...")
    return process_dataframe(gen_df, num_workers=num_workers, column_name='Generated CIF')


def build_generated_structures(df_proc):
    """Build pymatgen Structure objects for valid generated CIFs."""
    structures = [None] * len(df_proc)
    
    # This part is fast, so no need to parallelize
    for idx in tqdm(df_proc.index, desc="Building generated structures"):
        if df_proc.at[idx, "is_valid"]:
            try:
                cif_str = df_proc.at[idx, "Generated CIF"]
                structures[idx] = Structure.from_str(cif_str, fmt="cif")
            except Exception:
                # If structure building fails here, it's not valid
                df_proc.at[idx, "is_valid"] = False
    
    return structures


def extract_generated_formulas(structures):
    """Extract unique reduced formulas from a list of pymatgen structures."""
    print("Extracting unique reduced formulas from generated structures...")
    
    formulas = set()
    for struct in tqdm(structures, desc="Extracting reduced formulas"):
        if struct is not None:
            try:
                formulas.add(struct.composition.reduced_formula)
            except Exception:
                continue
    
    print(f"Found {len(formulas)} unique reduced formulas in generated set")
    return formulas


#####################################
# VUN Metrics Functions
#####################################

def _validity_worker(cif_str: str) -> bool:
    """Worker function to check if a single CIF string is valid."""
    if not cif_str or not isinstance(cif_str, str):
        return False
    try:
        # is_valid contains multiple checks (formula, multiplicity, bonds, etc.)
        return is_valid(cif_str, bond_length_acceptability_cutoff=1.0)
    except Exception:
        return False

def get_valid(df_proc, num_workers):
    """Compute validity for each CIF in parallel."""
    print("\nValidity Metrics")
    
    max_workers_recommended = 16
    max_workers = min(num_workers, max_workers_recommended)
    cifs_to_check = df_proc["Generated CIF"].tolist()
    
    results = [False] * len(df_proc)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use map to apply the worker function to each CIF string
        future_to_cif = executor.map(_validity_worker, cifs_to_check)
        
        # tqdm shows progress as results are completed
        results = list(tqdm(future_to_cif, total=len(cifs_to_check), desc="Checking validity"))

    df_proc["is_valid"] = results
    valid_count = df_proc['is_valid'].sum()
    
    print(f"{valid_count} valid CIFs out of {len(df_proc)} total")
    print(f"Validity rate: {valid_count / len(df_proc) * 100:.2f}%")
    
    return df_proc


def _uniqueness_worker(args_tuple: Tuple[int, str, float]) -> Tuple[int, str, float]:
    """Worker to compute BAWL hash and get a metric for uniqueness selection."""
    from material_hasher.hasher.bawl import BAWLHasher
    idx, cif_str, ehull_val = args_tuple
    
    try:
        struct = Structure.from_str(cif_str, fmt="cif")
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

def get_unique(df_gen, workers):
    """Identify unique structures based on BAWL hashing."""
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
        tasks = zip(df_valid.index, cifs_to_check, ehull_values)
        results = executor.map(_uniqueness_worker, tasks)
        
        # Process results to find unique structures using iterative method
        is_unique = pd.Series(False, index=df_gen.index)
        best_indices, best_metrics = {}, {}
        
        for idx, hash_val, metric_val in tqdm(results, total=len(df_valid), desc="Finding best unique structures"):
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


def _novelty_worker(args_tuple):
    """Worker to check if a single generated structure is novel."""
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
            ref_struct = Structure.from_str(ref_cif_str, fmt="cif")
            if matcher.fit(gen_struct, ref_struct):
                return False  # Found a match, so it's not novel
        except Exception:
            continue  # Ignore faulty reference CIFs

    return True # No match found after checking all references

def get_novelty(df_gen, base_comps, ltol, stol, angle_tol, structures, workers):
    """Check novelty of generated structures against a reference dataset."""
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
        comp_key = struct.composition.reduced_formula if struct else None
        tasks.append((struct, comp_key, base_comps, ltol, stol, angle_tol))

    # Run novelty checks in parallel
    results = [False] * len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = executor.map(_novelty_worker, tasks)
        results = list(tqdm(future_to_task, total=len(tasks), desc="Checking novelty"))

    # Assign results back to the dataframe
    df_gen["is_novel"] = False
    df_gen.loc[df_to_check.index, "is_novel"] = results
    
    novel_count = df_gen['is_novel'].sum()
    print(f"Found {novel_count} novel CIFs.")
    return df_gen

#####################################
# Novelty check helpers
#####################################

def load_and_filter_training_data(hf_dataset, processed_data_path, num_workers, gen_formulas):
    """Load training dataset and filter it to include only relevant compositions."""
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

def build_reference_compositions(proc_train_df, gen_formulas):
    """Build a dictionary of reference CIFs, grouped by composition."""
    print("Filtering training dataset to compositions present in generated set...")
    
    base_comps = defaultdict(list)
    
    # Filter the dataframe first for efficiency
    relevant_train_df = proc_train_df[proc_train_df['Reduced Formula'].isin(gen_formulas)]
    
    for _, row in tqdm(relevant_train_df.iterrows(), total=len(relevant_train_df), desc="Grouping training CIFs"):
        cif_string = row.get('CIF')
        comp_key = row.get('Reduced Formula')
        
        if cif_string and comp_key:
            base_comps[comp_key].append(cif_string)
            
    print(f"Filtered training set to {len(relevant_train_df)} CIFs relevant for novelty check.")
    return base_comps

################################
# Reference Data for Ehull calcs
################################

def download_mp_data(mp_data_path):
    """Download MP computed structure entries if file doesn't exist"""
    if Path(mp_data_path).exists():
        print(f"MP data file already exists: {mp_data_path}")
        return
    
    print("MP data file not found. Downloading from matbench_discovery...")
    url = "https://figshare.com/ndownloader/files/40344436" 
    
    print("Downloading MP computed structure entries...")
    print("File size: ~170 MB compressed...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(mp_data_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading MP data") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded {Path(mp_data_path).stat().st_size / 1024 / 1024:.1f} MB to {mp_data_path}")

class MPDataProvider:
    """Efficient MP data provider that builds phase diagrams on-demand for specific chemical systems"""
    def __init__(self, mp_data_path):
        print("Loading MP entries...")
        df_mp = pd.read_json(mp_data_path)
        
        # Filter to only GGA entries (like reference script)
        if 'index' in df_mp.columns:
            df_mp = df_mp[df_mp['index'].str.contains("GGA")]
        print(f"Found {len(df_mp)} MP entries")
        
        # Convert to ComputedEntry objects and organize by chemical system
        print("Organizing MP entries by chemical system...")
        self.entries_by_chemsys = {}
        
        for entry_dict in tqdm(df_mp.entry, desc="Processing MP entries"):
            if 'GGA' in entry_dict['parameters']['run_type']:
                entry = ComputedEntry.from_dict(entry_dict)
                # Filter out R2SCAN entries (exactly like mace_ehull_copy.py)
                if not np.any(['R2SCAN' in a.name for a in entry.energy_adjustments]):
                    # Get chemical system (sorted elements)
                    elements = sorted(entry.composition.elements)
                    chemsys = tuple(str(el) for el in elements)
                    
                    if chemsys not in self.entries_by_chemsys:
                        self.entries_by_chemsys[chemsys] = []
                    self.entries_by_chemsys[chemsys].append(entry)
        
        # Cache for built phase diagrams
        self.pd_cache = {}
        print(f"Organized {sum(len(v) for v in self.entries_by_chemsys.values())} entries across {len(self.entries_by_chemsys)} chemical systems")
    
    def get_phase_diagram(self, elements):
        """Get or build phase diagram for a specific chemical system"""
        chemsys = tuple(sorted(str(el) for el in elements))
        
        if chemsys in self.pd_cache:
            return self.pd_cache[chemsys]
        
        # Get ALL entries that contain ONLY these elements (like MP API's get_entries_in_chemsys)
        # This includes pure element entries and all combinations
        relevant_entries = []
        element_set = set(str(el) for el in elements)
        
        for system, entries in self.entries_by_chemsys.items():
            # Check if this chemical system is a subset of our target elements
            system_elements = set(system)
            if system_elements.issubset(element_set):
                relevant_entries.extend(entries)
        
        if len(relevant_entries) < 2:
            # Not enough entries to build a meaningful phase diagram
            self.pd_cache[chemsys] = None
            return None
        
        # Build phase diagram with all relevant entries
        pd_sys = PhaseDiagram(relevant_entries)
        self.pd_cache[chemsys] = pd_sys
        
        return pd_sys

    def compute_ehull_and_eform(self, structure: Structure, energy_eV: float):
        """Compute e_above_hull exactly like mace_ehull_copy.py"""
        elements = sorted({el.symbol for el in structure.composition.elements})
        pd_sys = self.get_phase_diagram(elements)
        
        if pd_sys is None:
            return np.nan, np.nan

        entry = ComputedEntry(composition=structure.composition, energy=energy_eV)

        # Apply MP2020 corrections only to user entry (same as mace_ehull_copy.py)
        compat = MaterialsProject2020Compatibility(check_potcar=False)
        try:
            entry.parameters["software"] = "non-vasp"
            entry.parameters["run_type"] = "GGA"
            entry = compat.process_entry(entry, clean=True)
            if entry is None:
                return np.nan, np.nan
        except Exception:
            return np.nan, np.nan

        eh = pd_sys.get_e_above_hull(entry, allow_negative=True)
        eform_pa = pd_sys.get_form_energy_per_atom(entry)
        return eh, eform_pa

################################
# Property metrics
################################

### Density
def get_density(cif):
    """Calculate density from CIF string, returning NaN on any error."""
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            structure = Structure.from_str(cif, fmt='cif')
            for warn in w:
                if ("Incorrect stoichiometry" in str(warn.message)):
                    return np.nan
            return structure.density
    except Exception:
        return np.nan
    
### Bandgap with ALIGNN
def _predict_bandgap(df_valid, num_workers):
    """Predict bandgap using ALIGNN for valid structures."""
    print("Getting predictions for band gaps...")

    # Add project root to path for imports
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from _utils._metrics.ALIGNN_props import _parse_single_cif, get_multiple_predictions
    except Exception as e:
        print(f"Error: {e}")
        print("ALIGNN not installed correctly in this environment.")
        sys.exit(1)
    
    # Parse CIFs for ALIGNN prediction
    data_to_parse = [(idx, row["Generated CIF"]) for idx, row in df_valid.iterrows()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        parsed_iter = executor.map(_parse_single_cif, data_to_parse)
        parsed_results = list(tqdm(parsed_iter, total=len(data_to_parse), desc="Parsing CIFs for ALIGNN"))
    
    # Extract valid atoms and their indices
    atoms_list = [atoms for _, atoms in parsed_results if atoms is not None]
    jids_list = [idx for idx, atoms in parsed_results if atoms is not None]
    
    # Get ALIGNN predictions
    bg_results = get_multiple_predictions(
        atoms_array=atoms_list, jids=jids_list, cutoff=8, max_neighbors=12,
        model_name="mp_gappbe_alignn", batch_size=1, workers=num_workers,
    )
    return pd.Series(bg_results)

### Density and Bandgap main functions
def _predict_density(df_valid):
    """Calculate density for valid structures."""
    print("Getting density predictions...")
    return df_valid["Generated CIF"].apply(get_density)

def predict_properties(gen_df_proc, property_targets, num_workers):
    """Predict properties using ALIGNN and density calculations."""
    df_valid = gen_df_proc[gen_df_proc["is_valid"]].copy()
    
    for prop in property_targets:
        if 'bandgap' in prop.lower() or 'bg' in prop.lower():
            bg_series = _predict_bandgap(df_valid, num_workers)
            gen_df_proc['ALIGNN_bg (eV)'] = bg_series
        elif 'density' in prop.lower() or 'den' in prop.lower():
            density_series = _predict_density(df_valid)
            gen_df_proc['gen_density (g/cm3)'] = density_series
    
    return gen_df_proc


#####################################
# Basic CIF evaluation functions
# Adapted from original CrystaLLM repo: https://github.com/lantunes/CrystaLLM
#####################################

def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            if electronegativity_diff >= 1.7:
                # use ionic radii
                if site.specie.X < connected_site.specie.X:
                    expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                else:
                    expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
            else:
                expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count if bond_count > 0 else 0

    return normalized_score


def is_space_group_consistent(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()

    return is_match


def is_formula_consistent(cif_str):
    parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula


def is_atom_site_multiplicity_consistent(cif_str):
    # Parse the CIF string
    parser = CifParser.from_str(cif_str)
    cif_data = parser.as_dict()

    # Extract the chemical formula sum from the CIF data
    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]

    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    # Validate if the expected and actual atom counts match
    return expected_atoms == actual_atoms


def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True

def is_valid(cif_str, bond_length_acceptability_cutoff=1.0, debug=False):
    if not is_formula_consistent(cif_str):
        if debug:
            print(f"Formula is inconsistent for {cif_str}")
        return False
    if not is_atom_site_multiplicity_consistent(cif_str):
        if debug:
            print(f"Atom site multiplicity is inconsistent for {cif_str}")
        return False
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
        if debug:
            print(f"Bond length is unreasonable for {cif_str}")
        return False
    if not is_space_group_consistent(cif_str):
        if debug:
            print(f"Space group is inconsistent for {cif_str}")
        return False
    return True