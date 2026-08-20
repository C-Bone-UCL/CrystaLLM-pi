"""Provide Materials Project reference data for hull calculations.

The module isolates Materials Project download and caching because these
operations are slow and are only required by stability metrics.
"""

import os

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from tqdm import tqdm


class MPDataProvider:
    """Provide Materials Project data with on-demand phase-diagram construction."""
    def __init__(self, mp_data_path: str) -> None:
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
    
    def get_phase_diagram(self, elements: frozenset) -> object:
        """Return or construct the phase diagram for a chemical system."""
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

    def compute_ehull_and_eform(self, structure: Structure, energy_eV: float) -> tuple:
        """Compute energy above hull and formation energy."""
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


def download_mp_data(mp_data_path: str) -> None:
    """Download MP data if needed."""
    if Path(mp_data_path).exists():
        print(f"MP data file already exists: {mp_data_path}")
        return
    
    print("MP data file not found. Downloading from matbench_discovery...")
    url = "https://ndownloader.figshare.com/files/40344436"
    
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
