"""Helpers and visualizers for the SLME discovery pipeline notebook.

Handles dataset preparation, HHI sustainability metrics, novelty tagging, 
candidate selection, and distribution plotting for generated vs training materials.
"""

import json
import math
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from pymatgen.core import Composition, Element
from pymatgen.io.jarvis import JarvisAtomsAdaptor as JAA
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.core.atoms import Atoms
from smact.data_loader import lookup_element_hhis

from _utils._processing_utils import extract_reduced_formula

# Dataset Building and Preprocessing

def _symmetrize_cif(struct):
    """Convert structure to symmetrized CIF format with error handling."""
    sga = SpacegroupAnalyzer(struct)
    symm_struct = sga.get_symmetrized_structure()
    return str(CifWriter(symm_struct, symprec=0.1))


def structure_to_cif(struct):
    """Convert a pymatgen Structure to CIF format w/ symmetry."""
    return _symmetrize_cif(struct)


def extract_formula(struct):
    """Extract the reduced formula from a pymatgen Structure."""
    return struct.composition.reduced_formula


def build_finetuning_dataset(structure_path: str, slme_path: str, output_parquet: str):
    """Constructs the fine-tuning dataset from raw JSON dictionaries.
    
    Replaces the notebook's iterative concatenation with a faster list accumulation.
    """
    with open(structure_path, 'r') as f:
        structure_dict = json.load(f)
    with open(slme_path, 'r') as f:
        slme_dict = json.load(f)

    records = []
    for material_id, struct_dict in tqdm(structure_dict.items(), desc="Processing structures"):
        ats = Atoms.from_dict(struct_dict)
        structure = JAA.get_structure(ats)
        cif_str = structure_to_cif(structure)
        formula = extract_formula(structure)
        
        records.append({
            'Material ID': material_id,
            'Reduced Formula': formula,
            'CIF': cif_str,
            'SLME': slme_dict.get(material_id)
        })

    df = pd.DataFrame(records)
    df.to_parquet(output_parquet, index=False)
    print(f"Dataset successfully built and saved to {output_parquet}")
    return df

# Metric Calculations and Data Processing

def get_hhi_scores_from_cif(cif_str: str):
    # Parses formula and calculates HHI and Euclidean distance (Eq 1)
    try:
        formula_str = extract_reduced_formula(cif_str)
        comp = Composition(formula_str).get_el_amt_dict()
        
        masses = []
        hhi_p_vals = []
        hhi_r_vals = []
        
        for symbol, count in comp.items():
            atomic_weight = Element(symbol).atomic_mass
            hhi_data = lookup_element_hhis(symbol)
            
            # Catch missing or incomplete smact data
            if not hhi_data or hhi_data[0] is None or hhi_data[1] is None:
                print(f"Warning: Incomplete HHI data for {symbol}. Skipping HHI for {formula_str}.", file=sys.stderr)
                return None, None, None
                
            hhi_p_vals.append(hhi_data[0])
            hhi_r_vals.append(hhi_data[1])
            masses.append(count * atomic_weight)
            
        total_mass = sum(masses)
        if total_mass == 0:
            return None, None, None
            
        hhi_p_mat = sum((m / total_mass) * p for m, p in zip(masses, hhi_p_vals))
        hhi_r_mat = sum((m / total_mass) * r for m, r in zip(masses, hhi_r_vals))
        
        # Calculate Eq 1: Euclidean distance from origin
        hhi_dist = math.hypot(hhi_p_mat, hhi_r_mat)
        
        return hhi_p_mat, hhi_r_mat, hhi_dist
        
    except Exception as e:
        print(f"Error processing CIF for HHI: {e}", file=sys.stderr)
        return None, None, None


def build_novelty_tag(row: pd.Series) -> str:
    # Dynamically builds novelty tags based on boolean columns
    struct_tags = []
    comp_tags = []
    
    if row.get("is_novel"): struct_tags.append("ft")
    if row.get("is_novel_pt"): struct_tags.append("pt")
        
    if row.get("is_comp_novel"): comp_tags.append("ft")
    if row.get("is_comp_novel_pt"): comp_tags.append("pt")
        
    final_tags = []
    if struct_tags:
        final_tags.append(f"Novel-{'-'.join(struct_tags)}")
    if comp_tags:
        final_tags.append(f"CompNovel-{'-'.join(comp_tags)}")
        
    return "__".join(final_tags) if final_tags else "NotNovel"


def parse_novelty_from_tag(tag: str):
    # Splits out structural and compositional novelties
    parts = tag.split("__")
    struct_str = "None"
    comp_str = "None"
    
    for p in parts:
        if p.startswith("Novel-"):
            struct_str = p.replace("Novel-", "")
        elif p.startswith("CompNovel-"):
            comp_str = p.replace("CompNovel-", "")
            
    return struct_str, comp_str


def select_top_materials(df: pd.DataFrame, top_n_slme=15, top_n_sustain=15, slme_threshold=25):
    # Uses vectorized pandas filtering to isolate candidates
    materials = {}
    summary_records = []
    
    # Isolate top SLME candidates
    top_slme_df = df.nlargest(top_n_slme, "predicted_slme").copy()
    top_slme_df["_selection_metric"] = "SLME"
    top_slme_df["_name_template"] = "SLME-{slme}"
    
    # Isolate top sustainable candidates (filtered by threshold)
    sustain_candidates = df[df["predicted_slme"] > slme_threshold]
    top_sustain_df = sustain_candidates.nsmallest(top_n_sustain, "HHI_distance_to_0").copy()
    top_sustain_df["_selection_metric"] = "HHI-SLME"
    top_sustain_df["_name_template"] = "Sustain-SLME-{slme}"
    
    combined_df = pd.concat([top_slme_df, top_sustain_df])
    
    # Add a position rank based on their subgroup
    combined_df['Position'] = combined_df.groupby('_selection_metric').cumcount() + 1
    
    for _, row in combined_df.iterrows():
        try:
            formula = extract_reduced_formula(row["Generated CIF"])
        except Exception:
            formula = "UnknownFormula"
            
        novelty = build_novelty_tag(row)
        slme_val = int(row['predicted_slme']) if pd.notna(row['predicted_slme']) else 0
        
        template = row["_name_template"].format(slme=slme_val)
        position = row["Position"]
        mat_name = f"{formula}__{template}_top_{position}__{novelty}"
        
        materials[mat_name] = row["Generated CIF"]
        
        struct_nov, comp_nov = parse_novelty_from_tag(novelty)
        summary_records.append({
            "Reduced Formula": formula,
            "Position": position,
            "Metric": row["_selection_metric"],
            "Pred. SLME": row["predicted_slme"],
            "HHI_p": row.get("HHI_p"),
            "HHI_r": row.get("HHI_r"),
            "HHI_dist": row.get("HHI_distance_to_0"),
            "E_hull_mace (eV/atom)": row.get("ehull_mace_mp"),
            "Structure Nov.": struct_nov,
            "Composition Nov.": comp_nov,
        })
        
    return materials, pd.DataFrame(summary_records)


def run_material_selection(input_parquet: str, output_dir: str, output_csv: str, top_n_slme=15, top_n_sustain=15, slme_threshold=25):
    # Main runner for extracting the materials
    df = pd.read_parquet(input_parquet)
    materials, summary_df = select_top_materials(
        df, top_n_slme=top_n_slme, top_n_sustain=top_n_sustain, slme_threshold=slme_threshold
    )
    
    os.makedirs(output_dir, exist_ok=True)
    for name, cif_str in materials.items():
        with open(os.path.join(output_dir, f"{name}.cif"), "w") as f:
            f.write(cif_str)
            
    print(f"Exported {len(materials)} materials to {output_dir}/")
    
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved summary dataframe to {output_csv}")
    
    return materials, summary_df


# Plotting and Visualization

def plot_slme_distribution(gen_merged: pd.DataFrame, training_hse_gaps: pd.Series, training_slmes: pd.Series, output_path="SLME_plot.png"):
    # Generates a scatter distribution plot with marginal histograms
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3, height_ratios=[0.75, 3, 0.1], width_ratios=[3, 0.75, 0.1], 
                  hspace=0.25, wspace=0.25)
                  
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.scatter(gen_merged['predicted_gap'], gen_merged['predicted_slme'], 
                    color='steelblue', alpha=0.8, s=3, marker='o')
                    
    line1 = ax_main.axvline(x=1.3, color='r', linestyle='--', linewidth=2.5, 
                            label='Theoretical Best Bandgap [1.3 eV]')
    line2 = ax_main.axhline(y=33.2, color='orange', linestyle='--', linewidth=2.5, 
                            label='Target SLME')
                            
    ax_main.set_xlabel('Predicted band-gap [eV]', fontsize=18)
    ax_main.set_ylabel('Predicted SLME [%]', fontsize=18)
    ax_main.set_xlim(-1, 10)
    ax_main.tick_params(axis='both', which='major', labelsize=18)
    
    for spine in ax_main.spines.values():
        spine.set_linewidth(1.2)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(training_hse_gaps.values, bins=40, alpha=0.5, color='lightcoral', 
                density=True, label='Training')
    ax_top.hist(gen_merged['predicted_gap'].dropna(), bins=40, alpha=0.5, color='steelblue', 
                density=True, label='Generated')
    ax_top.axvline(x=1.3, color='r', linestyle='--', linewidth=2.5, alpha=1.0)
    ax_top.set_ylabel('Density', fontsize=18)
    ax_top.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.2)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(training_slmes.values, bins=40, alpha=0.5, color='lightcoral', 
                  density=True, orientation='horizontal', label='Training')
    ax_right.hist(gen_merged['predicted_slme'].dropna(), bins=40, alpha=0.5, color='steelblue', 
                  density=True, orientation='horizontal', label='Generated')
    ax_right.axhline(y=33.2, color='orange', linestyle='--', linewidth=2.5, alpha=1.0)
    ax_right.set_xlabel('Density', fontsize=18)
    ax_right.set_xticks([0.0, 0.3])
    ax_right.tick_params(axis='both', which='major', labelsize=18)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    
    for spine in ax_right.spines.values():
        spine.set_linewidth(1.2)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    from matplotlib.patches import Patch
    training_patch = Patch(facecolor='lightcoral', alpha=0.5, label='Training')
    generated_patch = Patch(facecolor='steelblue', alpha=0.5, label='Generated')
    
    legend_gs = fig.add_gridspec(1, 2, left=0.15, right=0.80, bottom=-0.09, top=-0.02, wspace=0.05)
    
    ax_leg1 = fig.add_subplot(legend_gs[0, 0])
    ax_leg1.legend([line1, line2], ['Ideal band-gap [1.3 eV]', 'Target SLME [33.2 %]'], loc='center', frameon=False, fontsize=14)
    ax_leg1.set_title('Reference Lines', fontsize=16, weight='bold', y=1.2)
    ax_leg1.axis('off')
    
    ax_leg2 = fig.add_subplot(legend_gs[0, 1])
    ax_leg2.legend([training_patch, generated_patch], ['Training', 'Generated'], loc='center', frameon=False, fontsize=14)
    ax_leg2.set_title('Data Sources', fontsize=16, weight='bold', y=1.2)
    ax_leg2.axis('off')
    
    num_high_slme = (gen_merged['predicted_slme'] >= 20).sum()
    print(f"Number of generated materials with SLME >= 20%: {num_high_slme}")
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        
    plt.show()


__all__ = [
    "build_finetuning_dataset",
    "build_novelty_tag",
    "extract_formula",
    "get_hhi_scores_from_cif",
    "parse_novelty_from_tag",
    "plot_slme_distribution",
    "run_material_selection",
    "select_top_materials",
    "structure_to_cif",
]