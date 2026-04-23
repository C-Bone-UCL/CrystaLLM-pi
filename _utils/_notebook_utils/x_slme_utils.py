"""Helpers for notebooks/X_SLME.ipynb."""

import os
import sys

import pandas as pd
from pymatgen.core import Composition, Element
from smact.data_loader import lookup_element_hhis

from _utils._processing_utils import extract_reduced_formula as _shared_extract_reduced_formula


def parse_formula(formula_string: str) -> dict:
    composition = Composition(formula_string)
    return composition.get_el_amt_dict()


def hhi_scores_from_formula(formula: dict):
    masses = []
    hhi_p_vals = []
    hhi_r_vals = []

    for symbol, count in formula.items():
        atomic_weight = Element(symbol).atomic_mass
        hhi_data = lookup_element_hhis(symbol)

        if hhi_data is None:
            print(
                f"Warning: No HHI data for element {symbol} (excluded by source paper). "
                f"Cannot calculate HHI for material '{formula}'.",
                file=sys.stderr,
            )
            return None, None

        hhi_p, hhi_r = hhi_data
        if hhi_p is None or hhi_r is None:
            print(
                f"Warning: Incomplete HHI data for element {symbol}. "
                f"Cannot calculate HHI for material '{formula}'.",
                file=sys.stderr,
            )
            return None, None

        masses.append(count * atomic_weight)
        hhi_p_vals.append(hhi_p)
        hhi_r_vals.append(hhi_r)

    total_mass = sum(masses)
    if total_mass == 0:
        return None, None

    hhi_p_material = sum((mass / total_mass) * hhi for mass, hhi in zip(masses, hhi_p_vals))
    hhi_r_material = sum((mass / total_mass) * hhi for mass, hhi in zip(masses, hhi_r_vals))
    return hhi_p_material, hhi_r_material


def get_hhi_scores_from_cif(cif_str):
    try:
        formula_str = _shared_extract_reduced_formula(cif_str)
        formula_dict = parse_formula(formula_str)
        scores = hhi_scores_from_formula(formula_dict)
    except Exception as exc:
        print(f"Error processing CIF: {exc}", file=sys.stderr)
        print(f"Problematic CIF:\n{cif_str}\n", file=sys.stderr)
        scores = (None, None)
    return scores


def extract_formula(cif_str):
    try:
        return _shared_extract_reduced_formula(cif_str)
    except Exception:
        return "UnknownFormula"


def parse_novelty_from_tag(tag):
    struct_nov = []
    comp_nov = []

    if "Novel-ft-pt" in tag:
        struct_nov = ["ft", "pt"]
    elif "Novel-ft" in tag:
        struct_nov = ["ft"]
    elif "Novel-pt" in tag:
        struct_nov = ["pt"]

    if "CompNovel-ft-pt" in tag:
        comp_nov = ["ft", "pt"]
    elif "CompNovel-ft" in tag:
        comp_nov = ["ft"]
    elif "CompNovel-pt" in tag:
        comp_nov = ["pt"]

    struct_str = "-".join(struct_nov) if struct_nov else "None"
    comp_str = "-".join(comp_nov) if comp_nov else "None"
    return struct_str, comp_str


def build_novelty_tag(row):
    tags = []

    if row["is_novel"] and row["is_novel_pt"]:
        tags.append("Novel-ft-pt")
    elif row["is_novel"]:
        tags.append("Novel-ft")
    elif row["is_novel_pt"]:
        tags.append("Novel-pt")

    if row["is_comp_novel"] and row["is_comp_novel_pt"]:
        tags.append("CompNovel-ft-pt")
    elif row["is_comp_novel"]:
        tags.append("CompNovel-ft")
    elif row["is_comp_novel_pt"]:
        tags.append("CompNovel-pt")

    return "__".join(tags) if tags else "NotNovel"


def _material_summary_record(
    row: pd.Series,
    *,
    formula: str,
    position: int,
    metric_name: str,
    struct_nov: str,
    comp_nov: str,
) -> dict:
    return {
        "Reduced Formula": formula,
        "Position": position,
        "Metric": metric_name,
        "Pred. SLME": row["predicted_slme"],
        "HHI_p": row["HHI_p"],
        "HHI_r": row["HHI_r"],
        "HHI_dist": row["HHI_distance_to_0"],
        "E_hull_mace (eV/atom)": row["ehull_mace_mp"],
        "Structure Nov.": struct_nov,
        "Composition Nov.": comp_nov,
    }


def select_top_materials(df, top_n_slme=15, top_n_sustain=15, slme_threshold=25):
    materials = {}
    summary_records = []

    selection_specs = (
        ("SLME", df.nlargest(top_n_slme, "predicted_slme"), "SLME-{slme}"),
        (
            "HHI-SLME",
            df[df["predicted_slme"] > slme_threshold].nsmallest(top_n_sustain, "HHI_distance_to_0"),
            "Sustain-SLME-{slme}",
        ),
    )

    for metric_name, selected_df, name_template in selection_specs:
        for position, (_, row) in enumerate(selected_df.iterrows(), start=1):
            formula = extract_formula(row["Generated CIF"])
            novelty = build_novelty_tag(row)
            material_name = (
                f"{formula}__{name_template.format(slme=int(row['predicted_slme']))}"
                f"_top_{position}__{novelty}"
            )
            materials[material_name] = row["Generated CIF"]

            struct_nov, comp_nov = parse_novelty_from_tag(novelty)
            summary_records.append(
                _material_summary_record(
                    row,
                    formula=formula,
                    position=position,
                    metric_name=metric_name,
                    struct_nov=struct_nov,
                    comp_nov=comp_nov,
                )
            )

    summary_df = pd.DataFrame(summary_records)
    return materials, summary_df


def export_materials(materials, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for name, cif_str in materials.items():
        filepath = os.path.join(output_dir, f"{name}.cif")
        with open(filepath, "w") as handle:
            handle.write(cif_str)

    print(f"Exported {len(materials)} materials to {output_dir}/")


def run_material_selection(
    input_parquet,
    output_dir,
    output_csv,
    top_n_slme=15,
    top_n_sustain=15,
    slme_threshold=25,
):
    df = pd.read_parquet(input_parquet)
    materials, summary_df = select_top_materials(
        df,
        top_n_slme=top_n_slme,
        top_n_sustain=top_n_sustain,
        slme_threshold=slme_threshold,
    )

    export_materials(materials, output_dir)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved summary dataframe to {output_csv}")
    return materials, summary_df


__all__ = [
    "build_novelty_tag",
    "export_materials",
    "extract_formula",
    "get_hhi_scores_from_cif",
    "hhi_scores_from_formula",
    "parse_formula",
    "parse_novelty_from_tag",
    "run_material_selection",
    "select_top_materials",
]