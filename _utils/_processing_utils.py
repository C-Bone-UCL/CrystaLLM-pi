"""
General utilities for processing CIF files and and extracting structural information.
"""

import math
import re
import pandas as pd
import os
import json
import numpy as np
import logging
import re
import _tokenizer
from importlib import reload
import commentjson
from io import StringIO

reload(_tokenizer)

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser

logger = logging.getLogger(__name__)

# Adapted from original CrystaLLM repo: https://github.com/lantunes/CrystaLLM

def get_unit_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha_rad = math.radians(alpha_deg)
    beta_rad = math.radians(beta_deg)
    gamma_rad = math.radians(gamma_deg)

    volume = (a * b * c * math.sqrt(1 - math.cos(alpha_rad) ** 2 - math.cos(beta_rad) ** 2 - math.cos(gamma_rad) ** 2 +
                                    2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)))
    return volume


def get_atomic_props_block_for_formula(formula, oxi=False):
    comp = Composition(formula)
    return get_atomic_props_block(comp, oxi)


def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        return f"{float(val): .4f}"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return allen_electronegativity[str(elem)]
        return _format(elem.X)

    def _format_radius(elem):
        if elem.atomic_radius is None and str(elem) in noble_vdw_radii:
            return noble_vdw_radii[str(elem)]
        return _format(elem.atomic_radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format(el.average_ionic_radius))
             for el in sorted(composition.elements)}

    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    # use the average ionic radius
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {str(el): (float(el.oxi_state), _format(el.ionic_radius)) for el in sorted(composition.elements)}
        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        # if we know the oxidation state of the element, use the ionic radius for the given oxidation state
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]

    return str(CifBlock(data, loops, "")).replace("data_\n", "")


def replace_symmetry_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    # Convert each symmetry operator to a string like "x, y, z" etc.
    ops = [op.as_xyz_str() for op in symmops]

    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops
    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    # Construct a CIF block that contains the new symmetry operators
    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    # Pattern allowing for optional whitespace. We use (?m) so ^ matches start-of-line.
    pattern = r"(?m)(^loop_\s*\n\s*_symmetry_equiv_pos_site_id\s*\n\s*_symmetry_equiv_pos_as_xyz\s*\n\s*1\s*'x,\s*y,\s*z')"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated


def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        # If group(2) exists => it's the content inside single quotes;
        # otherwise group(3) => unquoted
        # print(f"match.group(2): {match.group(2)}")
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")


def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")


def extract_volume(cif_str):
    return extract_numeric_property(cif_str, "_cell_volume")


def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)


def extract_data_formula(cif_str):
    # match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    match = re.search(r"data_(\S+)", cif_str)
    if match:
        return match.group(1)
    raise Exception(f"could not find data_ in:\n{cif_str}")


def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")


def semisymmetrize_cif(cif_str):
    return re.sub(
        r"(_symmetry_equiv_pos_as_xyz\n)(.*?)(?=\n(?:\S| \S))",
        r"\1  1  'x, y, z'",
        cif_str,
        flags=re.DOTALL
    )


def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")


def add_atomic_props_block(cif_str, oxi=False):
    comp = Composition(extract_formula_nonreduced(cif_str))

    block = get_atomic_props_block(composition=comp, oxi=oxi)

    # the hypothesis is that the atomic properties should be the first thing
    #  that the model must learn to associate with the composition, since
    #  they will determine so much of what follows in the file
    pattern = r"_symmetry_space_group_name_H-M"
    match = re.search(pattern, cif_str)

    if match:
        start_pos = match.start()
        modified_cif = cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return modified_cif
    else:
        raise Exception(f"Pattern not found: {cif_str}")


def remove_atom_props_block(cif: str) -> str:
    """Strict removal of atomic properties block"""
    pattern = re.compile(
        r"(loop_\s*_atom_type_symbol.*?)(?=_symmetry_|loop_|_cell_)",
        flags=re.DOTALL
    )
    return re.sub(pattern, '', cif)


def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded


def array_split(arr, num_splits):
    split_size, remainder = divmod(len(arr), num_splits)
    splits = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (i < remainder)
        splits.append(arr[start:end])
        start = end
    return splits


def embeddings_from_csv(embedding_csv):
    df = pd.read_csv(embedding_csv)
    elements = list(df["element"])
    df.drop(["element"], axis=1, inplace=True)
    embeds_array = df.to_numpy()
    embedding_data = {
        elements[i]: embeds_array[i] for i in range(len(embeds_array))
    }
    return embedding_data

def remove_comments(cif_str: str) -> str:
    """
    Removes comments preceding the 'data_' block in a CIF string.
    """
    match = re.search(r'(data_.*)', cif_str, re.DOTALL)
    if match:
        return match.group(1)

    return cif_str  # Return as-is if no 'data_' block is found


def safe_filename(name: str) -> str:
    """Convert string to safe filename by replacing invalid characters."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", name)
    return name or "entry"

def load_mp_api_key(key_file: str) -> str:
    """Load MP API key from JSONC file - DEPRECATED: now using MP data file instead."""
    # This function is kept for backwards compatibility but is no longer used
    try:
        with open(key_file, "r") as f:
            data = commentjson.load(f)
        mp_key = str(data["MP_key"]).strip()
        if not mp_key:
            raise KeyError("MP_key empty in API key file")
        return mp_key
    except Exception as e:
        raise RuntimeError(f"Failed to read MP API key from '{key_file}': {e}")
    

# make a function that loads key dictionary from API_keys.jsonc
def load_api_keys(key_file: str = "API_keys.jsonc") -> dict:
    """Load API keys from JSONC file."""
    try:
        with open(key_file, "r") as f:
            data = commentjson.load(f)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read API keys from '{key_file}': {e}")



def _snap_site_to_int(site, ROUND_TOL=0.05):
    """
    Return (species, coords) if the site's majority species occupancy
    can be snapped to 1 or 0 within ROUND_TOL, else None.
    """
    sp, occ = max(site.species.items(), key=lambda kv: kv[1])
    if abs(occ - 1) <= ROUND_TOL:
        return sp, 1.0, site.frac_coords
    if abs(occ - 0) <= ROUND_TOL:
        return None                           
    return "FAIL"


def _round_structure(struct, ROUND_TOL=0.05):
    """
    Try to round *all* sites in the existing unit cell.
    Returns a new Structure or None on failure.
    """
    new_species, new_coords = [], []
    for site in struct:
        snapped = _snap_site_to_int(site, ROUND_TOL=ROUND_TOL)
        if snapped == "FAIL":
            return None
        if snapped is not None:               # keep full-occupancy site
            sp, _, fc = snapped
            new_species.append(sp)
            new_coords.append(fc)

    new_struct = Structure(struct.lattice, new_species, new_coords)
    return new_struct


def order_or_round_cif(cif_str, MANDATORY_ELEMENTS=None, ROUND_TOL=0.05):
    """
    Make every occupancy exactly 1 (or remove site) *without*
    expanding the cell.  Preserve element set; otherwise
    return None so the caller can skip the CIF.
    """
    try:
        struct = CifParser(StringIO(cif_str)).get_structures(primitive=False)[0]
        comp0  = struct.composition

        if all(abs(v - round(v)) < 1e-6 for v in comp0.values()):
            # already integer stoichiometry
            return cif_str

        rounded = _round_structure(struct, ROUND_TOL=ROUND_TOL)
        if rounded is None:
            return None

        comp1 = rounded.composition
        # element-set check
        if MANDATORY_ELEMENTS and set(comp1.elements) != set(comp0.elements):
            return None

        new_cif = rounded.to(fmt="cif")
        print(f"Non-rounded formula: {comp0}")
        print(f"Rounded  formula: {comp1}")
        return new_cif

    except Exception as e:
        print("Rounding failed:", e)
        return None
    

def normalize_property_column(dataframe, prop_name, norm_method):
    """Apply normalization to a single property column."""
    if prop_name not in dataframe.columns:
        print(f"Warning: Property column '{prop_name}' not found in dataframe")
        return dataframe
    
    # Round values first
    dataframe[prop_name] = dataframe[prop_name].apply(lambda x: round(x, 3))
    
    if norm_method == "power_log":
        print(f"\nNormalizing with power log method for {prop_name} (beta = 0.8)...")
        max_val = dataframe[prop_name].max()
        print(f"Max value of {prop_name}: {max_val}")
        if max_val == 0:
            norm_values = dataframe[prop_name]
        else:
            norm_values = (np.log(1 + dataframe[prop_name]) / np.log(1 + max_val)) ** 0.8
        dataframe["norm_" + prop_name] = norm_values.round(4)
        print(f"Max value of power log {prop_name}: {norm_values.max()}")
        print(f"Min value of power log {prop_name}: {norm_values.min()}")

    elif norm_method == "signed_log":
        print(f"\nNormalizing with signed log method for {prop_name}...")
        max_val = dataframe[prop_name].max()
        min_val = dataframe[prop_name].min()
        print(f"Max value of {prop_name}: {max_val}")
        print(f"Min value of {prop_name}: {min_val}")

        def signed_log(x):
            return np.sign(x) * np.log1p(np.abs(x))

        signed_values = signed_log(dataframe[prop_name])
        signed_max = signed_values.max()
        signed_min = signed_values.min()

        normed_values = (signed_values - signed_min) / (signed_max - signed_min)
        normed_values = normed_values ** 0.8

        print(f"Max value of signed log {prop_name}: {normed_values.max()}")
        print(f"Min value of signed log {prop_name}: {normed_values.min()}")

        dataframe["norm_" + prop_name] = normed_values.round(4)

    elif norm_method == "linear":
        print(f"\nNormalizing with linear method for {prop_name}...")
        min_val = dataframe[prop_name].min()
        max_val = dataframe[prop_name].max()
        print(f"Min value of {prop_name}: {min_val}")
        min_val = 0
        print(f"Setting min value of {prop_name} to 0 for linear normalization")

        print(f"Max value of {prop_name}: {max_val}")
        if max_val - min_val == 0:
            norm_values = dataframe[prop_name]
        else:
            norm_values = (dataframe[prop_name] - min_val) / (max_val - min_val)

        print(f"Max value of linear {prop_name}: {norm_values.max()}")
        print(f"Min value of linear {prop_name}: {norm_values.min()}")

        dataframe["norm_" + prop_name] = norm_values.round(4)

    elif norm_method == "log10":
        print(f"\nNormalizing with log10 method for {prop_name}...")
        print(f"Max value of {prop_name}: {dataframe[prop_name].max()}")
        print(f"Min value of {prop_name}: {dataframe[prop_name].min()}")

        # compute log10 and clamp invalid values to 0.0
        vals = [np.log10(float(x)) for x in dataframe[prop_name]]
        vals = [
            0.0 if np.isneginf(v) or np.isnan(v) or np.isinf(v) else v
            for v in vals
        ]
        dataframe["norm_" + prop_name] = np.round(vals, 4)

        print(f"Max value of log10 {prop_name}: {dataframe['norm_' + prop_name].max()}")
        print(f"Min value of log10 {prop_name}: {dataframe['norm_' + prop_name].min()}")
    
    return dataframe


def add_variable_brackets_to_cif(cif_str):
    """Add brackets to variable fields in CIF strings."""
    lines = cif_str.splitlines()
    new_lines = []
    i = 0
    constant_loop_keys = {"_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"}

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("data_"):
            prefix = "data_"
            rest = stripped[len(prefix):].strip()
            if not (rest.startswith("[") and rest.endswith("]")):
                new_line = prefix + "[" + rest + "]"
            else:
                new_line = line
            new_lines.append(new_line)
            i += 1

        elif stripped.startswith("loop_"):
            new_lines.append(line)
            i += 1
            headers = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                headers.append(lines[i])
                new_lines.append(lines[i])
                i += 1
            is_constant = any(header.split()[0] in constant_loop_keys for header in headers)
            if not is_constant:
                new_lines.append("[")
            while i < len(lines):
                current_line = lines[i]
                current_stripped = current_line.strip()
                if (current_stripped.startswith("data_") or
                    current_stripped.startswith("loop_") or
                    current_stripped.startswith("_")):
                    break
                new_lines.append(current_line)
                i += 1
            if not is_constant:
                new_lines.append("]")

        elif stripped.startswith("_"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                value_stripped = value.strip()
                if key == "_chemical_formula_sum":
                    if (value_stripped.startswith("'") and value_stripped.endswith("'")) or \
                       (value_stripped.startswith('"') and value_stripped.endswith('"')):
                        value_stripped = value_stripped[1:-1].strip()
                    value = "'[" + value_stripped + "]'"
                else:
                    if not ((value_stripped.startswith("[") and value_stripped.endswith("]")) or
                            (value_stripped.startswith("'") and value_stripped.endswith("'")) or
                            (value_stripped.startswith('"') and value_stripped.endswith('"'))):
                        value = "[" + value_stripped + "]"
                new_line = key + " " + value
            else:
                new_line = line
            new_lines.append(new_line)
            i += 1

        else:
            new_lines.append(line)
            i += 1

    return "\n".join(new_lines)