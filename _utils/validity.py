"""Provide structure-validity checks for generated CIFs.

The module is separate from ``metrics.py`` so generation can import validity
checks without loading Materials Project and MACE dependencies required by
hull metrics. ``is_valid`` is the composite validity check. ``is_sensible``
only examines cell parameters in the CIF text and is not part of ``is_valid``.
"""

import re
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from _utils.processing import (
    cif_parser,
    extract_data_formula,
    extract_formula_nonreduced,
    extract_space_group_symbol,
    extract_numeric_property,
)


def _configure_pymatgen_warning_filters() -> None:
    """Silence noisy pymatgen warnings that flood VUN/uniqueness logs."""
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*No oxidation states specified on sites!.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*CrystalNN: cannot locate an appropriate radius.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*No Pauling electronegativity for.*",
    )



def bond_length_reasonableness_score(cif_str: str, tolerance: float=0.32, h_factor: float=2.5) -> float | None:
    """Score bond lengths against the sum of covalent radii.

    The score is the fraction of bonds whose length lies within `tolerance` of the expected
    covalent-radii sum. A score of `1.0` means every counted bond passes. Neighbours are obtained
    with pymatgen's `CrystalNN`. Bonds involving hydrogen use a widened tolerance controlled by
    `h_factor`.

    Returns `None` for a disordered structure, meaning not checked rather than passed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        structure = cif_parser(cif_str).parse_structures(primitive=False)[0]

    # TODO: make function work for disordered materials. CrystalNN reads site.specie,
    # which raises on partial occupancies, as do the radii lookups below.
    if not structure.is_ordered:
        return None

    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
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

            # Penalise bond lengths that are too short or too long.
            # Hydrogen bonds use a looser tolerance.
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count if bond_count > 0 else 0

    return normalized_score


def is_space_group_consistent(cif_str: str, allow_stated_p1_mismatch: bool=False) -> bool:
    """Check whether a CIF's structure matches its declared space group.

    `allow_stated_p1_mismatch` permits a CIF declaring P1 when the detected structure has higher
    symmetry.
    """
    structure = cif_parser(cif_str).parse_structures(primitive=False)[0]
    parser = cif_parser(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()
    if not is_match and allow_stated_p1_mismatch:
        stated_normalized = stated_space_group.replace(" ", "").upper()
        if stated_normalized == "P1":
            return True

    return is_match


def _compositions_match(declared: Composition, geometric: Composition) -> bool:
    """Compare a declared formula with an atom-site composition, ignoring cell scale.

    `reduced_composition` only divides out an integer factor, so fractional occupancies come
    back unscaled and fail on scale alone. Mole fractions take over there, at a tolerance
    tight enough to still reject a CIF declaring Fe12C4 whose sites hold Fe2C.
    """
    if all(abs(amount - round(amount)) < 1e-6 for amount in geometric.values()):
        return declared.reduced_composition.almost_equals(
            geometric.reduced_composition, rtol=0.1, atol=0.1
        )
    return declared.fractional_composition.almost_equals(
        geometric.fractional_composition, rtol=0.01, atol=0.01
    )


def is_formula_consistent(cif_str: str) -> bool:
    """Check whether the chemical formula declared by the CIF matches its atom-site composition.

    Compared up to cell scale, so a supercell matches the formula it is a supercell of.
    """
    try:
        parser = cif_parser(cif_str)
        cif_data = parser.as_dict()
        key = list(cif_data.keys())[0]

        formula_data = Composition(extract_data_formula(cif_str))
        formula_sum = Composition(cif_data[key].get("_chemical_formula_sum", ""))
        formula_structural = Composition(cif_data[key].get("_chemical_formula_structural", ""))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                structure = parser.parse_structures(primitive=False)[0]
            except Exception:
                # Some valid disordered CIFs need occupancy rescaling to parse.
                parser = cif_parser(cif_str, occupancy_tolerance=2.0)
                structure = parser.parse_structures(primitive=False)[0]
        formula_geometry = structure.composition

        names_match = (
            formula_data.reduced_formula == formula_sum.reduced_formula ==
            formula_structural.reduced_formula
        )
        return names_match and _compositions_match(formula_sum, formula_geometry)

    except Exception:
        return False


def is_atom_site_multiplicity_consistent(cif_str: str) -> bool:
    # Parse the CIF string
    """Check whether each atom site's stated multiplicity matches its symmetry orbit."""
    parser = cif_parser(cif_str)
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


def is_sensible(cif_str: str, length_lo: float=0.5, length_hi: float=1000., angle_lo: float=10., angle_hi: float=170.) -> bool:
    """Check whether the unit-cell dimensions fall within the supplied physical bounds.

    The check uses only cell parameters parsed from the CIF text. All lengths must lie between
    `length_lo` and `length_hi` in Å, and all angles must lie between `angle_lo` and `angle_hi` in
    degrees. This check is separate from `is_valid`.
    """
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


def is_valid(cif_str: str, bond_length_acceptability_cutoff: float=1.0, allow_stated_p1_mismatch: bool=False, debug: bool=False) -> bool:
    """Check whether a generated CIF passes the structural validity checks.

    The checks are applied in order and stop at the first failure. The formula must match the atom
    sites, atom-site multiplicities must be self-consistent, the bond-length score must reach
    `bond_length_acceptability_cutoff`, and the detected symmetry must match the declared space
    group. `allow_stated_p1_mismatch` permits a declared P1 when the structure has higher symmetry.

    `is_sensible` is not part of this composite check.
    """
    if not is_formula_consistent(cif_str):
        if debug:
            print(f"Formula is inconsistent for {cif_str}")
        return False
    if not is_atom_site_multiplicity_consistent(cif_str):
        if debug:
            print(f"Atom site multiplicity is inconsistent for {cif_str}")
        return False
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score is not None and bond_length_score < bond_length_acceptability_cutoff:
        if debug:
            print(f"Bond length is unreasonable for {cif_str}")
        return False
    if not is_space_group_consistent(cif_str, allow_stated_p1_mismatch=allow_stated_p1_mismatch):
        if debug:
            print(f"Space group is inconsistent for {cif_str}")
        return False
    return True


def _validity_worker(cif_str: str, bond_length_acceptability_cutoff: float, allow_stated_p1_mismatch: bool = False) -> bool:
    """Worker function to check if a single CIF string is valid."""
    _configure_pymatgen_warning_filters()
    if not cif_str or not isinstance(cif_str, str):
        return False
    try:
        # is_valid contains multiple checks (formula, multiplicity, bonds, etc.)
        return is_valid(
            cif_str,
            bond_length_acceptability_cutoff=bond_length_acceptability_cutoff,
            allow_stated_p1_mismatch=allow_stated_p1_mismatch,
            debug=True,
        )
    except Exception:
        return False


def get_density(cif: str) -> float:
    """Compute the crystallographic density of a CIF in g/cm³.

    Returns `NaN` when the CIF cannot be parsed or pymatgen reports incorrect stoichiometry, so
    batch scoring keeps invalid rows instead of raising.
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            structure = cif_parser(cif).parse_structures(primitive=False)[0]
            for warn in w:
                if ("Incorrect stoichiometry" in str(warn.message)):
                    return np.nan
            return structure.density
    except Exception:
        return np.nan
