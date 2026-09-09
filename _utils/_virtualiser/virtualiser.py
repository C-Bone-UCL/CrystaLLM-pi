#!/usr/bin/env python3
"""Virtualise selected element groups in an ordered crystal structure.

The tool replaces each selected group (two or more elements) on a shared sublattice with fractional
mixed occupancy, refines the resulting structure to higher symmetry, and writes the virtual crystal
as a CIF. Vacancies are not handled explicitly, and all sites containing members of a group are
assumed to belong to the same sublattice.

Contribution by Dr Ricardo Grau-Crespo:
    https://github.com/rgraucrespo

Usage:
    ```bash
    python _utils/_virtualiser/virtualiser.py --in Mg3ZnO4.cif --config config.yaml --out virtual.cif
    ```
"""
import argparse
from pathlib import Path

import yaml  # PyYAML
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter


def load_config(yaml_path: Path) -> dict:
    """Load the virtualiser YAML configuration.

    The configuration supplies ``symprec``, ``angle_tolerance``, and the
    ``virtual_pairs`` list, whose entries name two or more elements to merge
    onto one shared sublattice. Element groups can also be supplied inline on
    the command line.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # defaults
    cfg = cfg or {}
    cfg.setdefault("symprec", 0.003)
    cfg.setdefault("angle_tolerance", 0.5)
    cfg.setdefault("virtual_pairs", [])
    # normalise groups to tuple(sorted(...))
    vpairs = []
    for group in cfg["virtual_pairs"]:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            raise ValueError(f"virtual_pairs entries must list at least 2 elements. Got: {group}")
        vpairs.append(tuple(sorted(str(e) for e in group)))
    cfg["virtual_pairs"] = vpairs
    return cfg


def compute_pair_fractions(struct: Structure, pair: tuple[str, ...]) -> dict[str, float]:
    """Fraction of pure-element sites each member of a group occupies.

    Only sites holding a single pure element count toward the totals. Already-disordered sites are
    ignored, since their occupancy is not a clean vote for any member. Returns fractions keyed by
    element symbol, or zeros when no member is present.
    """
    counts = {el: 0 for el in pair}
    for site in struct.sites:
        # Minimal rule: a site belongs to the group only if it is a *pure* member element
        if len(site.species) == 1:
            el = str(list(site.species.as_dict().keys())[0])
            if el in counts:
                counts[el] += 1
    total = sum(counts.values())
    if total == 0:
        return {el: 0.0 for el in pair}
    return {el: c / total for el, c in counts.items()}


def virtualise_structure(struct: Structure, virtual_pairs: list[tuple[str, ...]]) -> Structure:
    # Build a mapping from elements that are in any group to their group-fractions
    """Merge grouped elements onto shared sites with fractional occupancy.

    Each site containing a member of a selected group is replaced by a mixed site weighted by the
    group fractions. Absent groups are skipped and existing disordered sites are preserved.
    Oxidation states are removed from the result before symmetry processing.
    """
    replace_map: dict[str, dict[str, float]] = {}
    for group in virtual_pairs:
        fracs = compute_pair_fractions(struct, group)
        if all(f == 0.0 for f in fracs.values()):
            continue
        for el in group:
            replace_map[el] = fracs

    new_species = []
    new_coords = []
    for site in struct.sites:
        if len(site.species) == 1:
            el = list(site.species.as_dict().keys())[0]
            el = str(el)
            if el in replace_map:
                fracs = replace_map[el]
                spec_map = {Element(k): float(v) for k, v in fracs.items() if v > 0.0}
                new_species.append(spec_map)
            else:
                new_species.append(site.species)
        else:
            # already disordered, keep as-is
            new_species.append(site.species)
        new_coords.append(site.frac_coords)

    virt = Structure(struct.lattice, new_species, new_coords, coords_are_cartesian=False,
                     site_properties=struct.site_properties if struct.site_properties else None)
    virt.remove_oxidation_states()  # spglib requires species without oxidation states
    return virt


def promote_symmetry(struct: Structure, symprec: float, angle_tol: float) -> Structure:
    """Refine a structure to the highest symmetry consistent with the supplied tolerances.

    If refinement fails, the conventional standard cell is used instead.
    """
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    try:
        refined = sga.get_refined_structure()
    except Exception:
        refined = sga.get_conventional_standard_structure()
    return refined


def main() -> None:
    """Read a CIF, virtualise the requested element groups, and write the result.
    """
    ap = argparse.ArgumentParser(description="Virtualise specified element groups, promote symmetry, and write CIF.")
    ap.add_argument("--in", dest="infile", required=True, help="Input CIF (ordered supercell).")
    ap.add_argument("--config", dest="config", required=True, help="YAML config with symprec/angle_tolerance/virtual_pairs.")
    ap.add_argument("--out", dest="outfile", required=True, help="Output CIF for virtual crystal (refined).")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    struct = Structure.from_file(args.infile)
    virt = virtualise_structure(struct, cfg["virtual_pairs"])
    refined = promote_symmetry(virt, cfg["symprec"], cfg["angle_tolerance"])

    CifWriter(refined, symprec=cfg["symprec"]).write_file(args.outfile)
    print(f"Wrote virtual crystal CIF to: {args.outfile}")
    sga = SpacegroupAnalyzer(refined, symprec=cfg["symprec"], angle_tolerance=cfg["angle_tolerance"])
    print(f"Space group: {sga.get_space_group_symbol()} (No. {sga.get_space_group_number()})")


if __name__ == "__main__":
    main()
