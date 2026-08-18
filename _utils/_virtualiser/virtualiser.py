#!/usr/bin/env python3
"""Minimal crystal virtualiser.

Usage:
  python -m _utils._virtualiser.virtualiser --in Mg3ZnO4.cif --config config.yaml --out virtual.cif

Config (YAML):
  symprec: 0.003
  angle_tolerance: 0.5
  virtual_pairs:
    - [Mg, Zn]

This tool:
  - Reads an ordered supercell CIF.
  - Virtualises specified element pairs by replacing each such site with the
     same fractional composition equal to the global fraction of those elements
     in the structure (computed over sites that belong to the pair).
  - Runs spglib via pymatgen to find a higher-symmetry refined parent.
  - Writes the refined 'virtual crystal' as CIF.

Limitations:
  - No explicit handling of vacancies yet.
  - We assume all sites containing either member of a pair are on the same
    sublattice and are virtualised identically.

Contribution by Dr Ricardo Grau-Crespo:
    https://github.com/rgraucrespo
"""
import argparse
from pathlib import Path

import yaml  # PyYAML
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter


def load_config(yaml_path: Path) -> dict:
    """Read the virtualiser YAML config.

    Supplies `symprec`, `angle_tolerance` and the `virtual_pairs` list. Element pairs may also be given inline on the command line instead of through a file.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # defaults
    cfg = cfg or {}
    cfg.setdefault("symprec", 0.003)
    cfg.setdefault("angle_tolerance", 0.5)
    cfg.setdefault("virtual_pairs", [])
    # normalise pairs to tuple(sorted(...))
    vpairs = []
    for pair in cfg["virtual_pairs"]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"virtual_pairs entries must be 2-element lists. Got: {pair}")
        a, b = pair
        vpairs.append(tuple(sorted((str(a), str(b)))))
    cfg["virtual_pairs"] = vpairs
    return cfg


def compute_pair_fractions(struct: Structure, pair: tuple[str, str]) -> dict[str, float]:
    """Fraction of pure-element sites each member of a pair occupies.

    Only sites holding a single pure element count toward the totals; already-disordered sites are ignored, since their occupancy is not a clean vote for either member. Returns both fractions keyed by element symbol, or zeros when neither element is present.
    """
    a, b = pair
    count_a = 0
    count_b = 0
    for site in struct.sites:
        # Minimal rule: treat a site as belonging to the pair only if it is a *pure* element a or b
        if len(site.species) == 1:
            el = list(site.species.as_dict().keys())[0]
            el = str(el)
            if el == a:
                count_a += 1
            elif el == b:
                count_b += 1
    total = count_a + count_b
    if total == 0:
        return {a: 0.0, b: 0.0}
    fa = count_a / total
    fb = count_b / total
    return {a: fa, b: fb}


def virtualise_structure(struct: Structure, virtual_pairs: list[tuple[str, str]]) -> Structure:
    # Build a mapping from elements that are in any pair to their partner-fractions
    """Merge paired elements onto shared sites with fractional occupancy.

    Turns an ordered supercell into a smaller virtual-crystal cell: every site holding one member of a pair is replaced by a mixed site carrying both, weighted by the fractions from `compute_pair_fractions`. Pairs absent from the structure are skipped, and sites that are already disordered pass through untouched. Oxidation states are stripped from the result, because spglib needs clean species for the symmetry step that usually follows.
    """
    replace_map: dict[str, dict[str, float]] = {}
    for pair in virtual_pairs:
        fracs = compute_pair_fractions(struct, pair)
        if fracs[pair[0]] == 0.0 and fracs[pair[1]] == 0.0:
            continue
        replace_map[pair[0]] = fracs
        replace_map[pair[1]] = fracs

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
            # already disordered; keep as-is
            new_species.append(site.species)
        new_coords.append(site.frac_coords)

    virt = Structure(struct.lattice, new_species, new_coords, coords_are_cartesian=False,
                     site_properties=struct.site_properties if struct.site_properties else None)
    virt.remove_oxidation_states()  # ensure clean species for spglib
    return virt


def promote_symmetry(struct: Structure, symprec: float, angle_tol: float) -> Structure:
    """Refine a structure to the highest symmetry consistent with the given tolerances.

    Virtualising an ordered supercell usually restores symmetry the ordering had broken, and this recovers it. Falls back to the conventional standard cell when refinement fails, which keeps the pipeline producing a usable structure rather than raising.
    """
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    try:
        refined = sga.get_refined_structure()
    except Exception:
        refined = sga.get_conventional_standard_structure()
    return refined


def main() -> None:
    """Read a CIF, virtualise the requested element pairs, and write the result.
    """
    ap = argparse.ArgumentParser(description="Virtualise specified element pairs, promote symmetry, and write CIF.")
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
