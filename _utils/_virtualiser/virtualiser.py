#!/usr/bin/env python3
"""Virtualise selected element groups in an ordered crystal structure.

The tool replaces each selected group (two or more elements) on a shared sublattice with fractional
mixed occupancy, refines the resulting structure to higher symmetry, and writes the virtual crystal
as a CIF. Vacancies are not handled explicitly, and all sites containing members of a group are
assumed to belong to the same sublattice.

`symprec` and `angle_tolerance` in the config may each be a single value or a list. With lists, every
combination is tried and the highest-symmetry result is kept, preferring the smallest tolerances on
ties. Generated structures are often slightly distorted, so a single tight tolerance can miss the
parent space group. `--in` may also be a directory of CIFs, in which case `--out` is an output
directory and a per-structure summary CSV is written alongside the CIFs.

Contribution by Dr Ricardo Grau-Crespo:
    https://github.com/rgraucrespo

Usage:
    ```bash
    python _utils/_virtualiser/virtualiser.py --in Mg3ZnO4.cif --config config.yaml --out virtual.cif
    python _utils/_virtualiser/virtualiser.py --in generated_cifs/ --config config.yaml --out virtual_cifs/
    ```
"""
import argparse
import csv
from pathlib import Path

import yaml  # PyYAML
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter


def _as_tolerance_list(value, name: str) -> list[float]:
    """Return a tolerance setting as a non-empty list of positive floats."""
    values = value if isinstance(value, (list, tuple)) else [value]
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number or a list of numbers. Got: {value}") from err
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must be positive and non-empty. Got: {value}")
    return values


def load_config(yaml_path: Path) -> dict:
    """Load the virtualiser YAML configuration.

    The configuration supplies ``symprec``, ``angle_tolerance``, and the
    ``virtual_pairs`` list, whose entries name two or more elements to merge
    onto one shared sublattice. ``symprec`` and ``angle_tolerance`` may each be
    a single value or a list of values to scan. Element groups can also be
    supplied inline on the command line.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    # defaults
    cfg = cfg or {}
    cfg.setdefault("symprec", 0.003)
    cfg.setdefault("angle_tolerance", 0.5)
    cfg.setdefault("virtual_pairs", [])
    _as_tolerance_list(cfg["symprec"], "symprec")
    _as_tolerance_list(cfg["angle_tolerance"], "angle_tolerance")
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


def promote_symmetry_scan(
    struct: Structure,
    symprecs: list[float],
    angle_tols: list[float],
) -> tuple[Structure, dict, list[dict]]:
    """Refine a virtualised structure at every tolerance pair and keep the most symmetric result.

    Tolerance pairs are tried from smallest to largest ``symprec``, then smallest to largest angle
    tolerance, and a later pair only replaces the current best when it gives a strictly higher
    space-group number. The kept result is therefore the highest symmetry found, reached with the
    least tolerance. Pairs for which spglib fails are recorded in the scan and skipped.

    Args:
        struct: Virtualised structure, as returned by ``virtualise_structure``.
        symprecs: Symmetry precisions to try, in Å.
        angle_tols: Angle tolerances to try, in degrees.

    Returns:
        The refined structure, a dict describing the kept result (``symprec``, ``angle_tolerance``,
        ``spacegroup_symbol``, ``spacegroup_number``), and one dict per tolerance pair tried.
    """
    best: tuple[Structure, dict] | None = None
    scan: list[dict] = []
    for sp in sorted(symprecs):
        for at in sorted(angle_tols):
            record = {"symprec": sp, "angle_tolerance": at}
            try:
                refined = promote_symmetry(struct, sp, at)
                sga = SpacegroupAnalyzer(refined, symprec=sp, angle_tolerance=at)
                record.update(spacegroup_symbol=sga.get_space_group_symbol(),
                              spacegroup_number=sga.get_space_group_number())
            except Exception as err:  # spglib can fail at very coarse tolerances
                record["error"] = str(err)
                scan.append(record)
                continue
            scan.append(record)
            if best is None or record["spacegroup_number"] > best[1]["spacegroup_number"]:
                best = (refined, record)
    if best is None:
        raise RuntimeError("Symmetry refinement failed for every tolerance pair tried.")
    return best[0], dict(best[1]), scan


def _virtualise_file(infile: Path, outfile: Path, cfg: dict) -> dict:
    """Virtualise one CIF, write the refined virtual crystal, and return its summary row."""
    struct = Structure.from_file(infile)
    virt = virtualise_structure(struct, cfg["virtual_pairs"])
    refined, info, _ = promote_symmetry_scan(
        virt,
        _as_tolerance_list(cfg["symprec"], "symprec"),
        _as_tolerance_list(cfg["angle_tolerance"], "angle_tolerance"),
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(refined, symprec=info["symprec"]).write_file(outfile)
    a, b, c = refined.lattice.abc
    return {"input": str(infile), "output": str(outfile), **info,
            "a": round(a, 6), "b": round(b, 6), "c": round(c, 6), "natoms": len(refined)}


def main() -> None:
    """Read one CIF or a directory of CIFs, virtualise the requested element groups, and write the results.
    """
    ap = argparse.ArgumentParser(description="Virtualise specified element groups, promote symmetry, and write CIF.")
    ap.add_argument("--in", dest="infile", required=True,
                    help="Input CIF (ordered supercell), or a directory of CIFs.")
    ap.add_argument("--config", dest="config", required=True, help="YAML config with symprec/angle_tolerance/virtual_pairs.")
    ap.add_argument("--out", dest="outfile", required=True,
                    help="Output CIF for the virtual crystal (refined), or an output directory when --in is a directory.")
    ap.add_argument("--summary", dest="summary", default=None,
                    help="CSV summary path for directory input (default: <out>/virtualise_summary.csv).")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    in_path, out_path = Path(args.infile), Path(args.outfile)

    if not in_path.is_dir():
        row = _virtualise_file(in_path, out_path, cfg)
        print(f"Wrote virtual crystal CIF to: {out_path}")
        print(f"Space group: {row['spacegroup_symbol']} (No. {row['spacegroup_number']}) "
              f"at symprec={row['symprec']}, angle_tolerance={row['angle_tolerance']}")
        return

    cifs = sorted(in_path.glob("*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No .cif files found in {in_path}")
    rows = []
    for cif in cifs:
        try:
            rows.append(_virtualise_file(cif, out_path / f"{cif.stem}_virtual.cif", cfg))
        except Exception as err:
            rows.append({"input": str(cif), "error": str(err)})
            print(f"Failed: {cif.name}: {err}")

    summary = Path(args.summary) if args.summary else out_path / "virtualise_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = ["input", "output", "spacegroup_symbol", "spacegroup_number", "symprec",
              "angle_tolerance", "a", "b", "c", "natoms", "error"]
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum("error" not in r for r in rows)
    print(f"Virtualised {n_ok}/{len(cifs)} CIFs into {out_path}; summary: {summary}")


if __name__ == "__main__":
    main()
