#!/usr/bin/env python3
"""Virtualise selected element groups in an ordered crystal structure.

Each group of two or more elements is merged onto one shared sublattice with fractional occupancy, and the result is refined to higher symmetry and written as a CIF. Vacancies are not handled, and every site holding a member of a group is assumed to belong to the same sublattice.

Contribution by Dr Ricardo Grau-Crespo: https://github.com/rgraucrespo

Usage:
    ```bash
    python _utils/_virtualiser/virtualiser.py --in Mg3ZnO4.cif --config config.yaml --out virtual.cif
    python _utils/_virtualiser/virtualiser.py --in generated_cifs/ --config config.yaml --out virtual_cifs/
    ```
"""
import argparse
import csv
from pathlib import Path

import yaml
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SUMMARY_FIELDS = [
    "input", "output", "spacegroup_symbol", "spacegroup_number", "symprec", "angle_tolerance",
    "a", "b", "c", "natoms", "error",
]


def _as_tolerance_list(value: float | list[float], name: str) -> list[float]:
    """Return a tolerance setting as a non-empty list of positive floats."""
    values = value if isinstance(value, (list, tuple)) else [value]
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a number or a list of numbers. Got: {value}") from err

    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must be positive and non-empty. Got: {value}")
    return values


def load_config(yaml_path: str | Path) -> dict:
    """Load and validate the virtualiser YAML configuration.

    The config holds `symprec` (in Angstrom) and `angle_tolerance` (in degrees), each a single value or a list to scan, and `virtual_pairs`, whose entries name two or more elements to merge onto one sublattice. Groups come back as sorted tuples.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("symprec", 0.003)
    cfg.setdefault("angle_tolerance", 0.5)
    cfg.setdefault("virtual_pairs", [])
    _as_tolerance_list(cfg["symprec"], "symprec")
    _as_tolerance_list(cfg["angle_tolerance"], "angle_tolerance")

    for group in cfg["virtual_pairs"]:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            raise ValueError(f"virtual_pairs entries must list at least 2 elements. Got: {group}")
    cfg["virtual_pairs"] = [tuple(sorted(str(e) for e in group)) for group in cfg["virtual_pairs"]]
    return cfg


def _pure_element(site) -> str | None:
    """Return the element symbol of a fully occupied single-species site, else None."""
    return str(next(iter(site.species))) if len(site.species) == 1 else None


def compute_pair_fractions(struct: Structure, pair: tuple[str, ...]) -> dict[str, float]:
    """Return the fraction of the group's sites that each member occupies.

    Only sites holding one pure element count, since an already disordered site is not a clean vote for any member. Returns zeros when no member is present.
    """
    counts = {el: 0 for el in pair}
    for site in struct.sites:
        el = _pure_element(site)
        if el in counts:
            counts[el] += 1

    total = sum(counts.values())
    if total == 0:
        return {el: 0.0 for el in pair}
    return {el: c / total for el, c in counts.items()}


def virtualise_structure(struct: Structure, virtual_pairs: list[tuple[str, ...]]) -> Structure:
    """Merge grouped elements onto shared sites with fractional occupancy.

    Each pure site of a group member becomes a mixed site weighted by the group fractions. Groups absent from the structure are skipped, already disordered sites are kept, and oxidation states are removed for spglib.
    """
    # Map every member of a present group to that group's fractions
    replace_map: dict[str, dict[str, float]] = {}
    for group in virtual_pairs:
        fracs = compute_pair_fractions(struct, group)
        if any(fracs.values()):
            replace_map.update({el: fracs for el in group})

    new_species = []
    for site in struct.sites:
        el = _pure_element(site)
        if el in replace_map:
            new_species.append({Element(k): v for k, v in replace_map[el].items() if v > 0})
        else:
            new_species.append(site.species)

    virt = Structure(
        struct.lattice,
        new_species,
        [site.frac_coords for site in struct.sites],
        site_properties=struct.site_properties or None,
    )
    virt.remove_oxidation_states()
    return virt


def promote_symmetry(struct: Structure, symprec: float, angle_tol: float) -> Structure:
    """Refine to the highest symmetry within the tolerances, else return the conventional cell."""
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    try:
        return sga.get_refined_structure()
    except Exception:
        return sga.get_conventional_standard_structure()


def promote_symmetry_scan(
    struct: Structure,
    symprecs: list[float],
    angle_tols: list[float],
) -> tuple[Structure, dict, list[dict]]:
    """Refine a virtualised structure at every tolerance pair and keep the most symmetric result.

    Generated cells are often slightly distorted, so one tight tolerance can miss the parent space group. Pairs run from the smallest `symprec` up, then the smallest angle up, and a later pair replaces the best only with a strictly higher space-group number, so the kept result is the highest symmetry reached at the least tolerance. Pairs where spglib fails are recorded and skipped.

    Args:
        struct: virtualised structure from `virtualise_structure`
        symprecs: symmetry precisions to try, in Angstrom
        angle_tols: angle tolerances to try, in degrees

    Returns:
        Refined structure, kept record (`symprec`, `angle_tolerance`, `spacegroup_symbol`, `spacegroup_number`), and one record per pair tried
    """
    best: tuple[Structure, dict] | None = None
    scan: list[dict] = []
    for sp in sorted(symprecs):
        for at in sorted(angle_tols):
            record = {"symprec": sp, "angle_tolerance": at}
            scan.append(record)
            try:
                refined = promote_symmetry(struct, sp, at)
                sga = SpacegroupAnalyzer(refined, symprec=sp, angle_tolerance=at)
                record.update(
                    spacegroup_symbol=sga.get_space_group_symbol(),
                    spacegroup_number=sga.get_space_group_number(),
                )
            except Exception as err:  # spglib can fail at very coarse tolerances
                record["error"] = str(err)
                continue

            if best is None or record["spacegroup_number"] > best[1]["spacegroup_number"]:
                best = (refined, record)

    if best is None:
        raise RuntimeError("Symmetry refinement failed for every tolerance pair tried.")
    return best[0], dict(best[1]), scan


def _virtualise_file(infile: Path, outfile: Path, cfg: dict) -> dict:
    """Virtualise one CIF, write the refined virtual crystal, and return its summary row."""
    virt = virtualise_structure(Structure.from_file(infile), cfg["virtual_pairs"])
    refined, info, _ = promote_symmetry_scan(
        virt,
        _as_tolerance_list(cfg["symprec"], "symprec"),
        _as_tolerance_list(cfg["angle_tolerance"], "angle_tolerance"),
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(refined, symprec=info["symprec"]).write_file(outfile)
    a, b, c = refined.lattice.abc
    return {
        "input": str(infile),
        "output": str(outfile),
        **info,
        "a": round(a, 6),
        "b": round(b, 6),
        "c": round(c, 6),
        "natoms": len(refined),
    }


def main() -> None:
    """Parse arguments and virtualise one CIF or a directory of CIFs."""
    ap = argparse.ArgumentParser(description="Virtualise element groups, promote symmetry, and write CIFs.")
    ap.add_argument("--in", dest="infile", required=True, help="Input ordered CIF, or a directory of CIFs")
    ap.add_argument("--config", required=True, help="YAML config with symprec, angle_tolerance and virtual_pairs")
    ap.add_argument("--out", dest="outfile", required=True, help="Output CIF, or a directory when --in is one")
    ap.add_argument("--summary", default=None, help="Summary CSV for directory input (default: <out>/virtualise_summary.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    in_path, out_path = Path(args.infile), Path(args.outfile)

    if not in_path.is_dir():
        row = _virtualise_file(in_path, out_path, cfg)
        print(f"Wrote virtual crystal CIF to: {out_path}")
        print(
            f"Space group: {row['spacegroup_symbol']} (No. {row['spacegroup_number']}) "
            f"at symprec={row['symprec']}, angle_tolerance={row['angle_tolerance']}"
        )
        return

    cifs = sorted(in_path.glob("*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No .cif files found in {in_path}")

    # One failed CIF is recorded in the summary rather than stopping the batch
    rows = []
    for cif in cifs:
        try:
            rows.append(_virtualise_file(cif, out_path / f"{cif.stem}_virtual.cif", cfg))
        except Exception as err:
            rows.append({"input": str(cif), "error": str(err)})
            print(f"Failed: {cif.name}: {err}")

    summary = Path(args.summary) if args.summary else out_path / "virtualise_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum("error" not in r for r in rows)
    print(f"Virtualised {n_ok}/{len(cifs)} CIFs into {out_path}, summary in {summary}")


if __name__ == "__main__":
    main()
