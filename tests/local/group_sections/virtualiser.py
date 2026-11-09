"""Local test section: crystal virtualiser."""

import os
import tempfile


class VirtualiserTests:
    """Test the crystal virtualiser functions."""

    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data

    # Helpers

    def _make_ordered_struct(self):
        """Build a minimal ordered Mg3ZnO4-like spinel structure programmatically."""
        from pymatgen.core import Structure, Lattice

        # Simple cubic cell with 2 Mg, 1 Zn, and 3 O on distinct sites
        lattice = Lattice.cubic(4.3)
        species = ["Mg", "Mg", "Zn", "O", "O", "O"]
        coords = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
        ]
        return Structure(lattice, species, coords)

    # Tests

    def test_import(self):
        """Virtualiser module imports cleanly."""
        from _utils._virtualiser import (
            virtualise_structure,
            promote_symmetry,
            compute_pair_fractions,
            load_config,
        )
        assert callable(virtualise_structure)
        assert callable(promote_symmetry)
        assert callable(compute_pair_fractions)
        assert callable(load_config)

    def test_compute_pair_fractions(self):
        """Pair fractions match the pure-element site counts."""
        from _utils._virtualiser import compute_pair_fractions

        struct = self._make_ordered_struct()
        fracs = compute_pair_fractions(struct, ("Mg", "Zn"))

        assert abs(fracs["Mg"] - 2 / 3) < 1e-9
        assert abs(fracs["Zn"] - 1 / 3) < 1e-9
        assert abs(fracs["Mg"] + fracs["Zn"] - 1.0) < 1e-9

    def test_compute_pair_fractions_absent_element(self):
        """Returns zero fractions when neither element is in the structure."""
        from _utils._virtualiser import compute_pair_fractions

        struct = self._make_ordered_struct()
        fracs = compute_pair_fractions(struct, ("Fe", "Cu"))
        assert fracs["Fe"] == 0.0
        assert fracs["Cu"] == 0.0

    def test_virtualise_structure(self):
        """Ordered sites are replaced by mixed-occupancy sites."""
        from _utils._virtualiser import virtualise_structure

        struct = self._make_ordered_struct()
        virt = virtualise_structure(struct, [("Mg", "Zn")])

        # All Mg and Zn sites should now be disordered (len(species) > 1)
        for site in virt.sites:
            els = list(site.species.as_dict().keys())
            if any(e in ("Mg", "Zn") for e in els):
                assert len(els) == 2, "Mg/Zn sites should be mixed-occupancy after virtualisation"

        # O sites should be untouched
        for site in virt.sites:
            els = list(site.species.as_dict().keys())
            if "O" in els:
                assert len(els) == 1, "O sites should remain pure"

    def test_virtualise_structure_preserves_composition(self):
        """Element counts are conserved after virtualisation."""
        from pymatgen.core import Composition
        from _utils._virtualiser import virtualise_structure

        struct = self._make_ordered_struct()
        virt = virtualise_structure(struct, [("Mg", "Zn")])

        orig_comp = struct.composition.fractional_composition
        virt_comp = virt.composition.fractional_composition
        for el in ["Mg", "Zn", "O"]:
            assert abs(orig_comp[el] - virt_comp[el]) < 1e-6, (
                f"Composition mismatch for {el}: {orig_comp[el]} vs {virt_comp[el]}"
            )

    def _make_ordered_bcc_alloy(self):
        """Build an ordered Mo2Ta2W2 stack of three conventional bcc cells."""
        from pymatgen.core import Structure, Lattice

        lattice = Lattice.orthorhombic(3.2, 3.2, 9.6)
        species = ["Mo", "Ta", "W", "Mo", "Ta", "W"]
        coords = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 1 / 6],
            [0.0, 0.0, 1 / 3],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 2 / 3],
            [0.5, 0.5, 5 / 6],
        ]
        return Structure(lattice, species, coords)

    def test_virtualise_structure_ternary(self):
        """A ternary group merges every member site into a 3-component mixed site."""
        from _utils._virtualiser import virtualise_structure, promote_symmetry
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        struct = self._make_ordered_bcc_alloy()
        virt = virtualise_structure(struct, [("Mo", "Ta", "W")])

        for site in virt.sites:
            occ = site.species.as_dict()
            assert len(occ) == 3, "every alloy site should hold all three elements"
            assert all(abs(v - 1 / 3) < 1e-9 for v in occ.values())
        orig_comp = struct.composition.fractional_composition
        virt_comp = virt.composition.fractional_composition
        for el in ("Mo", "Ta", "W"):
            assert abs(orig_comp[el] - virt_comp[el]) < 1e-6

        # The ordered bcc stack must collapse to the disordered bcc parent.
        refined = promote_symmetry(virt, symprec=0.003, angle_tol=0.5)
        assert SpacegroupAnalyzer(refined, symprec=0.003).get_space_group_number() == 229, "expected Im-3m"

    def test_promote_symmetry(self):
        """promote_symmetry rebuilds the virtualised cell in its conventional setting.

        Asserting only that a Structure with sites comes back is satisfied by returning the input
        untouched, which is the one failure worth catching in a function whose whole job is to
        change the cell. The ordered Mg/Zn cell is Cm, virtualising the pair raises it to R3m, and
        symmetrising then expands the primitive 6-site cell into the 18-site conventional one.
        """
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from _utils._virtualiser import virtualise_structure, promote_symmetry

        struct = self._make_ordered_struct()
        virt = virtualise_structure(struct, [("Mg", "Zn")])
        refined = promote_symmetry(virt, symprec=0.01, angle_tol=0.5)

        assert isinstance(refined, Structure)
        assert len(refined.sites) > len(virt.sites), "conventional cell should be larger than the input"
        assert SpacegroupAnalyzer(refined, symprec=0.01).get_space_group_number() == 160, "expected R3m"
        # The virtual Mg/Zn site has to survive symmetrisation, not collapse to one element.
        assert {"Mg", "Zn"} <= {str(sp) for site in refined.sites for sp in site.species}

    def test_load_config(self):
        """load_config reads YAML and applies defaults."""
        from _utils._virtualiser import load_config

        cfg_text = "virtual_pairs:\n  - [Mg, Zn]\n"
        cfg_path = os.path.join(self.temp_dir, "test_cfg.yaml")
        with open(cfg_path, "w") as f:
            f.write(cfg_text)

        cfg = load_config(cfg_path)
        assert cfg["symprec"] == 0.003       # default
        assert cfg["angle_tolerance"] == 0.5  # default
        assert ("Mg", "Zn") in cfg["virtual_pairs"]

    def test_full_pipeline_to_cif(self):
        """End-to-end: ordered structure, virtualised, then CIF written to disk."""
        from pymatgen.io.cif import CifWriter
        from _utils._virtualiser import virtualise_structure, promote_symmetry

        struct = self._make_ordered_struct()
        virt = virtualise_structure(struct, [("Mg", "Zn")])
        refined = promote_symmetry(virt, symprec=0.01, angle_tol=0.5)

        out_cif = os.path.join(self.temp_dir, "virtual_test.cif")
        CifWriter(refined, symprec=0.01).write_file(out_cif)
        assert os.path.exists(out_cif), "Output CIF file was not created"
        assert os.path.getsize(out_cif) > 0, "Output CIF file is empty"

    # Tolerance scan and batch mode

    def _make_distorted_rocksalt(self):
        """Build an L1_2-ordered Mg3ZnO4 rock-salt cell with small fixed displacements.

        The ideal cell is Pm-3m and virtualises to Fm-3m, but displacements of up to 0.006 in fractional coordinates (about 0.03 Angstrom) keep it P1 at the default symprec of 0.003 Angstrom, as in a generated structure.
        """
        import numpy as np
        from pymatgen.core import Lattice, Structure

        species = ["Zn", "Mg", "Mg", "Mg", "O", "O", "O", "O"]
        ideal = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                          [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]])
        shifts = np.array([[0.006, -0.004, 0.003], [-0.005, 0.002, 0.004], [0.003, 0.006, -0.002],
                           [-0.004, -0.003, 0.005], [0.002, -0.006, -0.004], [0.005, 0.003, -0.006],
                           [-0.003, 0.004, 0.002], [0.004, -0.002, -0.005]])
        return Structure(Lattice.cubic(4.23), species, ideal + shifts)

    def _run_cli(self, in_path, cfg_text: str, out_path):
        """Run the virtualiser CLI from the repo root and return the completed process."""
        import subprocess
        import sys
        from pathlib import Path

        cfg_path = Path(self.temp_dir) / f"{Path(in_path).stem}_cfg.yaml"
        cfg_path.write_text(cfg_text)
        return subprocess.run(
            [sys.executable, "-m", "_utils._virtualiser.virtualiser",
             "--in", str(in_path), "--config", str(cfg_path), "--out", str(out_path)],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
        )

    def test_load_config_tolerance_lists(self):
        """load_config accepts tolerance lists and rejects non-positive values."""
        from _utils._virtualiser import load_config

        cfg_path = os.path.join(self.temp_dir, "scan_cfg.yaml")
        with open(cfg_path, "w") as f:
            f.write("symprec: [0.003, 0.1]\nangle_tolerance: [0.5, 5]\nvirtual_pairs:\n  - [Mg, Zn]\n")
        cfg = load_config(cfg_path)
        assert cfg["symprec"] == [0.003, 0.1]
        assert cfg["angle_tolerance"] == [0.5, 5]

        bad_path = os.path.join(self.temp_dir, "bad_cfg.yaml")
        with open(bad_path, "w") as f:
            f.write("symprec: [0.003, -0.1]\nvirtual_pairs:\n  - [Mg, Zn]\n")
        try:
            load_config(bad_path)
        except ValueError:
            return
        raise AssertionError("negative symprec should be rejected")

    def test_promote_symmetry_scan(self):
        """The scan finds the Fm-3m parent that a single tight tolerance misses.

        At symprec 0.003 Angstrom the distorted cell stays below Fm-3m. Scanning up to 0.1 Angstrom must return Fm-3m at the smallest tolerances that reach it (0.1 Angstrom, 0.5 degrees), with one record per pair tried.
        """
        from _utils._virtualiser import promote_symmetry_scan, virtualise_structure

        virt = virtualise_structure(self._make_distorted_rocksalt(), [("Mg", "Zn")])

        _, tight, _ = promote_symmetry_scan(virt, [0.003], [0.5])
        assert tight["spacegroup_number"] < 225, "distorted cell should not be Fm-3m at 0.003 A"

        refined, info, scan = promote_symmetry_scan(virt, [0.1, 0.003, 0.01], [5.0, 0.5])
        assert info["spacegroup_number"] == 225, "expected Fm-3m"
        assert info["symprec"] == 0.1 and info["angle_tolerance"] == 0.5, "ties keep smallest tolerances"
        assert len(scan) == 6, "one record per tolerance pair"
        assert abs(refined.lattice.a - 4.23) < 0.05

        occ = {str(el): amt for site in refined.sites for el, amt in site.species.items() if str(el) in ("Mg", "Zn")}
        assert abs(occ["Mg"] - 0.75) < 1e-6 and abs(occ["Zn"] - 0.25) < 1e-6

    def test_cli_directory_mode(self):
        """The CLI virtualises a directory of CIFs and writes a summary CSV."""
        import csv
        from pathlib import Path

        in_dir = Path(self.temp_dir) / "virt_batch_in"
        out_dir = Path(self.temp_dir) / "virt_batch_out"
        in_dir.mkdir(exist_ok=True)
        self._make_distorted_rocksalt().to(filename=str(in_dir / "distorted.cif"))
        self._make_ordered_struct().to(filename=str(in_dir / "spinel_like.cif"))

        cfg = "symprec: [0.003, 0.01, 0.1]\nangle_tolerance: 0.5\nvirtual_pairs:\n  - [Mg, Zn]\n"
        result = self._run_cli(in_dir, cfg, out_dir)
        assert result.returncode == 0, result.stderr

        with open(out_dir / "virtualise_summary.csv") as f:
            rows = {Path(r["input"]).name: r for r in csv.DictReader(f)}
        assert set(rows) == {"distorted.cif", "spinel_like.cif"}
        assert rows["distorted.cif"]["spacegroup_symbol"] == "Fm-3m"
        assert (out_dir / "distorted_virtual.cif").exists()
        assert (out_dir / "spinel_like_virtual.cif").exists()

    def test_cli_single_file_scalar_config(self):
        """A scalar config on one CIF keeps the original single-file behaviour."""
        from pathlib import Path
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        in_cif = Path(self.temp_dir) / "single_in.cif"
        out_cif = Path(self.temp_dir) / "single_out.cif"
        self._make_ordered_struct().to(filename=str(in_cif))

        result = self._run_cli(in_cif, "symprec: 0.01\nangle_tolerance: 0.5\nvirtual_pairs:\n  - [Mg, Zn]\n", out_cif)
        assert result.returncode == 0, result.stderr
        assert "Space group: R3m (No. 160)" in result.stdout

        written = Structure.from_file(out_cif)
        assert SpacegroupAnalyzer(written, symprec=0.01).get_space_group_number() == 160
