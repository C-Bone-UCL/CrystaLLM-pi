"""Local test section: load and generate."""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixtures_dir = os.path.join(script_dir, "tests", "fixtures")

class LoadAndGenerateTests:
    """Test the new load_and_generate.py script."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_cli_help_runs(self) -> None:
        """The entrypoint must at least build its parser and exit cleanly.

        Nothing else drives _load_and_generate.py's main(), so a broken argparse or a
        NameError at import level would otherwise only surface for a user.
        """
        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, "_load_and_generate.py"), "--help"],
            capture_output=True, text=True, cwd=script_dir, timeout=120,
        )

        assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr[-400:]}"
        assert "--hf_model_path" in result.stdout, "help text should list the model flag"

    def test_cli_rejects_unknown_model(self) -> None:
        """An unregistered model must fail before any generation work starts."""
        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, "_load_and_generate.py"),
             "--hf_model_path", "nobody/not-a-real-model",
             "--reduced_formula_list", "TiO2",
             "--output_parquet", os.path.join(self.temp_dir, "unused.parquet")],
            capture_output=True, text=True, cwd=script_dir, timeout=120,
        )

        assert result.returncode != 0, "unknown model should not exit 0"
        assert "registry" in result.stderr.lower(), f"unhelpful error: {result.stderr[-400:]}"

    def test_hf_model_loading(self) -> None:
        """Built-in models remain available without an external registry."""
        import _load_and_generate

        model_info = _load_and_generate.MODEL_INFO["c-bone/CrystaLLM-pi_base"]
        assert model_info["model_type"] == "Base"

    def test_model_registry_json_schema(self) -> None:
        """Packaged default registry parses and every entry is complete."""
        from _utils._direct_gen_utils import MODEL_INFO

        required = {"description", "conditions", "example_conditions",
                    "max", "min", "normalization", "model_type", "condition_format"}
        optional = {"config_overrides"}
        assert MODEL_INFO, "default registry is empty"
        for path, info in MODEL_INFO.items():
            missing = required - set(info)
            assert not missing, f"{path} missing keys: {missing}"
            unknown = set(info) - required - optional
            assert not unknown, f"{path} unknown keys: {unknown}"
            assert info["model_type"] in {"Base", "PKV", "Slider", "Prefix", "PrefixXRD", "Residual"}, \
                f"{path}: unknown model_type {info['model_type']!r}"
            assert info["condition_format"] in {None, "scalar", "xrd_top20", "xrd_continuous"}, \
                f"{path}: unknown condition_format {info['condition_format']!r}"
            if "config_overrides" in info:
                assert isinstance(info["config_overrides"], dict), \
                    f"{path}: config_overrides must be a dict"

    def test_custom_model_registry(self) -> None:
        """A custom registry resolves a Hub path and normalizes raw conditions."""
        import _load_and_generate

        model_path = "student/MP-20-Density"
        registry_path = os.path.join(self.temp_dir, "custom_models.json")
        output_path = os.path.join(self.temp_dir, "custom-model-output.parquet")
        registry = {
            model_path: {
                "description": "Tutorial density model",
                "conditions": 1,
                "example_conditions": ["5.0"],
                "max": 10.0,
                "min": 0.0,
                "normalization": "linear",
                "model_type": "PKV",
            }
        }
        with open(registry_path, "w", encoding="utf-8") as file:
            json.dump(registry, file)

        original_argv = sys.argv[:]
        original_generate = _load_and_generate.generate_cifs_with_hf_model
        original_registry = dict(_load_and_generate.MODEL_INFO)
        observed = {}

        def _fake_generate(
            df_prompts: pd.DataFrame,
            hf_model_path: str,
            args: argparse.Namespace,
            worker_count: int = 1,
        ) -> pd.DataFrame:
            observed["model_path"] = hf_model_path
            observed["condition_vector"] = df_prompts.iloc[0]["condition_vector"]
            return pd.DataFrame([
                {"Material ID": "SiO2_Z1_1", "Generated CIF": "data_test"}
            ])

        try:
            _load_and_generate.generate_cifs_with_hf_model = _fake_generate
            sys.argv = [
                "_load_and_generate.py",
                "--hf_model_path", model_path,
                "--model_registry", registry_path,
                "--condition_lists", "5.0",
                "--reduced_formula_list", "SiO2",
                "--z_list", "1",
                "--output_parquet", output_path,
                "--skip_postprocess",
            ]
            _load_and_generate.main()

            assert observed["model_path"] == model_path
            assert float(observed["condition_vector"]) == 0.5
            assert os.path.exists(output_path)
        finally:
            sys.argv = original_argv
            _load_and_generate.generate_cifs_with_hf_model = original_generate
            _load_and_generate.MODEL_INFO.clear()
            _load_and_generate.MODEL_INFO.update(original_registry)

    def test_invalid_custom_model_registry(self) -> None:
        """The registry root must map model paths to metadata."""
        from _utils._direct_gen_utils import _load_custom_model_registry

        registry_path = os.path.join(self.temp_dir, "invalid-models.json")
        with open(registry_path, "w", encoding="utf-8") as file:
            json.dump(["student/broken"], file)

        try:
            _load_custom_model_registry(registry_path)
        except ValueError as exc:
            assert "JSON object" in str(exc)
        else:
            raise AssertionError("A list registry should raise ValueError")

    def test_prompt_generation_from_args(self):
        """Test underlying manual prompt generation tool."""
        from _utils._generating.make_prompts import create_manual_prompts
        
        df_prompts = create_manual_prompts(
            compositions=["Si1O2"],
            condition_lists=[["0.5"]],
            level="level_2",
            spacegroups=None,
        )
        
        assert len(df_prompts) > 0, "Should generate prompts"
        assert 'Prompt' in df_prompts.columns, "Should have Prompt column"

    def test_xrd_raw_file_parsing_and_conversion(self):
        """Verify raw XRD files are dynamically processed and scaled."""
        from _utils._direct_gen_utils import parse_xrd_file_to_condition_vector
        raw_xy = os.path.join(fixtures_dir, "test_rutile_raw.xy")
        
        if not os.path.exists(raw_xy):
            raise FileNotFoundError(f"Required raw XRD fixture not found: {raw_xy}")
            
        # Parse using MoKa wavelength to test dynamic scaling
        vector = parse_xrd_file_to_condition_vector(raw_xy, wavelength=0.71073)
        
        assert len(vector) == 40, "Condition vector should have exactly 40 elements"
        
        thetas = vector[:20]
        intensities = vector[20:]
        
        assert all((0 <= t <= 1.0) or (t == -100) for t in thetas), "Thetas out of bounds"
        assert all((0 <= i <= 1.0) or (i == -100) for i in intensities), "Intensities out of bounds"
        
        # The highest intensity should be exactly 1.0 after internal normalization
        valid_intensities = [i for i in intensities if i != -100]
        assert max(valid_intensities) == 1.0, "Max intensity should be normalized to 1.0"

    def test_mattergen_xrd_generation_smoke(self):
        """Try a minimal Mattergen-XRD generation run with explicit Z."""
        import _load_and_generate
        from _utils import _direct_gen_utils

        xrd_file = os.path.join(fixtures_dir, "test_rutile_processed.csv")
        if not os.path.exists(xrd_file):
            raise FileNotFoundError(f"Required XRD csv not found for smoke test: {xrd_file}")

        args = argparse.Namespace(
            hf_model_path="c-bone/CrystaLLM-pi_Mattergen-XRD",
            input_parquet=None,
            reduced_formula_list="TiO2",
            z_list="2",
            search_zs=False,
            xrd_files=[xrd_file],
            xrd_wavelength=1.54056,
            condition_lists=None,
            level="level_4",
            spacegroups="P4_2/mnm",
            do_sample="False",
            top_k=15,
            top_p=0.95,
            temperature=1.0,
            gen_max_length=256,
            num_return_sequences=1,
            max_return_attempts=1,
            max_samples=1,
            scoring_mode="None",
            target_valid_cifs=0,
            num_workers=1,
            skip_postprocess=True,
            verbose=False,
            output_parquet=os.path.join(self.temp_dir, "mattergen_smoke.parquet"),
            output_cif_dir=None,
        )

        canonical = _direct_gen_utils.canonicalize_reduced_formulas(["TiO2"])
        specs = _direct_gen_utils.build_reduced_formula_specs(
            canonical, [2], [{"xrd": xrd_file, "sg": "P4_2/mnm", "cond": None}], xrd_format="xrd_top20", xrd_wavelength=1.54056
        )
        
        df_prompts = _load_and_generate.generate_prompts_from_specs(specs, args)
        assert len(df_prompts) == 1, "Smoke test should create exactly one prompt"

        df_generated = _load_and_generate.generate_cifs_with_hf_model(
            df_prompts=df_prompts,
            hf_model_path=args.hf_model_path,
            args=args,
        )
        assert len(df_generated) >= 1, (
            "Smoke generation returned no rows. "
            "This test now runs with target_valid_cifs=0, so empty output usually means generation failed rather than filtering."
        )
        assert "Generated CIF" in df_generated.columns, (
            f"Expected 'Generated CIF' column, got columns={list(df_generated.columns)}"
        )

    def test_mattergen_xrd_allows_missing_xrd_inputs(self):
        """Mattergen-XRD should run without xrd_files by passing missing conditioning."""
        import _load_and_generate
        from _utils import _direct_gen_utils

        canonical = _direct_gen_utils.canonicalize_reduced_formulas(["TiO2"])
        specs = _direct_gen_utils.build_reduced_formula_specs(
            canonical,
            [2],
            [{"xrd": None, "sg": None, "cond": None}],
            xrd_format="xrd_top20",
            xrd_wavelength=1.54056,
        )
        assert specs[0]["condition_vector"] is None

        output_parquet = os.path.join(self.temp_dir, "mattergen_no_xrd.parquet")
        original_argv = sys.argv[:]
        original_generate = _load_and_generate.generate_cifs_with_hf_model

        def _fake_generate(df_prompts, hf_model_path, args, worker_count=1):
            rows = []
            for _, row in df_prompts.iterrows():
                rows.append({
                    "Material ID": f"{row['Material ID']}_1",
                    "Generated CIF": "data_test\n_atom_site_type_symbol",
                })
            return pd.DataFrame(rows)

        try:
            _load_and_generate.generate_cifs_with_hf_model = _fake_generate
            sys.argv = [
                "_load_and_generate.py",
                "--hf_model_path", "c-bone/CrystaLLM-pi_Mattergen-XRD",
                "--reduced_formula_list", "TiO2",
                "--z_list", "2",
                "--output_parquet", output_parquet,
                "--skip_postprocess",
            ]
            _load_and_generate.main()

            assert os.path.exists(output_parquet), "Expected output parquet without XRD inputs"
            df = pd.read_parquet(output_parquet)
            assert len(df) == 1
            assert "Generated CIF" in df.columns
        finally:
            sys.argv = original_argv
            _load_and_generate.generate_cifs_with_hf_model = original_generate

    def test_direct_generation_logp_smoke(self):
        """Run the README LOGP ranked Z-search flow and verify it emits a ranked CIF parquet."""
        output_parquet = os.path.join(self.temp_dir, "readme_logp_base_sio2.parquet")
        cmd = [
            sys.executable,
            os.path.join(script_dir, "_load_and_generate.py"),
            "--hf_model_path", "c-bone/CrystaLLM-pi_base",
            "--reduced_formula_list", "SiO2",
            "--search_zs",
            "--scoring_mode", "LOGP",
            "--target_valid_cifs", "3",
            "--num_return_sequences", "10",
            "--output_parquet", output_parquet,
            "--skip_postprocess",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"README LOGP smoke generation failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        assert os.path.exists(output_parquet), "Expected README LOGP smoke test to write a parquet output"

        df_generated = pd.read_parquet(output_parquet)
        assert len(df_generated) >= 1, "README LOGP smoke generation should return at least one row"
        assert "Generated CIF" in df_generated.columns, "Expected Generated CIF output column"
        assert "score" in df_generated.columns, "Expected LOGP score column in output"
        assert "reduced_formula_target" in df_generated.columns, "Expected reduced formula metadata in output"

        generated_cifs = df_generated["Generated CIF"].dropna().astype(str)
        assert not generated_cifs.empty, "Expected at least one non-empty CIF"
        assert generated_cifs.str.contains("data_").all(), "Generated outputs should look like CIF text"
        assert generated_cifs.str.contains("_atom_site_type_symbol").all(), "Generated CIF should include atomic site data"

        finite_scores = pd.to_numeric(df_generated["score"], errors="coerce")
        finite_scores = finite_scores[finite_scores.notna()]
        assert not finite_scores.empty, "Expected finite LOGP scores"
        assert finite_scores.is_monotonic_increasing, "LOGP-ranked outputs should be sorted by score"
        assert set(df_generated["reduced_formula_target"].dropna()) == {"SiO2"}, "Expected SiO2-only output for this README example"

    def test_multi_gpu_single_prompt_worker_resolution(self):
        """num_workers_gpu is the single knob: unset fans out, N caps, 1 forces single-GPU."""
        from _utils import _direct_gen_utils

        original_get_visible_gpu_count = _direct_gen_utils.get_visible_gpu_count
        try:
            _direct_gen_utils.get_visible_gpu_count = lambda: 4

            workers = _direct_gen_utils.resolve_multi_gpu_workers(
                argparse.Namespace(num_workers_gpu=None), n_prompts=1)
            assert workers == 4, f"Expected 4 workers for single prompt fanout, got {workers}"

            workers = _direct_gen_utils.resolve_multi_gpu_workers(
                argparse.Namespace(num_workers_gpu=2), n_prompts=1)
            assert workers == 2, f"Expected the cap of 2 workers, got {workers}"

            workers = _direct_gen_utils.resolve_multi_gpu_workers(
                argparse.Namespace(num_workers_gpu=1), n_prompts=1)
            assert workers == 0, f"Expected 0 (single-process path) for a cap of 1, got {workers}"
        finally:
            _direct_gen_utils.get_visible_gpu_count = original_get_visible_gpu_count

    def test_scoring_mode_normalization_helper(self):
        """Shared scoring mode normalization should handle common casing."""
        from _utils._generating.generate_CIFs import _normalize_scoring_mode

        assert _normalize_scoring_mode("None") == "none"
        assert _normalize_scoring_mode("none") == "none"
        assert _normalize_scoring_mode("LOGP") == "logp"

    def test_reduced_formula_prompt_expansion(self):
        """Reduced formula mode should expand each formula to Z=1..4 prompts during a search."""
        import _load_and_generate
        from _utils import _direct_gen_utils

        args = argparse.Namespace(level="level_2", verbose=False)
        formulas = ["SiO2", "TiO2"]
        canonical = _direct_gen_utils.canonicalize_reduced_formulas(formulas)
        # Expand each formula × 4 Z values for the index-based API
        expanded_formulas = [f for f in canonical for _ in [1, 2, 3, 4]]
        expanded_z_values = [z for _ in canonical for z in [1, 2, 3, 4]]
        expanded_properties = [{"xrd": None, "sg": None, "cond": None} for _ in expanded_formulas]

        specs = _direct_gen_utils.build_reduced_formula_specs(expanded_formulas, expanded_z_values, expanded_properties, xrd_format=None)
        df_prompts = _load_and_generate.generate_prompts_from_specs(specs, args)
        
        assert len(df_prompts) == 8, "Expected 2 formulas x 4 Z values"
        assert "reduced_formula_target" in df_prompts.columns
        assert "Z_search" in df_prompts.columns
        assert set(df_prompts["Z_search"].tolist()) == {1, 2, 3, 4}
        assert df_prompts["Material ID"].str.contains(r"_Z[1-4]$").all(), "Material IDs should include Z suffix for composition-based prompts"

    def test_reduced_formula_selection_modes(self):
        """Selection should keep one row per reduced formula for LOGP and None modes."""
        from _utils import _direct_gen_utils

        df_prompts = pd.DataFrame([
            {"Material ID": "SiO2_Z1", "reduced_formula_target": "SiO2", "Z_search": 1, "prompt_order": 1},
            {"Material ID": "SiO2_Z2", "reduced_formula_target": "SiO2", "Z_search": 2, "prompt_order": 2},
            {"Material ID": "TiO2_Z1", "reduced_formula_target": "TiO2", "Z_search": 1, "prompt_order": 3},
            {"Material ID": "TiO2_Z2", "reduced_formula_target": "TiO2", "Z_search": 2, "prompt_order": 4},
        ])

        df_generated = pd.DataFrame([
            {"Material ID": "SiO2_Z1_1", "Generated CIF": "cif_bad", "score": 0.5, "is_valid": False},
            {"Material ID": "SiO2_Z2_1", "Generated CIF": "cif_good_a", "score": 0.8, "is_valid": True},
            {"Material ID": "SiO2_Z2_2", "Generated CIF": "cif_good_b", "score": 0.2, "is_valid": True},
            {"Material ID": "TiO2_Z1_1", "Generated CIF": "cif_good_c", "score": 0.4, "is_valid": True},
            {"Material ID": "TiO2_Z2_1", "Generated CIF": "cif_bad_2", "score": 0.1, "is_valid": False},
        ])

        out_logp = _direct_gen_utils.reduce_rows_for_reduced_formula_search(
            df_generated=df_generated,
            df_prompts=df_prompts,
            formulas_in_order=["SiO2", "TiO2"],
            scoring_mode="logp",
        )
        assert len(out_logp) == 2
        si_row = out_logp[out_logp["reduced_formula_target"] == "SiO2"].iloc[0]
        assert si_row["Generated CIF"] == "cif_good_b"

    def test_reduced_formula_selection_xrd_fit_direction(self):
        """pearson keeps the highest score per formula, unlike lower-better logp."""
        from _utils import _direct_gen_utils

        df_prompts = pd.DataFrame([
            {"Material ID": "TiO2_Z1", "reduced_formula_target": "TiO2", "Z_search": 1, "prompt_order": 1},
            {"Material ID": "TiO2_Z2", "reduced_formula_target": "TiO2", "Z_search": 2, "prompt_order": 2},
        ])
        df_generated = pd.DataFrame([
            {"Material ID": "TiO2_Z1_1", "Generated CIF": "cif_low", "score": 0.1, "is_valid": True},
            {"Material ID": "TiO2_Z2_1", "Generated CIF": "cif_high", "score": 0.9, "is_valid": True},
        ])

        out = _direct_gen_utils.reduce_rows_for_reduced_formula_search(
            df_generated=df_generated,
            df_prompts=df_prompts,
            formulas_in_order=["TiO2"],
            scoring_mode="pearson",
        )
        assert len(out) == 1
        assert out.iloc[0]["Generated CIF"] == "cif_high", "pearson must keep the highest score"

    def test_xrd_fit_scores_discriminate(self):
        """XRD fit scoring must prefer the phase that produced the scan.

        The candidate CIF is a raw model generation: asymmetric unit plus a placeholder
        operator list. Skipping the symmetry expansion drops its pearson r below 0.3,
        so the threshold also protects that step.
        """
        import numpy as np
        from pymatgen.core import Lattice, Structure
        from _utils._generating.xrd_fit import pearson_score, simulate_profile
        from _utils._preprocessing._process_exp_XRD_continuous import process_exp_file_to_continuous

        scan = os.path.join(fixtures_dir, "Rutile-TiO2-unproc.txt")
        raw_cif = os.path.join(fixtures_dir, "raw_gen_rutile.cif")

        profile = np.asarray(process_exp_file_to_continuous(scan, 1.54059, True))
        input_iq = profile[:, 1]

        with open(raw_cif, encoding="utf-8") as fh:
            rutile_sim = simulate_profile(fh.read())

        # Wrong-phase contrast: fluorite-structured TiO2, same formula, different pattern.
        fluorite = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(4.8), ["Ti", "O"], [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        wrong_sim = simulate_profile(fluorite.to(fmt="cif"))

        rutile_r, wrong_r = pearson_score(input_iq, rutile_sim), pearson_score(input_iq, wrong_sim)

        assert rutile_r > 0.3, f"raw generated rutile should fit its own scan, got r={rutile_r:.3f}"
        assert rutile_r > wrong_r, f"pearson ranked the wrong phase over rutile ({wrong_r:.3f} vs {rutile_r:.3f})"

    def test_reduced_formula_selection_uses_provided_cif_text(self):
        """Selection should validate the CIF text as provided when consistency flags are absent."""
        from _utils import _direct_gen_utils

        df_prompts = pd.DataFrame([
            {"Material ID": "TiO2_Z1", "reduced_formula_target": "TiO2", "Z_search": 1, "prompt_order": 1},
        ])
        df_generated = pd.DataFrame([
            {"Material ID": "TiO2_Z1_1", "Generated CIF": "raw_cif", "score": 0.1},
        ])

        original_is_valid = _direct_gen_utils.is_valid
        try:
            _direct_gen_utils.is_valid = lambda cif, **kwargs: cif == "raw_cif"

            out = _direct_gen_utils.reduce_rows_for_reduced_formula_search(
                df_generated=df_generated,
                df_prompts=df_prompts,
                formulas_in_order=["TiO2"],
                scoring_mode="logp",
            )

            assert len(out) == 1
            row = out.iloc[0]
            assert row["Generated CIF"] == "raw_cif"
        finally:
            _direct_gen_utils.is_valid = original_is_valid

    def test_level1_dummy_formula_canonicalization(self):
        """Level-1 dummy formula token X should bypass pymatgen canonicalization."""
        from _utils._direct_gen_utils import canonicalize_reduced_formulas

        out = canonicalize_reduced_formulas(["X"])
        assert out == ["X"]

    def test_logp_zero_target_is_invalid_configuration(self):
        """Mirror CLI behavior: LOGP mode cannot be paired with target_valid_cifs=0."""
        cmd = [
            sys.executable,
            os.path.join(script_dir, "_load_and_generate.py"),
            "--hf_model_path", "c-bone/CrystaLLM-pi_SLME",
            "--condition_lists", "25.0",
            "--level", "level_1",
            "--scoring_mode", "LOGP",
            "--target_valid_cifs", "0",
            "--output_parquet", os.path.join(self.temp_dir, "invalid_logp.parquet"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "scoring_mode=LOGP requires --target_valid_cifs > 0." in proc.stderr

    def test_search_zs_zero_target_keeps_all_generated_rows(self):
        """search_zs should not reduce rows when target_valid_cifs is set to 0."""
        import _load_and_generate

        output_parquet = os.path.join(self.temp_dir, "search_zs_all_rows.parquet")
        original_argv = sys.argv[:]
        original_build_specs = _load_and_generate.build_reduced_formula_specs
        original_generate_prompts = _load_and_generate.generate_prompts_from_specs
        original_generate = _load_and_generate.generate_cifs_with_hf_model
        original_reduce = _load_and_generate.reduce_rows_for_reduced_formula_search

        def _fake_specs(*args, **kwargs):
            return [{"Material ID": f"SiO2_Z{i}", "reduced_formula_target": "SiO2", "Z_search": i} for i in [1, 2, 3, 4]]

        def _fake_prompts(specs, args):
            return pd.DataFrame([
                {"Material ID": "SiO2_Z1", "reduced_formula_target": "SiO2", "Z_search": 1, "prompt_order": 1},
                {"Material ID": "SiO2_Z2", "reduced_formula_target": "SiO2", "Z_search": 2, "prompt_order": 2},
                {"Material ID": "SiO2_Z3", "reduced_formula_target": "SiO2", "Z_search": 3, "prompt_order": 3},
                {"Material ID": "SiO2_Z4", "reduced_formula_target": "SiO2", "Z_search": 4, "prompt_order": 4},
            ])

        def _fake_generate(df_prompts, hf_model_path, args, worker_count=1):
            rows = []
            for _, row in df_prompts.iterrows():
                for seq in range(5):
                    rows.append({
                        "Material ID": f"{row['Material ID']}_{seq}",
                        "Generated CIF": f"cif_{row['Z_search']}_{seq}",
                    })
            return pd.DataFrame(rows)

        def _fake_reduce(*args, **kwargs):
            raise AssertionError("Reducer should not be called when target_valid_cifs=0")

        try:
            _load_and_generate.build_reduced_formula_specs = _fake_specs
            _load_and_generate.generate_prompts_from_specs = _fake_prompts
            _load_and_generate.generate_cifs_with_hf_model = _fake_generate
            _load_and_generate.reduce_rows_for_reduced_formula_search = _fake_reduce

            sys.argv = [
                "_load_and_generate.py",
                "--hf_model_path", "c-bone/CrystaLLM-pi_base",
                "--output_parquet", output_parquet,
                "--reduced_formula_list", "SiO2",
                "--search_zs",
                "--target_valid_cifs", "0",
                "--num_return_sequences", "5",
                "--max_return_attempts", "2",
                "--skip_postprocess",
            ]
            _load_and_generate.main()

            df = pd.read_parquet(output_parquet)
            assert len(df) == 20, f"Expected 20 rows (4 Z x 5 sequences), got {len(df)}"
        finally:
            sys.argv = original_argv
            _load_and_generate.build_reduced_formula_specs = original_build_specs
            _load_and_generate.generate_prompts_from_specs = original_generate_prompts
            _load_and_generate.generate_cifs_with_hf_model = original_generate
            _load_and_generate.reduce_rows_for_reduced_formula_search = original_reduce

    def test_search_zs_early_stop_selects_first_valid(self):
        """Early-stop Z search drops a formula once found and ships one clean row for it."""
        import _load_and_generate

        output_parquet = os.path.join(self.temp_dir, "early_stop_first_valid.parquet")
        original_argv = sys.argv[:]
        original_generate = _load_and_generate.generate_cifs_with_hf_model
        seen_zs = []

        def _fake_generate(df_prompts, hf_model_path, args, worker_count=1):
            seen_zs.extend(df_prompts["Z_search"].tolist())
            if df_prompts.iloc[0]["Z_search"] != 2:  # only Z=2 "succeeds"
                return pd.DataFrame()
            return pd.DataFrame([{
                "Material ID": f"{df_prompts.iloc[0]['Material ID']}_1",
                "Generated CIF": "data_test",
                "is_consistent": True,
            }])

        try:
            _load_and_generate.generate_cifs_with_hf_model = _fake_generate
            sys.argv = [
                "_load_and_generate.py",
                "--hf_model_path", "c-bone/CrystaLLM-pi_base",
                "--reduced_formula_list", "SiO2",
                "--search_zs",
                "--output_parquet", output_parquet,
                "--skip_postprocess",
            ]
            _load_and_generate.main()

            assert seen_zs == [1, 2], f"Expected early stop after Z=2, searched {seen_zs}"
            df = pd.read_parquet(output_parquet)
            assert len(df) == 1, f"Expected 1 selected row, got {len(df)}"
            assert df.iloc[0]["Material ID"] == "SiO2_Z2_1"
            assert df.iloc[0]["reduced_formula_target"] == "SiO2"
            for col in ("Z_search", "prompt_order", "is_consistent", "is_valid"):
                assert col not in df.columns, f"Bookkeeping column {col} leaked into output"
        finally:
            sys.argv = original_argv
            _load_and_generate.generate_cifs_with_hf_model = original_generate

    def test_condition_format_routing(self) -> None:
        """Registry-driven condition_format resolution with the Slider legacy fallback."""
        from _utils import _direct_gen_utils
        from _utils._direct_gen_utils import XRD_FORMATS, get_condition_format

        # Built-in entries carry the key explicitly
        assert get_condition_format("c-bone/CrystaLLM-pi_Chili100K-XRD") == "xrd_top20"
        assert get_condition_format("c-bone/CrystaLLM-pi_density") == "scalar"
        assert get_condition_format("c-bone/CrystaLLM-pi_base") is None
        assert get_condition_format("not/registered") is None

        # scalar is NOT an XRD format: PKV models must keep their --condition_lists path
        assert "scalar" not in XRD_FORMATS

        # Legacy fallback: a user-overlay Slider entry without the key is top-20 XRD
        _direct_gen_utils.MODEL_INFO["overlay/legacy-slider"] = {"model_type": "Slider"}
        try:
            assert get_condition_format("overlay/legacy-slider") == "xrd_top20"
        finally:
            del _direct_gen_utils.MODEL_INFO["overlay/legacy-slider"]

    def test_continuous_xrd_spec_building(self) -> None:
        """build_reduced_formula_specs converts a raw scan to a nested (1000, 2) profile."""
        import numpy as np
        from _utils import _direct_gen_utils

        # Synthetic gaussian scan (the tiny fixtures fail MIN_POINTS_ON_GRID by design)
        two_theta = np.linspace(10.0, 80.0, 3000)
        intensity = 50.0 + 1000.0 * np.exp(-0.5 * ((two_theta - 27.4) / 0.15) ** 2)
        scan_path = os.path.join(self.temp_dir, "synthetic_gaussian.xy")
        with open(scan_path, "w", encoding="utf-8") as fh:
            fh.write("Wavelength = 1.54059\n")
            fh.writelines(f"{t:.6f} {i:.6f}\n" for t, i in zip(two_theta, intensity))

        canonical = _direct_gen_utils.canonicalize_reduced_formulas(["TiO2"])
        specs = _direct_gen_utils.build_reduced_formula_specs(
            canonical, [2], [{"xrd": scan_path, "sg": None, "cond": None}],
            xrd_format="xrd_continuous", xrd_wavelength=1.54059,
        )

        cond = specs[0]["condition_vector"]
        assert isinstance(cond, list) and len(cond) == 1000
        assert all(len(pair) == 2 for pair in cond)
        assert cond[0][0] == 0.0, "Q grid must start at 0.0"
        assert abs(max(pair[1] for pair in cond) - 1.0) < 1e-9, "intensity must be max-normalized"

    def test_condition_lists_uneven_input_rejected(self) -> None:
        """Ragged --condition_lists strings raise instead of silently dropping values."""
        from _utils._direct_gen_utils import build_formula_condition_map

        try:
            build_formula_condition_map(
                ["NaCl", "KCl"], ["2.16, 0.0", "3.0, 0.1, 7.7"], "c-bone/CrystaLLM-pi_density"
            )
            assert False, "uneven condition lists should raise"
        except ValueError as err:
            assert "same number" in str(err), f"Unexpected error: {err}"

        # The corrected per-formula form still round-trips (values normalized per dimension)
        cond = build_formula_condition_map(
            ["NaCl"], ["2.16, 0.0"], "c-bone/CrystaLLM-pi_density"
        )
        assert len(cond) == 1 and len(cond[0].split(",")) == 2
