"""Local test section: load and generate."""

import os
import argparse
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
    
    def test_hf_model_loading(self):
        """Test HF model loading functionality."""
        import _load_and_generate
    
    def test_prompt_generation_from_args(self):
        """Test underlying manual prompt generation tool."""
        from _utils._generating.make_prompts import create_manual_prompts
        
        df_prompts = create_manual_prompts(
            compositions=["Si1O2"],
            condition_lists=[["0.5"]],
            level="level_2",
            spacegroups=None,
            raw_mode=False
        )
        
        assert len(df_prompts) > 0, "Should generate prompts"
        assert 'Prompt' in df_prompts.columns, "Should have Prompt column"

    def test_xrd_raw_file_parsing_and_conversion(self):
        """Verify raw XRD files are dynamically processed and scaled."""
        from _utils._direct_gen_utils import parse_xrd_file_to_condition_vector
        raw_xy = os.path.join(fixtures_dir, "test_rutile_raw.xy")
        
        if not os.path.exists(raw_xy):
            print("Skipping raw XRD test, fixture not found.")
            return
            
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
            target_valid_cifs=1,
            num_workers=1,
            skip_postprocess=True,
            verbose=False,
            output_parquet=os.path.join(self.temp_dir, "mattergen_smoke.parquet"),
            output_cif_dir=None,
        )

        canonical = _direct_gen_utils.canonicalize_reduced_formulas(["TiO2"])
        property_map = {"TiO2": {"xrd": xrd_file, "sg": "P4_2/mnm", "cond": None}}
        specs = _direct_gen_utils.build_reduced_formula_specs(canonical, {"TiO2": [2]}, property_map, is_xrd=True, xrd_wavelength=1.54056)
        
        df_prompts = _load_and_generate.generate_prompts_from_specs(specs, args)
        assert len(df_prompts) == 1, "Smoke test should create exactly one prompt"

        df_generated = _load_and_generate.generate_cifs_with_hf_model(
            df_prompts=df_prompts,
            hf_model_path=args.hf_model_path,
            args=args,
        )
        assert "generated_cif" in df_generated.columns or "Generated CIF" in df_generated.columns
        assert len(df_generated) >= 1, "Smoke generation should return at least one row"

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
        """Single prompt should still enable multi-GPU when forced and GPUs are visible."""
        from _utils import _direct_gen_utils

        original_get_visible_gpu_count = _direct_gen_utils.get_visible_gpu_count
        try:
            _direct_gen_utils.get_visible_gpu_count = lambda: 4
            args = argparse.Namespace(
                output_cif_dir=None,
                multi_gpu="true",
                num_workers_gpu=None,
            )
            workers = _direct_gen_utils.resolve_multi_gpu_workers(args, n_prompts=1)
            assert workers == 4, f"Expected 4 workers for single prompt fanout, got {workers}"
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
        property_map = {f: {"xrd": None, "sg": None, "cond": None} for f in canonical}
        z_mapping = {f: [1, 2, 3, 4] for f in canonical}

        specs = _direct_gen_utils.build_reduced_formula_specs(canonical, z_mapping, property_map, is_xrd=False)
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

    def test_reduced_formula_selection_uses_provided_cif_text(self):
        """Selection should validate the CIF text as provided when consistency flags are absent."""
        from _utils import _direct_gen_utils

        df_prompts = pd.DataFrame([
            {"Material ID": "TiO2_Z1", "reduced_formula_target": "TiO2", "Z_search": 1, "prompt_order": 1},
        ])
        df_generated = pd.DataFrame([
            {"Material ID": "TiO2_Z1_1", "Generated CIF": "raw_cif", "score": 0.1},
        ])

        original_validity_worker = _direct_gen_utils._validity_worker
        try:
            _direct_gen_utils._validity_worker = lambda cif, **kwargs: cif == "raw_cif"

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
            _direct_gen_utils._validity_worker = original_validity_worker

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