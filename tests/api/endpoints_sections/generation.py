"""API endpoint tests: generation."""

import os
import json

from tests.api.endpoints_sections._base import IntegrationMixin

class GenerationEndpointTests(IntegrationMixin):
    """Test generation endpoints."""
    
    def __init__(
        self,
        client,
        temp_dir: str,
        test_data: dict,
        mode: str = "smoke",
        docker_mode: bool = False,
        verbose: bool = False,
        integration_tag: str | None = None,
    ):
        self._init_integration(
            client=client,
            temp_dir=temp_dir,
            test_data=test_data,
            mode=mode,
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )

    def _test_output(self, filename: str, smoke_name: str | None = None) -> str:
        if self.is_integration:
            return self._out(filename)
        return os.path.join(self.temp_dir, smoke_name or filename)

    def test_generate_base_explicit_z(self):
        """Test unconditional generation with explicit Z and spacegroup targeting."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("base_generated.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "num_return_sequences": 5,
            "max_return_attempts": 2,
            "output_parquet": output_parquet
        })
        data = self._wait_and_assert(response, job_name="generate_base_explicit_z", timeout=600)
        assert "_load_and_generate" in data["command"]
        assert "--reduced_formula_list TiO2" in data["command"]
        assert "--z_list 2" in data["command"]

        if self.is_integration and self.verbose:
            self._show_generated_cifs(output_parquet)
        
    def test_direct_generation_mapped_lists(self):
        """Test direct generation with mapped parallel lists for multiple structures."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_mapped.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": output_parquet,
            "reduced_formula_list": "TiO2,SiO2",
            "z_list": "2,4",
            "condition_lists": ["1.8,0.0", "5.0,0.0"],
            "level": "level_3",
            "num_return_sequences": 5
        })
        data = self._wait_and_assert(response, job_name="generate_mapped_lists", timeout=600)
        assert "--reduced_formula_list TiO2,SiO2" in data["command"]
        assert "--z_list 2,4" in data["command"]
        assert "--condition_lists" in data["command"]
        
    def test_direct_generation_slme_level_1(self):
        """Test Level 1 direct generation fallback without formulas."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("generated.parquet", "gen_cond.parquet")
        # Add target_valid_cifs to force a file output
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_SLME",
            "output_parquet": output_parquet,
            "condition_lists": ["25.0"],
            "level": "level_1",
            "num_return_sequences": 5,
            "max_return_attempts": 5,
            "target_valid_cifs": 1
        })
        data = self._wait_and_assert(response, job_name="generate_slme_lvl1", timeout=600)
        assert "--reduced_formula_list X" in data["command"]
        assert "--condition_lists" in data["command"]

        if self.is_integration:
            self._assert_output_exists(output_parquet, label="generated parquet")

        if self.is_integration and self.verbose:
            self._show_generated_cifs(output_parquet)
        
    def test_direct_generation_cod_xrd_early_stop(self):
        """Test Early-Stopping Z-Search with COD-XRD."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_xrd_early_stop.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_COD-XRD",
            "output_parquet": output_parquet,
            "reduced_formula_list": "TiO2",
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "search_zs": True,
            "xrd_files": ["/app/tests/fixtures/test_rutile_processed.csv"],
            "num_return_sequences": 5,
            "max_return_attempts": 2,
            "target_valid_cifs": 1,
            "scoring_mode": "none"
        })
        data = self._wait_and_assert(response, job_name="generate_xrd_early_stop", timeout=600)
        assert "--search_zs" in data["command"]
        assert "--target_valid_cifs 1" in data["command"]
        assert "--xrd_files" in data["command"]
        assert "/app/tests/fixtures/test_rutile_processed.csv" in data["command"]

    def test_direct_generation_mattergen_xrd_logp(self):
        """Test Mattergen-XRD generation with LOGP ranked Z-Search."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_mattergen_xrd.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
            "output_parquet": output_parquet,
            "reduced_formula_list": "TiO2",
            "search_zs": True,
            "xrd_files": ["/app/tests/fixtures/test_rutile_processed.csv"],
            "num_return_sequences": 10,
            "max_return_attempts": 2,
            "target_valid_cifs": 5,
            "scoring_mode": "LOGP",
            "temperature": 1.0
        })
        data = self._wait_and_assert(response, job_name="generate_mattergen_xrd_logp", timeout=900)
        cmd = data["command"]
        assert "c-bone/CrystaLLM-pi_Mattergen-XRD" in cmd
        assert "--search_zs" in cmd
        assert "--scoring_mode LOGP" in cmd
        assert "--target_valid_cifs 5" in cmd

    def test_direct_generation_mattergen_xrd_without_xrd_files(self):
        """Mattergen-XRD should still dispatch without xrd_files for missing Slider conditioning."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_mattergen_xrd_no_xrd.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
            "output_parquet": output_parquet,
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "num_return_sequences": 2,
            "target_valid_cifs": 1,
        })
        data = self._wait_and_assert(response, job_name="generate_mattergen_xrd_no_xrd", timeout=600)
        cmd = data["command"]
        assert "c-bone/CrystaLLM-pi_Mattergen-XRD" in cmd
        assert "--reduced_formula_list TiO2" in cmd
        assert "--z_list 2" in cmd
        assert "--xrd_files" not in cmd

    def test_direct_generation_raw_xrd_conversion(self):
        """Test direct generation handles raw XRD files and wavelength conversion."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_raw_xrd.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "xrd_files": ["/app/tests/fixtures/test_rutile_raw.xy"],
            "xrd_wavelength": 0.71073,
            "num_return_sequences": 2,
            "output_parquet": output_parquet
        })
        data = self._wait_and_assert(response, job_name="generate_raw_xrd", timeout=600)
        cmd = data["command"]
        assert "--xrd_files /app/tests/fixtures/test_rutile_raw.xy" in cmd
        assert "--xrd_wavelength 0.71073" in cmd
        
    def test_direct_generation_input_parquet_mode(self):
        if self._should_skip_integration(): return
        output_parquet = self._test_output("gen_parquet.parquet")
        
        # FIX: Point to the prompts file created by the earlier automatic prompts test
        input_parquet = self._out("prompts_auto.parquet") if self.is_integration else self._input_parquet(self.test_data['test_file'])
        
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": output_parquet,
            "input_parquet": input_parquet
        })
        data = self._wait_and_assert(response, job_name="generate_input_parquet", timeout=600)

    def test_direct_generation_reduced_formula_conflict(self):
        """Reduced formula mode should reject input_parquet in the same request."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_reduced_formula_conflict.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": output_parquet,
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "reduced_formula_list": "SiO2",
        })
        assert response.status_code == 422
        
    def test_make_prompts_manual(self):
        """Test make-prompts in manual mode."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("prompts.parquet")
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": output_parquet,
            "manual": True,
            "compositions": "Li1Fe1P1O4,Na1Mn1O2,Si1O2",
            "condition_lists": ["0.5,0.0"],
            "level": "level_2"
        })
        data = self._wait_and_assert(response, job_name="make_prompts")
        assert "make_prompts" in data["command"]
        assert "--level" in data["command"]
        assert "level_2" in data["command"]
        
    def test_make_prompts_automatic(self):
        """Test make-prompts in automatic mode."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("prompts_auto.parquet")
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": output_parquet,
            "manual": False,
            "automatic": True,
            "HF_dataset": "c-bone/SLME_1K_dtest",
            "split": "test",
            "level": "level_2"
        })
        data = self._wait_and_assert(response, job_name="make_prompts_automatic")
        assert "--automatic" in data["command"]
        assert "--HF_dataset" in data["command"]

    def test_generate_cifs(self):
        """Test CIF generation with config."""
        if self._should_skip_integration():
            return

        if self.is_integration:
            container_config_path = self._out("gen_config.jsonc")
            # Clever trick: strip '/app/' to get the local host path!
            local_config_path = container_config_path.replace("/app/", "")
            
            input_prompts = self._out("prompts_auto.parquet")
            output_cifs = self._out("gen_cifs_from_config.parquet")
        else:
            local_config_path = os.path.join(self.temp_dir, "gen_config.jsonc")
            container_config_path = local_config_path
            input_prompts = self.test_data['test_file']
            output_cifs = os.path.join(self.temp_dir, "out.parquet")

        config = {
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "input_parquet": input_prompts,
            "output_parquet": output_cifs,
            "num_return_sequences": 2,
            "max_return_attempts": 1,
            "max_samples": 2,
            }
        
        with open(local_config_path, 'w') as f:
            json.dump(config, f)
            
        response = self.client.post("/generate/cifs", json={
            "config_file": container_config_path
        })
        data = self._wait_and_assert(response, job_name="generate_cifs", timeout=600)
        assert "generate_CIFs" in data["command"]

    def test_direct_generation_search_zs_all_rows_mode(self):
        """search_zs with target_valid_cifs=0 should pass through all generated rows mode."""
        if self._should_skip_integration():
            return

        output_parquet = self._test_output("gen_search_zs_all_rows.parquet")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": output_parquet,
            "reduced_formula_list": "SiO2,TiO2",
            "search_zs": True,
            "target_valid_cifs": 0,
            "num_return_sequences": 5,
            "max_return_attempts": 2,
            "scoring_mode": "None",
        })
        data = self._wait_and_assert(response, job_name="generate_search_zs_all_rows", timeout=600)
        cmd = data["command"]
        assert "--search_zs" in cmd
        assert "--target_valid_cifs 0" in cmd
        assert "--scoring_mode None" in cmd
        
    def test_evaluate_cifs(self):
        """Test CIF evaluation endpoint."""
        if self._should_skip_integration():
            return

        input_parquet = self._out("base_generated.parquet") if self.is_integration else self.test_data['test_file']
        if self.is_integration:
            self._assert_output_exists(input_parquet, label="evaluate input parquet")
        save_valid_parquet = self._test_output("valid_cifs.parquet")
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": input_parquet,
            "num_workers": 4,
            "save_valid_parquet": save_valid_parquet
        })
        data = self._wait_and_assert(response, job_name="evaluate_cifs")
        assert "evaluate_CIFs" in data["command"]
        assert "--save_valid_parquet" in data["command"]
        
    def test_postprocess(self):
        """Test postprocessing endpoint."""
        if self._should_skip_integration():
            return

        input_parquet = self._out("base_generated.parquet") if self.is_integration else self.test_data['test_file']
        if self.is_integration:
            self._assert_output_exists(input_parquet, label="postprocess input parquet")
        output_parquet = self._test_output("postprocessed.parquet", "postproc.parquet")
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": input_parquet,
            "output_parquet": output_parquet,
            "num_workers": 2
        })
        data = self._wait_and_assert(response, job_name="postprocess")
        assert "postprocess" in data["command"]

        if self.is_integration:
            self._assert_output_exists(output_parquet, label="postprocessed parquet")