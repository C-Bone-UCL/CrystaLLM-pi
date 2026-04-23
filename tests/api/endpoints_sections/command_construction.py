"""API endpoint tests: command construction."""

class CommandConstructionTests:
    """Test that CLI commands are constructed correctly."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_deduplicate_command_structure(self):
        """Verify deduplicate command has correct structure."""
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/data/in.parquet",
            "output_parquet": "/data/out.parquet",
            "property_columns": "[\"Bandgap\"]"
        })
        cmd = response.json()["command"]
        
        # check command structure
        assert cmd.startswith("python -m _utils._preprocessing._deduplicate")
        assert "--input_parquet /data/in.parquet" in cmd
        assert "--output_parquet /data/out.parquet" in cmd
        assert '--property_columns ["Bandgap"]' in cmd or "--property_columns [\"Bandgap\"]" in cmd
        
    def test_direct_generation_condition_lists_format(self):
        """Verify condition_lists are passed correctly to CLI."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_slme",
            "output_parquet": "/out.parquet",
            "level": "level_1",
            "condition_lists": ["24.7"]
        })
        cmd = response.json()["command"]
        
        # condition_lists should be space-separated after the flag
        assert "--condition_lists 24.7" in cmd, f"Expected condition_lists format not found in: {cmd}"
        
    def test_direct_generation_all_params_in_command(self):
        """Verify all generation params are in command."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "condition_lists": ["1.82,0.0"],
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "num_return_sequences": 5,
            "max_return_attempts": 10,
            "temperature": 0.9
        })
        cmd = response.json()["command"]
        
        # all params should be present
        assert "--spacegroups P4_2/mnm" in cmd
        assert "--level level_4" in cmd
        assert "--num_return_sequences 5" in cmd
        assert "--max_return_attempts 10" in cmd
        assert "--temperature 0.9" in cmd
        assert "c-bone/CrystaLLM-pi_bandgap" in cmd

    def test_direct_generation_output_cif_dir_without_parquet(self):
        """Verify direct generation accepts output_cif_dir as the only output target."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_cif_dir": "/out-cifs",
            "reduced_formula_list": "TiO2",
            "z_list": "2"
        })

        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--output_cif_dir /out-cifs" in cmd
        assert "--output_parquet" not in cmd

    def test_direct_generation_rejects_search_zs_with_z_list(self):
        """Verify direct generation rejects mutually exclusive Z controls."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2",
            "search_zs": True,
            "z_list": "2"
        })
        assert response.status_code == 422
        assert "search_zs" in response.json()["detail"]

    def test_direct_generation_rejects_xrd_file_count_mismatch(self):
        """Verify direct generation rejects XRD file counts that do not match formulas."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2,SiO2",
            "xrd_files": ["/tmp/test_rutile_processed.csv"]
        })
        assert response.status_code == 422
        assert "XRD files" in response.json()["detail"]

    def test_direct_generation_rejects_spacegroup_count_mismatch(self):
        """Verify direct generation rejects incompatible spacegroup mappings."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2,SiO2",
            "spacegroups": "P4_2/mnm,P6_3/mmc,Fm-3m"
        })
        assert response.status_code == 422
        assert "spacegroups" in response.json()["detail"]

    def test_direct_generation_rejects_condition_vector_count_mismatch(self):
        """Verify direct generation rejects incompatible condition vector mappings."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2,SiO2",
            "condition_lists": ["1.2,0.0", "3.1,0.0", "5.0,0.0"]
        })
        assert response.status_code == 422
        assert "condition vector" in response.json()["detail"]

    def test_direct_generation_scoring_mode_case_passthrough(self):
        """Verify lower-case scoring mode is accepted and forwarded."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "condition_lists": ["1.82,0.0"],
            "scoring_mode": "logp"
        })
        cmd = response.json()["command"]
        assert "--scoring_mode logp" in cmd

    def test_direct_generation_logp_zero_target_rejected(self):
        """LOGP scoring should require target_valid_cifs > 0."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "condition_lists": ["1.82,0.0"],
            "scoring_mode": "logp",
            "target_valid_cifs": 0
        })
        assert response.status_code == 422
        assert "target_valid_cifs > 0" in response.json()["detail"]

    def test_direct_generation_none_zero_target_allowed(self):
        """None scoring should allow target_valid_cifs = 0 and forward command."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "reduced_formula_list": "TiO2",
            "z_list": "2",
            "condition_lists": ["1.82,0.0"],
            "scoring_mode": "None",
            "target_valid_cifs": 0
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--scoring_mode None" in cmd
        assert "--target_valid_cifs 0" in cmd
        
    def test_train_torchrun_format(self):
        """Verify torchrun command format for multi-GPU."""
        response = self.client.post("/train", json={
            "config_file": "/config.jsonc",
            "multi_gpu": True,
            "nproc_per_node": 2
        })
        cmd = response.json()["command"]
        
        assert cmd.startswith("torchrun"), f"Should start with torchrun: {cmd}"
        assert "--nproc_per_node=2" in cmd
        assert "-m _train" in cmd
        assert "--config /config.jsonc" in cmd

    def test_make_prompts_automatic_extended_args(self):
        """Verify automatic prompt generation forwards the newer helper args."""
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": "/tmp/prompts.parquet",
            "automatic": True,
            "input_parquet": "/tmp/input.parquet",
            "split": "test",
            "level": "level_3",
            "raw": True,
            "cif_column": "Generated CIF",
            "condition_columns": ["Bandgap (eV)"],
            "remove_ref_columns": True
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--automatic" in cmd
        assert "--input_parquet /tmp/input.parquet" in cmd
        assert "--raw" in cmd
        assert "--cif_column Generated CIF" in cmd
        assert "--condition_columns Bandgap (eV)" in cmd
        assert "--remove_ref_columns" in cmd

    def test_make_prompts_manual_mode_passthrough(self):
        """Verify manual prompt generation forwards the pairing mode."""
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": "/tmp/prompts.parquet",
            "manual": True,
            "compositions": "LiFePO4,NaMnO2",
            "condition_lists": ["0.1", "0.2"],
            "mode": "paired"
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--manual" in cmd
        assert "--mode paired" in cmd

    def test_evaluate_cifs_extended_args(self):
        """Verify evaluate-cifs forwards metrics_out and debug flags."""
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": "/tmp/generated.parquet",
            "metrics_out": "/tmp/metrics.parquet",
            "num_workers": 4,
            "save_valid_parquet": "/tmp/valid.parquet",
            "debug": True
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--metrics_out /tmp/metrics.parquet" in cmd
        assert "--save_valid_parquet /tmp/valid.parquet" in cmd
        assert "--debug" in cmd

    def test_postprocess_column_name_passthrough(self):
        """Verify postprocess forwards the CIF column override."""
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": "/tmp/generated.parquet",
            "output_parquet": "/tmp/post.parquet",
            "num_workers": 4,
            "column_name": "CIF"
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--column_name CIF" in cmd