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