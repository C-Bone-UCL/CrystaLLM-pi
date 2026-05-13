"""API endpoint tests: preprocessing."""

import os

from tests.api.endpoints_sections._base import IntegrationMixin

class PreprocessingEndpointTests(IntegrationMixin):
    """Test preprocessing endpoints."""
    
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
        
    def test_deduplicate_valid_request(self):
        """Test deduplicate with valid request."""
        if self._should_skip_integration():
            return

        output_parquet = self._out("test_dedup.parquet") if self.is_integration else os.path.join(self.temp_dir, "dedup_out.parquet")
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_parquet,
            "property_columns": "[\"Bandgap (eV)\"]",
            "filter_na_columns": "[\"Bandgap (eV)\"]"
        })
        data = self._wait_and_assert(response, job_name="deduplicate")
        if self.is_integration:
            assert data["status"] == "completed"
        else:
            assert data["status"] in ["pending", "running"]
        assert "deduplicate" in data["command"].lower() or "_deduplicate" in data["command"]
        
    def test_deduplicate_with_all_optional_params(self):
        """Test deduplicate with all optional parameters."""
        if self._should_skip_integration():
            return

        output_parquet = self._out("test_dedup_full.parquet") if self.is_integration else os.path.join(self.temp_dir, "dedup_full.parquet")
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_parquet,
            "property_columns": "[\"Bandgap (eV)\"]",
            "filter_na_columns": "[\"Bandgap (eV)\"]",
            "filter_zero_columns": "[\"Density (g/cm^3)\"]",
            "filter_negative_columns": "[\"Bandgap (eV)\"]"
        })
        data = self._wait_and_assert(response, job_name="deduplicate_optional")
        # check command includes all params
        assert "--property_columns" in data["command"]
        assert "--filter_na_columns" in data["command"]
        
    def test_deduplicate_missing_required_field(self):
        """Test deduplicate fails with missing required field."""
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/some/path.parquet"
            # missing output_parquet
        })
        assert response.status_code == 422, f"Expected 422 validation error, got {response.status_code}"
        
    def test_clean_valid_request(self):
        """Test clean with valid request."""
        if self._should_skip_integration():
            return

        output_parquet = self._out("test_clean.parquet") if self.is_integration else os.path.join(self.temp_dir, "clean_out.parquet")
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_parquet,
            "num_workers": 4,
            "property_columns": "[\"Bandgap (eV)\"]",
            "property1_normaliser": "power_log"
        })
        data = self._wait_and_assert(response, job_name="clean")
        assert "_cleaning" in data["command"]
        
    def test_clean_with_normalizers(self):
        """Test clean with property normalizers."""
        if self._should_skip_integration():
            return

        output_parquet = self._out("test_clean_norm.parquet") if self.is_integration else os.path.join(self.temp_dir, "clean_norm.parquet")
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_parquet,
            "num_workers": 4,
            "property_columns": "[\"Bandgap (eV)\", \"Density (g/cm^3)\"]",
            "property1_normaliser": "power_log",
            "property2_normaliser": "linear"
        })
        data = self._wait_and_assert(response, job_name="clean_with_norm")
        assert "--property1_normaliser" in data["command"]
        assert "power_log" in data["command"]
        
    def test_clean_invalid_normalizer(self):
        """Test clean rejects invalid normalizer."""
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "clean_bad.parquet"),
            "property1_normaliser": "invalid_method"
        })
        assert response.status_code == 422, "Should reject invalid normalizer"
        
    def test_save_dataset_valid_request(self):
        """Test save-dataset with valid request."""
        if self._should_skip_integration():
            return

        output_path = self._out("test_dataset_api") if self.is_integration else "test_dataset"
        test_size = 0.0 if self.is_integration else 0.1
        valid_size = 0.0 if self.is_integration else 0.1
        response = self.client.post("/preprocessing/save-dataset", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_path,
            "test_size": test_size,
            "valid_size": valid_size,
            "HF_username": "test-user",
            "save_hub": False,  # don't actually push
            "save_local": True
        })
        data = self._wait_and_assert(response, job_name="save_dataset")
        assert "_save_dataset_to_HF" in data["command"]
        
    def test_save_dataset_with_splits(self):
        """Test save-dataset with custom splits."""
        if self._should_skip_integration():
            return

        output_path = self._out("test_dataset_splits") if self.is_integration else "test_dataset"
        response = self.client.post("/preprocessing/save-dataset", json={
            "input_parquet": self._input_parquet(self.test_data['test_file']),
            "output_parquet": output_path,
            "HF_username": "test-user",
            "test_size": 0.15,
            "valid_size": 0.15,
            "save_hub": False,
            "save_local": True,
            "duplicates": True
        })
        data = self._wait_and_assert(response, job_name="save_dataset_splits")
        assert "--test_size" in data["command"]
        assert "0.15" in data["command"]

    def test_xrd_preprocessing_valid_request(self):
        """Test XRD preprocessing endpoint with fixture CSV input."""
        if self._should_skip_integration():
            return

        output_csv = self._out("xrd_processed.csv") if self.is_integration else os.path.join(self.temp_dir, "xrd_processed.csv")
        xrd_input = "/app/tests/fixtures/test_rutile_processed.csv" if self.is_integration else self.test_data["fixture_xrd_csv"]
        response = self.client.post("/preprocessing/process-exp-xrd", json={
            "input_data": xrd_input,
            "xrd_wavelength": 1.54056,
            "output_csv": output_csv
        })
        data = self._wait_and_assert(response, job_name="xrd_preprocessing")
        assert "_process_exp_XRD_inputs" in data["command"]
        assert "--input_data" in data["command"]
        assert "--output_csv" in data["command"]

    def test_xrd_preprocessing_missing_required_field(self):
        """Test XRD preprocessing rejects missing required output field."""
        response = self.client.post("/preprocessing/process-exp-xrd", json={
            "input_data": self.test_data["fixture_xrd_csv"]
        })
        assert response.status_code == 422, "Should reject missing required fields"

    def test_calc_theor_xrd_smoke(self):
        """Test theoretical XRD preprocessing endpoint command construction and execution."""
        if self._should_skip_integration():
            return

        input_parquet = self._out("test_dedup.parquet") if self.is_integration else os.path.join(self.temp_dir, "dedup_out.parquet")
        output_parquet = self._out("test_dedup_with_xrd.parquet") if self.is_integration else os.path.join(self.temp_dir, "dedup_with_xrd.parquet")
        response = self.client.post("/preprocessing/calc-theor-xrd", json={
            "input_parquet": input_parquet,
            "output_parquet": output_parquet,
            "num_workers": 2,
            "column_name": "CIF"
        })
        data = self._wait_and_assert(response, job_name="calc_theor_xrd")
        assert "_calculate_theor_XRD" in data["command"]
        assert "--input_parquet" in data["command"]
        assert "--output_parquet" in data["command"]

        if self.is_integration:
            local_output = self._get_local_path(output_parquet)
            assert os.path.exists(local_output), f"Expected calc-theor-xrd output not found: {local_output}"

    def test_cifs_zip_to_parquet_smoke(self):
        """Smoke test CIF tarball preprocessing endpoint command construction."""
        output_parquet = os.path.join(self.temp_dir, "cifs_from_tarballs.parquet")
        response = self.client.post("/preprocessing/cifs-zip-to-parquet", json={
            "input_tarballs": ["/tmp/train.tar.gz", "/tmp/val.tar.gz"],
            "output_parquet": output_parquet,
            "database_name": "benchmark_db"
        })
        assert response.status_code == 200
        data = response.json()
        assert "_cifs_zip_to_parquet" in data["command"]
        assert "--input_tarballs" in data["command"]
        assert "--output_parquet" in data["command"]
        assert "--database_name" in data["command"]
