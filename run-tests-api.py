"""
API test suite for CrystaLLM-2.0 FastAPI endpoints.
Tests all REST API functionality including preprocessing, training, generation, 
and evaluation metrics endpoints. Uses FastAPI's TestClient for synchronous testing.

Usage:
    # Test against running Docker container (recommended)
    python run-tests-api.py --docker_url http://localhost:8000
    
    # Test locally with FastAPI TestClient (requires fastapi installed)
    python run-tests-api.py --hf_key YOUR_HF_KEY --wandb_key YOUR_WANDB_KEY
    
    # Run integration tests (slower, actually executes commands)
    python run-tests-api.py --docker_url http://localhost:8000 --integration
    
    # Run integration tests with verbose output (shows sample CIFs, VUN stats, E-hull values)
    python run-tests-api.py --docker_url http://localhost:8000 --integration --verbose

If we need to build: 
    docker build -t crystallm-api:local .


Docker setup (run this first):
    docker run --rm --gpus all -p 8000:8000 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/outputs:/app/outputs \
        -e HF_KEY="" \
        -e WANDB_KEY="" \
        --name crystallm-api \
        crystallm-api:local
"""

import os
import sys
import argparse
import tempfile
import shutil
import time
import json
from typing import Dict, Any, Optional, List

import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(script_dir)


class APITestSuite:
    """Main API test coordinator with temporary file management."""
    
    def __init__(self, hf_key: str, wandb_key: str, docker_url: Optional[str] = None):
        self.hf_key = hf_key
        self.wandb_key = wandb_key
        self.docker_url = docker_url
        self.temp_dir = None
        self.results = {}
        self.verbose = True
        self.client = None
        
    def setup(self):
        """Create temporary test environment and initialize test client."""
        self.temp_dir = tempfile.mkdtemp(prefix="crystallm_api_test_")
        if self.verbose:
            print(f"Test directory: {self.temp_dir}")
        
        # Set environment variables
        os.environ["HF_KEY"] = self.hf_key
        os.environ["WANDB_KEY"] = self.wandb_key
        
        if self.docker_url:
            # Use requests for Docker testing
            import requests
            self.client = DockerTestClient(self.docker_url)
        else:
            # Use FastAPI TestClient for local testing
            from fastapi.testclient import TestClient
            from crystallm_api import app, jobs
            jobs.clear()  # clear any existing jobs
            self.client = TestClient(app)
            
    def cleanup(self):
        """Remove all test files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up: {self.temp_dir}")
            
    def run_test(self, test_name: str, test_func):
        """Run individual test with error handling."""
        try:
            print(f"\nRunning: {test_name}...")
            test_func()
            self.results[test_name] = "PASS"
            print(f"  ✓ PASS")
        except Exception as e:
            self.results[test_name] = f"FAIL: {str(e)}"
            print(f"  ✗ FAIL: {str(e)}")
            
    def create_test_data(self) -> Dict[str, Any]:
        """Create minimal test CIF data."""
        test_cif = """# generated using pymatgen
data_SiO2
_symmetry_space_group_name_H-M   Pbcm
_cell_length_a   4.69102593
_cell_length_b   9.04290627
_cell_length_c   4.70625094
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   57
_chemical_formula_structural   SiO2
_chemical_formula_sum   'Si4 O8'
_cell_volume   199.64155449
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
  2  '-x, -y, -z'
  3  '-x, -y, z+1/2'
  4  'x, y, -z+1/2'
  5  'x, -y+1/2, -z'
  6  '-x, y+1/2, z'
  7  '-x, y+1/2, -z+1/2'
  8  'x, -y+1/2, z+1/2'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Si  Si0  4  0.25683649  0.75000000  0.00000000  1
  O  O1  4  0.07596292  0.65774525  0.25000000  1
  O  O2  4  0.43761182  0.65761148  0.75000000  1"""

        # Create test dataframe
        test_df = pd.DataFrame({
            'Material ID': ['test-1', 'test-2', 'test-3'],
            'Reduced Formula': ['SiO2', 'TiO2', 'Al2O3'], 
            'CIF': [test_cif, test_cif.replace('SiO2', 'TiO2').replace('Si', 'Ti'), 
                    test_cif.replace('SiO2', 'Al2O3').replace('Si', 'Al')],
            'Database': ['test'] * 3,
            'Bandgap (eV)': [2.5, 3.1, 8.8],
            'Density (g/cm^3)': [2.2, 4.2, 3.9]
        })
        
        test_file = os.path.join(self.temp_dir, "test_data.parquet")
        test_df.to_parquet(test_file, index=False)
        
        return {
            'test_cif': test_cif,
            'test_file': test_file,
            'test_df': test_df
        }
        
    def report_results(self):
        """Print final test results."""
        print("\n" + "="*60)
        print("API TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.results.values() if result == "PASS")
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "✓" if result == "PASS" else "✗"
            print(f"  {status} {test_name}: {result}")
        
        print(f"\nPassed: {passed}/{total} tests")
        if passed == total:
            print("All tests passed!")
            return 0
        else:
            print(f"Failed: {total - passed} tests")
            return 1


class DockerTestClient:
    """Simple wrapper for testing against a running Docker container."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        import requests
        self.session = requests.Session()
        
    def get(self, path: str) -> 'MockResponse':
        resp = self.session.get(f"{self.base_url}{path}")
        return MockResponse(resp)
        
    def post(self, path: str, json: dict = None) -> 'MockResponse':
        resp = self.session.post(f"{self.base_url}{path}", json=json)
        return MockResponse(resp)


class MockResponse:
    """Wrapper to make requests.Response look like TestClient response."""
    
    def __init__(self, resp):
        self._resp = resp
        self.status_code = resp.status_code
        
    def json(self):
        return self._resp.json()


# ==================== ENDPOINT TESTS ====================

class RootEndpointTests:
    """Test the root endpoint."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_root_returns_api_info(self):
        """Test that root endpoint returns API info."""
        response = self.client.get("/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "name" in data, "Response should contain 'name'"
        assert "version" in data, "Response should contain 'version'"
        assert "endpoints" in data, "Response should contain 'endpoints'"
        
    def test_root_lists_all_endpoint_categories(self):
        """Test that root lists all endpoint categories."""
        response = self.client.get("/")
        data = response.json()
        
        expected_categories = ["preprocessing", "training", "generation", "metrics", "jobs"]
        for cat in expected_categories:
            assert cat in data["endpoints"], f"Missing category: {cat}"


class JobManagementTests:
    """Test job management endpoints."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        response = self.client.get("/jobs")
        assert response.status_code == 200
        # can be empty or have jobs from other tests
        
    def test_get_nonexistent_job_returns_404(self):
        """Test getting a job that doesn't exist."""
        response = self.client.get("/jobs/nonexistent-job-id-12345")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        
    def test_job_creation_returns_pending_status(self):
        """Test that creating a job returns pending status."""
        # use a simple endpoint to create a job
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/tmp/fake_input.parquet",
            "output_parquet": "/tmp/fake_output.parquet"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending", f"Expected 'pending', got {data['status']}"
        assert "job_id" in data, "Response should have job_id"
        assert "command" in data, "Response should have command"
        
    def test_job_can_be_retrieved_after_creation(self):
        """Test that a created job can be retrieved."""
        # create job
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": "/tmp/test_input.parquet",
            "output_parquet": "/tmp/test_output.parquet"
        })
        job_id = response.json()["job_id"]
        
        # retrieve it
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id


class PreprocessingEndpointTests:
    """Test preprocessing endpoints."""
    
    def __init__(self, client, temp_dir: str, test_data: dict):
        self.client = client
        self.temp_dir = temp_dir
        self.test_data = test_data
        
    def test_deduplicate_valid_request(self):
        """Test deduplicate with valid request."""
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "dedup_out.parquet")
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["pending", "running"]
        assert "deduplicate" in data["command"].lower() or "_deduplicate" in data["command"]
        
    def test_deduplicate_with_all_optional_params(self):
        """Test deduplicate with all optional parameters."""
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "dedup_full.parquet"),
            "property_columns": "[\"Bandgap (eV)\"]",
            "filter_na_columns": "[\"Bandgap (eV)\"]",
            "filter_zero_columns": "[\"Density (g/cm^3)\"]",
            "filter_negative_columns": "[\"Bandgap (eV)\"]"
        })
        assert response.status_code == 200
        data = response.json()
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
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "clean_out.parquet")
        })
        assert response.status_code == 200
        data = response.json()
        assert "_cleaning" in data["command"]
        
    def test_clean_with_normalizers(self):
        """Test clean with property normalizers."""
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "clean_norm.parquet"),
            "num_workers": 4,
            "property_columns": "[\"Bandgap (eV)\", \"Density (g/cm^3)\"]",
            "property1_normaliser": "power_log",
            "property2_normaliser": "linear"
        })
        assert response.status_code == 200
        data = response.json()
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
        response = self.client.post("/preprocessing/save-dataset", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": "test_dataset",
            "HF_username": "test-user",
            "save_hub": False,  # don't actually push
            "save_local": True
        })
        assert response.status_code == 200
        data = response.json()
        assert "_save_dataset_to_HF" in data["command"]
        
    def test_save_dataset_with_splits(self):
        """Test save-dataset with custom splits."""
        response = self.client.post("/preprocessing/save-dataset", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": "test_dataset",
            "HF_username": "test-user",
            "test_size": 0.15,
            "valid_size": 0.15,
            "save_hub": False,
            "save_local": True,
            "duplicates": True
        })
        assert response.status_code == 200
        data = response.json()
        assert "--test_size" in data["command"]
        assert "0.15" in data["command"]


class TrainingEndpointTests:
    """Test training endpoints."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_train_single_gpu(self):
        """Test train with single GPU config."""
        # create minimal config
        config = {
            "model_name": "test-model",
            "huggingface_dataset": "c-bone/SLME_1K_dtest"
        }
        config_path = os.path.join(self.temp_dir, "test_train.jsonc")
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        response = self.client.post("/train", json={
            "config_file": config_path,
            "multi_gpu": False
        })
        assert response.status_code == 200
        data = response.json()
        assert "python" in data["command"]
        assert "_train" in data["command"]
        assert "torchrun" not in data["command"]
        
    def test_train_multi_gpu(self):
        """Test train with multi-GPU config."""
        config_path = os.path.join(self.temp_dir, "test_train_multi.jsonc")
        with open(config_path, 'w') as f:
            json.dump({"model_name": "test"}, f)
            
        response = self.client.post("/train", json={
            "config_file": config_path,
            "multi_gpu": True,
            "nproc_per_node": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert "torchrun" in data["command"]
        assert "--nproc_per_node=2" in data["command"]


class GenerationEndpointTests:
    """Test generation endpoints."""
    
    def __init__(self, client, temp_dir: str, test_data: dict):
        self.client = client
        self.temp_dir = temp_dir
        self.test_data = test_data
        
    def test_direct_generation_manual_mode(self):
        """Test direct generation in manual mode."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": os.path.join(self.temp_dir, "gen_out.parquet"),
            "manual": True,
            "compositions": "Si2O4,Ti2O4"
        })
        assert response.status_code == 200
        data = response.json()
        assert "_load_and_generate" in data["command"]
        assert "--manual" in data["command"]
        assert "--compositions" in data["command"]
        
    def test_direct_generation_with_conditions(self):
        """Test direct generation with condition values."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_density",
            "output_parquet": os.path.join(self.temp_dir, "gen_cond.parquet"),
            "manual": True,
            "compositions": "Si1",
            "condition_lists": ["1.1,0.0"],
            "level": "level_2"
        })
        assert response.status_code == 200
        data = response.json()
        assert "--condition_lists" in data["command"]
        
    def test_direct_generation_with_xrd_csv_files(self):
        """Test direct generation with XRD CSV files."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_COD-XRD",
            "output_parquet": os.path.join(self.temp_dir, "gen_xrd.parquet"),
            "manual": True,
            "compositions": "Ti2O4,Ti4O8",
            "xrd_csv_files": ["/data/anatase.csv", "/data/rutile.csv"],
            "mode": "paired",
            "level": "level_2"
        })
        assert response.status_code == 200
        data = response.json()
        assert "--xrd_csv_files" in data["command"]
        assert "/data/anatase.csv" in data["command"]
        assert "/data/rutile.csv" in data["command"]
        assert "--mode paired" in data["command"]
        
    def test_direct_generation_with_spacegroups(self):
        """Test direct generation with spacegroups (level_4)."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": os.path.join(self.temp_dir, "gen_sg.parquet"),
            "manual": True,
            "compositions": "Ti2O4",
            "condition_lists": ["1.8,0.0"],
            "spacegroups": "P4_2/mnm",
            "level": "level_4"
        })
        assert response.status_code == 200
        data = response.json()
        assert "--spacegroups" in data["command"]
        assert "P4_2/mnm" in data["command"]
        assert "--level level_4" in data["command"]
        
    def test_direct_generation_with_all_params(self):
        """Test direct generation with all generation parameters."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": os.path.join(self.temp_dir, "gen_full.parquet"),
            "manual": True,
            "compositions": "Si1",
            "condition_lists": ["1.1,0.0"],
            "level": "level_2",
            "mode": "cartesian",
            "do_sample": "True",
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.8,
            "gen_max_length": 512,
            "num_return_sequences": 5,
            "max_return_attempts": 10,
            "scoring_mode": "None",
            "model_type": "PKV",
            "num_workers": 2,
            "verbose": True
        })
        assert response.status_code == 200
        data = response.json()
        # check key params are in command
        assert "--num_return_sequences 5" in data["command"]
        assert "--max_return_attempts 10" in data["command"]
        assert "--temperature 0.8" in data["command"]
        assert "--top_k 20" in data["command"]
        assert "--model_type PKV" in data["command"]
        
    def test_direct_generation_input_parquet_mode(self):
        """Test direct generation with input parquet."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": os.path.join(self.temp_dir, "gen_parquet.parquet"),
            "manual": False,
            "input_parquet": self.test_data['test_file']
        })
        assert response.status_code == 200
        data = response.json()
        assert "--input_parquet" in data["command"]
        
    def test_make_prompts_manual(self):
        """Test make-prompts in manual mode."""
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": os.path.join(self.temp_dir, "prompts.parquet"),
            "manual": True,
            "compositions": "Li1Fe1PO4,Na1Mn1O2",
            "level": "level_3"
        })
        assert response.status_code == 200
        data = response.json()
        assert "make_prompts" in data["command"]
        assert "--level" in data["command"]
        assert "level_3" in data["command"]
        
    def test_make_prompts_automatic(self):
        """Test make-prompts in automatic mode."""
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": os.path.join(self.temp_dir, "prompts_auto.parquet"),
            "manual": False,
            "automatic": True,
            "HF_dataset": "c-bone/SLME_1K_dtest",
            "split": "test",
            "level": "level_2"
        })
        assert response.status_code == 200
        data = response.json()
        assert "--automatic" in data["command"]
        assert "--HF_dataset" in data["command"]
        
    def test_generate_cifs(self):
        """Test CIF generation with config."""
        config = {"checkpoint_path": "/some/path"}
        config_path = os.path.join(self.temp_dir, "gen_config.jsonc")
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        response = self.client.post("/generate/cifs", json={
            "config_file": config_path
        })
        assert response.status_code == 200
        data = response.json()
        assert "generate_CIFs" in data["command"]
        
    def test_evaluate_cifs(self):
        """Test CIF evaluation endpoint."""
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": self.test_data['test_file'],
            "num_workers": 4,
            "save_valid_parquet": os.path.join(self.temp_dir, "valid_cifs.parquet")
        })
        assert response.status_code == 200
        data = response.json()
        assert "evaluate_CIFs" in data["command"]
        assert "--save_valid_parquet" in data["command"]
        
    def test_postprocess(self):
        """Test postprocessing endpoint."""
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "postproc.parquet"),
            "num_workers": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert "postprocess" in data["command"]


class MetricsEndpointTests:
    """Test metrics endpoints."""
    
    def __init__(self, client, temp_dir: str, test_data: dict):
        self.client = client
        self.temp_dir = temp_dir
        self.test_data = test_data
        
    def test_vun_metrics(self):
        """Test VUN metrics endpoint."""
        response = self.client.post("/metrics/vun", json={
            "gen_data": self.test_data['test_file'],
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": os.path.join(self.temp_dir, "vun.csv"),
            "num_workers": 4
        })
        assert response.status_code == 200
        data = response.json()
        assert "VUN_metrics" in data["command"]
        assert "--huggingface_dataset" in data["command"]
        
    def test_ehull_metrics(self):
        """Test energy above hull metrics endpoint."""
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": self.test_data['test_file'],
            "output_parquet": os.path.join(self.temp_dir, "ehull.parquet"),
            "num_workers": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert "mace_ehull" in data["command"]


# ==================== COMMAND CONSTRUCTION TESTS ====================

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
        assert '--property_columns [\"Bandgap\"]' in cmd or "--property_columns [\"Bandgap\"]" in cmd
        
    def test_direct_generation_condition_lists_format(self):
        """Verify condition_lists are passed correctly to CLI."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_slme",
            "output_parquet": "/out.parquet",
            "manual": True,
            "compositions": "Ag16O16",
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
            "manual": True,
            "compositions": "Ti2O4",
            "condition_lists": ["1.82,0.0"],
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "num_return_sequences": 5,
            "max_return_attempts": 10,
            "temperature": 0.9,
            "model_type": "PKV"
        })
        cmd = response.json()["command"]
        
        # all params should be present
        assert "--spacegroups P4_2/mnm" in cmd
        assert "--level level_4" in cmd
        assert "--num_return_sequences 5" in cmd
        assert "--max_return_attempts 10" in cmd
        assert "--temperature 0.9" in cmd
        assert "--model_type PKV" in cmd
        
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


# ==================== API GAPS AND ISSUES ====================

class APIGapTests:
    """Test for known gaps and issues in the API."""
    
    def __init__(self, client, temp_dir: str):
        self.client = client
        self.temp_dir = temp_dir
        
    def test_direct_generation_all_parameters_now_supported(self):
        """Verify all generation params work in API (spacegroups, temp, model_type, etc)."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": "/out.parquet",
            "manual": True,
            "compositions": "Ti2O4",
            "condition_lists": ["1.8,0.0"],
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "num_return_sequences": 5,
            "max_return_attempts": 10,
            "temperature": 1.0,
            "top_k": 15,
            "top_p": 0.95,
            "model_type": "PKV",
            "mode": "cartesian"
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        # all params should now be in command
        assert "--spacegroups P4_2/mnm" in cmd, "spacegroups should be in command"
        assert "--num_return_sequences 5" in cmd, "num_return_sequences should be in command"
        assert "--max_return_attempts 10" in cmd, "max_return_attempts should be in command"
        assert "--temperature 1.0" in cmd, "temperature should be in command"
        assert "--model_type PKV" in cmd, "model_type should be in command"
    
    def test_xrd_generation_now_supported(self):
        """XRD generation is now supported via xrd_csv_files parameter."""
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_COD-XRD",
            "output_parquet": "/out_xrd.parquet",
            "manual": True,
            "compositions": "TiO2",
            "xrd_csv_files": ["/data/xrd_pattern.csv"],
            "mode": "broadcast"
        })
        assert response.status_code == 200
        cmd = response.json()["command"]
        assert "--xrd_csv_files" in cmd, "xrd_csv_files should be in command"
        assert "/data/xrd_pattern.csv" in cmd, "XRD file path should be in command"
        
    def test_missing_xrd_preprocessing_endpoint(self):
        """XRD calculation preprocessing exists in CLI but not API.
        
        Script: _utils/_preprocessing/_xrd.py
        This should be exposed as /preprocessing/xrd
        """
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["preprocessing"]
        assert "/preprocessing/xrd" not in endpoints, "XRD endpoint is missing (expected)"
        
    def test_missing_xrd_metrics_endpoint(self):
        """XRD metrics exist in CLI but not API.
        
        Script: _utils/_metrics/XRD_metrics.py
        This should be exposed as /metrics/xrd
        """
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        assert "/metrics/xrd" not in endpoints, "XRD metrics endpoint is missing (expected)"
        
    def test_missing_property_metrics_endpoint(self):
        """Property metrics exist in CLI but not API.
        
        Script: _utils/_metrics/property_metrics.py (ALIGNN predictions)
        This should be exposed as /metrics/property
        """
        response = self.client.get("/")
        endpoints = response.json()["endpoints"]["metrics"]
        # only vun and ehull exist
        assert len([e for e in endpoints if "property" in e]) == 0, "Property metrics endpoint missing (expected)"


# ==================== INTEGRATION TESTS ====================

class IntegrationTests:
    """Integration tests that run actual commands and wait for completion.
    
    These tests actually execute the full pipeline against a Docker container,
    waiting for jobs to complete and verifying outputs.
    """
    
    def __init__(self, client, temp_dir: str, test_data: dict, docker_mode: bool = False, verbose: bool = False):
        self.client = client
        self.temp_dir = temp_dir
        self.test_data = test_data
        self.docker_mode = docker_mode
        self.verbose = verbose
        # Use Docker paths when in docker mode
        self.data_dir = "/app/data" if docker_mode else temp_dir
        self.output_dir = "/app/outputs" if docker_mode else temp_dir
        # Local paths for reading outputs (Docker mounts outputs/ to /app/outputs)
        self.local_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        
        # Create test data in Docker-mounted directory if in docker mode
        if docker_mode:
            self._setup_docker_test_data()
    
    def _setup_docker_test_data(self):
        """Create test_input.parquet in the local data/ directory (mounted to /app/data)."""
        # Local path that maps to /app/data in Docker
        local_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(local_data_dir, exist_ok=True)
        
        test_file = os.path.join(local_data_dir, "test_input.parquet")
        if os.path.exists(test_file):
            print(f"      Test data already exists: {test_file}")
            return
            
        print(f"      Creating test data: {test_file}")
        test_df = self.test_data['test_df'].copy()
        test_df.to_parquet(test_file, index=False)
        print(f"      Created test_input.parquet with {len(test_df)} rows")
        
    def _get_local_path(self, docker_path: str) -> str:
        """Convert Docker path to local path for reading outputs."""
        if docker_path.startswith("/app/outputs"):
            return docker_path.replace("/app/outputs", self.local_output_dir)
        return docker_path
    
    def _show_generated_cifs(self, parquet_path: str, max_cifs: int = 2):
        """Display sample generated CIFs from parquet file."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] File not found: {local_path}")
            return
        
        df = pd.read_parquet(local_path)
        cif_col = "Generated CIF" if "Generated CIF" in df.columns else "CIF"
        
        print(f"\n      {'='*60}")
        print(f"      GENERATED STRUCTURES ({len(df)} total)")
        print(f"      {'='*60}")
        
        if "Reduced Formula" in df.columns:
            print(f"      Compositions: {df['Reduced Formula'].value_counts().to_dict()}")
        
        for i, row in df.head(max_cifs).iterrows():
            cif = row.get(cif_col, "N/A")
            formula = row.get("Reduced Formula", "Unknown")
            print(f"\n      --- Sample CIF {i+1}: {formula} ---")
            # show first 30 lines of CIF
            cif_lines = cif.split('\n')[:30] if isinstance(cif, str) else ["N/A"]
            for line in cif_lines:
                print(f"      {line}")
            total_lines = len(cif.split('\n')) if isinstance(cif, str) else 0
            if total_lines > 30:
                print(f"      ... ({total_lines} lines total)")
        print()
    
    def _show_validity_stats(self, parquet_path: str):
        """Display validity statistics from processed parquet."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] File not found: {local_path}")
            return
        
        df = pd.read_parquet(local_path)
        
        print(f"\n      {'='*60}")
        print(f"      VALIDITY STATISTICS")
        print(f"      {'='*60}")
        print(f"      Total structures: {len(df)}")
        
        if "is_valid" in df.columns:
            valid_count = df["is_valid"].sum()
            print(f"      Valid: {valid_count} ({100*valid_count/len(df):.1f}%)")
        if "is_unique" in df.columns:
            unique_count = df["is_unique"].sum()
            print(f"      Unique: {unique_count} ({100*unique_count/len(df):.1f}%)")
        if "is_novel" in df.columns:
            novel_count = df["is_novel"].sum()
            print(f"      Novel: {novel_count} ({100*novel_count/len(df):.1f}%)")
        print()
    
    def _show_vun_metrics(self, csv_path: str):
        """Display VUN metrics from CSV output."""
        local_path = self._get_local_path(csv_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] VUN CSV not found: {local_path}")
            return
        
        df = pd.read_csv(local_path)
        
        print(f"\n      {'='*60}")
        print(f"      VUN METRICS SUMMARY")
        print(f"      {'='*60}")
        print(df.to_string(index=False))
        print()
    
    def _show_ehull_stats(self, parquet_path: str):
        """Display E-hull statistics from parquet file."""
        local_path = self._get_local_path(parquet_path)
        if not os.path.exists(local_path):
            print(f"      [verbose] E-hull file not found: {local_path}")
            return
        
        df = pd.read_parquet(local_path)
        
        print(f"\n      {'='*60}")
        print(f"      E-HULL STATISTICS")
        print(f"      {'='*60}")
        
        ehull_col = "ehull_mace_mp" if "ehull_mace_mp" in df.columns else None
        
        if ehull_col:
            ehull_vals = df[ehull_col].dropna()
            print(f"      Total structures: {len(df)}")
            print(f"      Structures with E-hull: {len(ehull_vals)}")
            if len(ehull_vals) > 0:
                print(f"      E-hull range: {ehull_vals.min():.4f} to {ehull_vals.max():.4f} eV/atom")
                print(f"      E-hull mean: {ehull_vals.mean():.4f} eV/atom")
                print(f"      E-hull median: {ehull_vals.median():.4f} eV/atom")
                stable_count = (ehull_vals < 0.1).sum()
                print(f"      Stable (E-hull < 0.1 eV/atom): {stable_count} ({100*stable_count/len(ehull_vals):.1f}%)")
                
            # Show a few examples
            print(f"\n      Sample E-hull values:")
            for i, row in df.head(5).iterrows():
                formula = row.get("Reduced Formula", row.get("formula", "Unknown"))
                ehull = row.get(ehull_col, "N/A")
                print(f"        {formula}: {ehull:.4f} eV/atom" if isinstance(ehull, float) else f"        {formula}: {ehull}")
        else:
            print(f"      Available columns: {list(df.columns)}")
            print(f"      No E-hull column found")
        print()
        
    def wait_for_job(self, job_id: str, timeout: int = 300, job_name: str = "") -> dict:
        """Wait for a job to complete."""
        print(f"      Waiting for job {job_name or job_id}...", end="", flush=True)
        start = time.time()
        last_status = ""
        while time.time() - start < timeout:
            response = self.client.get(f"/jobs/{job_id}")
            job = response.json()
            if job["status"] != last_status:
                print(f" [{job['status']}]", end="", flush=True)
                last_status = job["status"]
            if job["status"] in ["completed", "failed"]:
                elapsed = time.time() - start
                print(f" ({elapsed:.1f}s)")
                return job
            time.sleep(3)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

    # ==================== PREPROCESSING TESTS ====================
    
    def test_preprocessing_deduplicate(self):
        """Test /preprocessing/deduplicate with real execution."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/preprocessing/deduplicate", json={
            "input_parquet": f"{self.data_dir}/test_input.parquet",
            "output_parquet": f"{self.output_dir}/test_dedup.parquet",
            "property_columns": "[\"Bandgap (eV)\"]",
            "filter_na_columns": "[\"Bandgap (eV)\"]"
        })
        assert response.status_code == 200, f"Request failed: {response.status_code}"
        job = self.wait_for_job(response.json()["job_id"], job_name="deduplicate")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    def test_preprocessing_clean(self):
        """Test /preprocessing/clean with real execution."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/preprocessing/clean", json={
            "input_parquet": f"{self.data_dir}/test_input.parquet",
            "output_parquet": f"{self.output_dir}/test_clean.parquet",
            "num_workers": 4,
            "property_columns": "[\"Bandgap (eV)\"]",
            "property1_normaliser": "power_log"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="clean")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    def test_preprocessing_save_dataset(self):
        """Test /preprocessing/save-dataset with real execution (local only)."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/preprocessing/save-dataset", json={
            "input_parquet": f"{self.data_dir}/test_input.parquet",
            "output_parquet": f"{self.output_dir}/test_dataset_api",
            "test_size": 0.0,  # no splits - test data only has 3 rows
            "valid_size": 0.0,
            "HF_username": "c-bone",
            "save_hub": False,  # don't actually push to HF
            "save_local": True
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="save-dataset")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    # ==================== GENERATION TESTS ====================
    
    def test_generate_base_unconditional(self):
        """Test unconditional generation with base model (no conditions)."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_base",
            "output_parquet": f"{self.output_dir}/base_generated.parquet",
            "manual": True,
            "compositions": "Ti2O4",
            "spacegroups": "P4_2/mnm",
            "level": "level_4",
            "num_return_sequences": 2,
            "max_return_attempts": 1,
            "model_type": "Base"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="generate_base")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
        if self.verbose:
            self._show_generated_cifs(f"{self.output_dir}/base_generated.parquet")
    
    def test_generate_direct(self):
        """Test /generate/direct - conditional PKV model generation."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_slme",
            "output_parquet": f"{self.output_dir}/generated.parquet",
            "manual": True,
            "compositions": "Ag16O16",
            "condition_lists": ["24.7"],
            "level": "level_2",
            "num_return_sequences": 2,
            "max_return_attempts": 1,
            "model_type": "PKV"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="generate_direct")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
        if self.verbose:
            self._show_generated_cifs(f"{self.output_dir}/generated.parquet")
        
    def test_generate_make_prompts(self):
        """Test /generate/make-prompts - create prompts for generation."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        response = self.client.post("/generate/make-prompts", json={
            "output_parquet": f"{self.output_dir}/prompts.parquet",
            "manual": True,
            "compositions": "Li1Fe1PO4,Na1Mn1O2,Si1O2",
            "condition_lists": ["0.5,0.0"],
            "level": "level_2"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="make_prompts")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    def test_generate_evaluate_cifs(self):
        """Test /generate/evaluate-cifs - evaluate structural validity."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        # requires generated.parquet from previous test
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": f"{self.output_dir}/generated.parquet",
            "num_workers": 4,
            "save_valid_parquet": f"{self.output_dir}/valid_cifs.parquet"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="evaluate_cifs")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    def test_generate_postprocess(self):
        """Test /generate/postprocess - post-process CIFs to standard format."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        # requires generated.parquet from previous test
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": f"{self.output_dir}/generated.parquet",
            "output_parquet": f"{self.output_dir}/postprocessed.parquet",
            "num_workers": 4
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="postprocess")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
    # ==================== METRICS TESTS ====================
    
    def test_metrics_vun(self):
        """Test /metrics/vun - Validity, Uniqueness, Novelty metrics."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        # requires postprocessed.parquet from previous test
        response = self.client.post("/metrics/vun", json={
            "gen_data": f"{self.output_dir}/postprocessed.parquet",
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": f"{self.output_dir}/vun_metrics.csv",
            "num_workers": 4
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="vun_metrics")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
        if self.verbose:
            self._show_vun_metrics(f"{self.output_dir}/vun_metrics.csv")
        
    def test_metrics_ehull(self):
        """Test /metrics/ehull - Energy above hull (thermodynamic stability)."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        # requires postprocessed.parquet from previous test
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": f"{self.output_dir}/postprocessed.parquet",
            "output_parquet": f"{self.output_dir}/ehull_results.parquet",
            "num_workers": 2
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="ehull_metrics")
        assert job["status"] == "completed", f"Job failed: {job.get('error')}"
        
        if self.verbose:
            self._show_ehull_stats(f"{self.output_dir}/ehull_results.parquet")
        
    # ==================== FULL PIPELINE TEST ====================
    
    def test_full_generation_pipeline(self):
        """Test the complete generation -> evaluate -> postprocess -> metrics pipeline."""
        if not self.docker_mode:
            print("    Skipping - requires Docker environment")
            return
            
        print("\n      === FULL PIPELINE TEST ===")
        
        # Step 1: Generate structures
        print("      Step 1/5: Generating structures...")
        response = self.client.post("/generate/direct", json={
            "hf_model_path": "c-bone/CrystaLLM-pi_bandgap",
            "output_parquet": f"{self.output_dir}/pipeline_generated.parquet",
            "manual": True,
            "compositions": "Si1,Ge1,Ga1As1",
            "condition_lists": ["1.5,0.0"],
            "level": "level_2",
            "num_return_sequences": 3,
            "max_return_attempts": 2,
            "model_type": "PKV"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=900, job_name="generate")
        assert job["status"] == "completed", f"Generation failed: {job.get('error')}"
        
        if self.verbose:
            self._show_generated_cifs(f"{self.output_dir}/pipeline_generated.parquet", max_cifs=3)
        
        # Step 2: Evaluate CIFs
        print("      Step 2/5: Evaluating structures...")
        response = self.client.post("/generate/evaluate-cifs", json={
            "input_parquet": f"{self.output_dir}/pipeline_generated.parquet",
            "num_workers": 4,
            "save_valid_parquet": f"{self.output_dir}/pipeline_valid.parquet"
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="evaluate")
        assert job["status"] == "completed", f"Evaluation failed: {job.get('error')}"
        
        # Step 3: Postprocess
        print("      Step 3/5: Postprocessing...")
        response = self.client.post("/generate/postprocess", json={
            "input_parquet": f"{self.output_dir}/pipeline_generated.parquet",
            "output_parquet": f"{self.output_dir}/pipeline_postprocessed.parquet",
            "num_workers": 4
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], job_name="postprocess")
        assert job["status"] == "completed", f"Postprocess failed: {job.get('error')}"
        
        # Step 4: VUN metrics
        print("      Step 4/5: Computing VUN metrics...")
        response = self.client.post("/metrics/vun", json={
            "gen_data": f"{self.output_dir}/pipeline_postprocessed.parquet",
            "huggingface_dataset": "c-bone/SLME_1K_dtest",
            "output_csv": f"{self.output_dir}/pipeline_vun.csv",
            "num_workers": 4
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="vun")
        assert job["status"] == "completed", f"VUN metrics failed: {job.get('error')}"
        
        if self.verbose:
            self._show_vun_metrics(f"{self.output_dir}/pipeline_vun.csv")
            self._show_validity_stats(f"{self.output_dir}/pipeline_postprocessed.parquet")
        
        # Step 5: E-hull metrics
        print("      Step 5/5: Computing E-hull metrics...")
        response = self.client.post("/metrics/ehull", json={
            "post_parquet": f"{self.output_dir}/pipeline_postprocessed.parquet",
            "output_parquet": f"{self.output_dir}/pipeline_ehull.parquet",
            "num_workers": 2
        })
        assert response.status_code == 200
        job = self.wait_for_job(response.json()["job_id"], timeout=600, job_name="ehull")
        assert job["status"] == "completed", f"E-hull metrics failed: {job.get('error')}"
        
        if self.verbose:
            self._show_ehull_stats(f"{self.output_dir}/pipeline_ehull.parquet")
        
        print("      === PIPELINE COMPLETE ===")


def run_all_tests(suite: APITestSuite, run_integration: bool = False, verbose: bool = False):
    """Run all test categories."""
    test_data = suite.create_test_data()
    
    # Root endpoint tests
    root_tests = RootEndpointTests(suite.client, suite.temp_dir)
    suite.run_test("root_returns_api_info", root_tests.test_root_returns_api_info)
    suite.run_test("root_lists_all_endpoint_categories", root_tests.test_root_lists_all_endpoint_categories)
    
    # Job management tests
    job_tests = JobManagementTests(suite.client, suite.temp_dir)
    suite.run_test("list_jobs_empty", job_tests.test_list_jobs_empty)
    suite.run_test("get_nonexistent_job_returns_404", job_tests.test_get_nonexistent_job_returns_404)
    suite.run_test("job_creation_returns_pending_status", job_tests.test_job_creation_returns_pending_status)
    suite.run_test("job_can_be_retrieved_after_creation", job_tests.test_job_can_be_retrieved_after_creation)
    
    # Preprocessing endpoint tests
    preproc_tests = PreprocessingEndpointTests(suite.client, suite.temp_dir, test_data)
    suite.run_test("deduplicate_valid_request", preproc_tests.test_deduplicate_valid_request)
    suite.run_test("deduplicate_with_all_optional_params", preproc_tests.test_deduplicate_with_all_optional_params)
    suite.run_test("deduplicate_missing_required_field", preproc_tests.test_deduplicate_missing_required_field)
    suite.run_test("clean_valid_request", preproc_tests.test_clean_valid_request)
    suite.run_test("clean_with_normalizers", preproc_tests.test_clean_with_normalizers)
    suite.run_test("clean_invalid_normalizer", preproc_tests.test_clean_invalid_normalizer)
    suite.run_test("save_dataset_valid_request", preproc_tests.test_save_dataset_valid_request)
    suite.run_test("save_dataset_with_splits", preproc_tests.test_save_dataset_with_splits)
    
    # Training endpoint tests
    train_tests = TrainingEndpointTests(suite.client, suite.temp_dir)
    suite.run_test("train_single_gpu", train_tests.test_train_single_gpu)
    suite.run_test("train_multi_gpu", train_tests.test_train_multi_gpu)
    
    # Generation endpoint tests
    gen_tests = GenerationEndpointTests(suite.client, suite.temp_dir, test_data)
    suite.run_test("direct_generation_manual_mode", gen_tests.test_direct_generation_manual_mode)
    suite.run_test("direct_generation_with_conditions", gen_tests.test_direct_generation_with_conditions)
    suite.run_test("direct_generation_with_xrd_csv_files", gen_tests.test_direct_generation_with_xrd_csv_files)
    suite.run_test("direct_generation_with_spacegroups", gen_tests.test_direct_generation_with_spacegroups)
    suite.run_test("direct_generation_with_all_params", gen_tests.test_direct_generation_with_all_params)
    suite.run_test("direct_generation_input_parquet_mode", gen_tests.test_direct_generation_input_parquet_mode)
    suite.run_test("make_prompts_manual", gen_tests.test_make_prompts_manual)
    suite.run_test("make_prompts_automatic", gen_tests.test_make_prompts_automatic)
    suite.run_test("generate_cifs", gen_tests.test_generate_cifs)
    suite.run_test("evaluate_cifs", gen_tests.test_evaluate_cifs)
    suite.run_test("postprocess", gen_tests.test_postprocess)
    
    # Metrics endpoint tests
    metrics_tests = MetricsEndpointTests(suite.client, suite.temp_dir, test_data)
    suite.run_test("vun_metrics", metrics_tests.test_vun_metrics)
    suite.run_test("ehull_metrics", metrics_tests.test_ehull_metrics)
    
    # Command construction tests
    cmd_tests = CommandConstructionTests(suite.client, suite.temp_dir)
    suite.run_test("deduplicate_command_structure", cmd_tests.test_deduplicate_command_structure)
    suite.run_test("direct_generation_condition_lists_format", cmd_tests.test_direct_generation_condition_lists_format)
    suite.run_test("direct_generation_all_params_in_command", cmd_tests.test_direct_generation_all_params_in_command)
    suite.run_test("train_torchrun_format", cmd_tests.test_train_torchrun_format)
    
    # API gap tests (document missing features)
    gap_tests = APIGapTests(suite.client, suite.temp_dir)
    suite.run_test("direct_generation_all_parameters_now_supported", gap_tests.test_direct_generation_all_parameters_now_supported)
    suite.run_test("xrd_generation_now_supported", gap_tests.test_xrd_generation_now_supported)
    suite.run_test("missing_xrd_preprocessing_endpoint", gap_tests.test_missing_xrd_preprocessing_endpoint)
    suite.run_test("missing_xrd_metrics_endpoint", gap_tests.test_missing_xrd_metrics_endpoint)
    suite.run_test("missing_property_metrics_endpoint", gap_tests.test_missing_property_metrics_endpoint)
    
    # Integration tests (optional, slower)
    if run_integration:
        print("\n" + "="*60)
        print("INTEGRATION TESTS (Docker execution)")
        print("="*60)
        
        int_tests = IntegrationTests(suite.client, suite.temp_dir, test_data, 
                                     docker_mode=suite.docker_url is not None, verbose=verbose)
        
        print("\n--- Preprocessing Tests ---")
        suite.run_test("integration_deduplicate", int_tests.test_preprocessing_deduplicate)
        suite.run_test("integration_clean", int_tests.test_preprocessing_clean)
        suite.run_test("integration_save_dataset", int_tests.test_preprocessing_save_dataset)
        
        print("\n--- Generation Tests ---")
        suite.run_test("integration_generate_base", int_tests.test_generate_base_unconditional)
        suite.run_test("integration_generate_direct", int_tests.test_generate_direct)
        suite.run_test("integration_make_prompts", int_tests.test_generate_make_prompts)
        suite.run_test("integration_evaluate_cifs", int_tests.test_generate_evaluate_cifs)
        suite.run_test("integration_postprocess", int_tests.test_generate_postprocess)
        
        print("\n--- Metrics Tests ---")
        suite.run_test("integration_vun_metrics", int_tests.test_metrics_vun)
        suite.run_test("integration_ehull_metrics", int_tests.test_metrics_ehull)
        
        print("\n--- Full Pipeline Test ---")
        suite.run_test("integration_full_pipeline", int_tests.test_full_generation_pipeline)


def main():
    """Run the API test suite."""
    parser = argparse.ArgumentParser(description="CrystaLLM-2.0 API Test Suite")
    parser.add_argument("--hf_key", type=str, default="", help="HuggingFace API key (not needed for Docker testing)")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API key (not needed for Docker testing)")
    parser.add_argument("--docker_url", type=str, default=None, 
                        help="URL of running Docker container (e.g., http://localhost:8000)")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests (slower, requires actual execution)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output: sample CIFs, validity stats, E-hull values")
    args = parser.parse_args()
    
    print("="*60)
    print("CrystaLLM-2.0 API Test Suite")
    print("="*60)
    
    if args.docker_url:
        print(f"Testing against Docker: {args.docker_url}")
    else:
        if not args.hf_key or not args.wandb_key:
            print("Warning: --hf_key and --wandb_key not provided.")
            print("For local testing, these are recommended.")
            print("For Docker testing, use --docker_url instead.")
        print("Testing locally with FastAPI TestClient")
    
    if args.verbose:
        print("Verbose mode: Will show sample outputs from integration tests")
    
    suite = APITestSuite(args.hf_key, args.wandb_key, args.docker_url)
    suite.setup()
    
    try:
        run_all_tests(suite, run_integration=args.integration, verbose=args.verbose)
    except Exception as e:
        print(f"\nFatal error during tests: {e}")
        import traceback
        traceback.print_exc()
    finally:
        suite.cleanup()
    
    return suite.report_results()


if __name__ == "__main__":
    exit(main())
