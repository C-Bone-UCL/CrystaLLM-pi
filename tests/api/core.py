"""Core API suite harness and HTTP client wrappers."""

"""
API test suite for CrystaLLM-pi FastAPI endpoints.
Tests all REST API functionality including preprocessing, training, generation, 
and evaluation metrics endpoints. Uses FastAPI's TestClient for synchronous testing.

Usage:
    # Test against running Docker container (recommended)
    conda run -n crystallmv2_venv python tests/api/suite.py --docker_url http://localhost:8000
    
    # Test locally with FastAPI TestClient (requires fastapi installed)
    python tests/api/suite.py --hf_key YOUR_HF_KEY --wandb_key YOUR_WANDB_KEY
    
    # Run integration tests (slower, actually executes commands)
    conda run -n crystallmv2_venv python tests/api/suite.py --docker_url http://localhost:8000 --integration
    
    # Run integration tests with verbose output (shows sample CIFs, VUN stats, E-hull values)
    conda run -n crystallmv2_venv python tests/api/suite.py --docker_url http://localhost:8000 --integration --verbose

    # Compatibility wrapper
    conda run -n crystallmv2_venv python run-tests-api.py --docker_url http://localhost:8000

Docker setup (run this first):
    export HF_KEY="your_hf_token_here"
    export WANDB_KEY="your_wandb_key_here"
    docker compose -f docker/docker-compose.yml up --build -d
    docker compose -f docker/docker-compose.yml logs -f api
"""

import os
import sys
import argparse
import tempfile
import shutil
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

from tests.fixtures.shared_cif_fixtures import (
    TEST_CIF_SIO2,
    PARTIAL_OCC_VALID_CIF,
    PARTIAL_OCC_INVALID_CIF,
)
from tests.fixtures.test_data_fixtures import build_test_dataframe, get_fixture_xrd_csv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


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
        self.error_log_path = None
        self.log_dir = None
        
    def setup(self):
        """Create temporary test environment and initialize test client."""
        self.temp_dir = tempfile.mkdtemp(prefix="crystallm_api_test_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(project_root, "outputs", "api_test_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.error_log_path = os.path.join(self.log_dir, f"api_tests_{timestamp}.log")
        with open(self.error_log_path, "w", encoding="utf-8") as handle:
            handle.write("CrystaLLM API Test Log\n")
            handle.write(f"Started at: {datetime.now().isoformat()}\n")

        if self.verbose:
            print(f"Test directory: {self.temp_dir}")
            print(f"Test log: {self.error_log_path}")
        
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
            from _api import app, jobs
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
            if self.error_log_path:
                with open(self.error_log_path, "a", encoding="utf-8") as handle:
                    handle.write("\n" + "=" * 80 + "\n")
                    handle.write(f"Test: {test_name}\n")
                    handle.write(f"Error: {str(e)}\n")
                    handle.write("Traceback:\n")
                    handle.write(traceback.format_exc())
            
    def create_test_data(self) -> Dict[str, Any]:
        """Create minimal test CIF data."""
        test_cif = TEST_CIF_SIO2
        partial_occ_valid_cif = PARTIAL_OCC_VALID_CIF
        partial_occ_invalid_cif = PARTIAL_OCC_INVALID_CIF

        test_df = build_test_dataframe(test_cif, include_al2o3_row=True)
        
        test_file = os.path.join(self.temp_dir, "test_data.parquet")
        test_df.to_parquet(test_file, index=False)
        
        return {
            'test_cif': test_cif,
            'partial_occ_valid_cif': partial_occ_valid_cif,
            'partial_occ_invalid_cif': partial_occ_invalid_cif,
            'test_file': test_file,
            'test_df': test_df,
            'fixture_xrd_csv': get_fixture_xrd_csv(project_root),
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
        if self.error_log_path:
            print(f"Test log file: {self.error_log_path}")
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

