"""Core local test harness utilities."""

"""
Comprehensive automated test suite for CrystaLLM-pi pipeline components.
Tests all major functionality including data processing, training, generation, 
and evaluation pipelines without creating permanent files.

Covers all scripts mentioned in README:
- Data processing: deduplication, cleaning, XRD calculation, HF formatting
- Training: model initialization, conditional model forward passes
- Generation: CIF generation, evaluation, postprocessing  
- Evaluation: VUN metrics, stability calculation, property prediction
- Dataloader: collator, round-robin packing, conditional mode
- HF integration: model loading, direct generation

Usage:
    python tests/local/suite.py          # Run on GPU if available, else CPU
    python tests/local/suite.py --cpu    # Force CPU execution
    python tests/local/suite.py --gpu    # Force GPU execution
    python run-tests.py                  # Compatibility wrapper
"""

import os
import sys
import argparse
import tempfile
import shutil
import traceback
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from tests.fixtures.shared_cif_fixtures import (
    TEST_CIF_SIO2,
    AUGMENTED_CIF_SIO2,
    PARTIAL_OCC_VALID_CIF,
    PARTIAL_OCC_INVALID_CIF,
)
from tests.fixtures.test_data_fixtures import build_test_dataframe, get_fixture_xrd_csv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Global device setting - will be set by main()
DEVICE = None

def get_device():
    """Get the device to use for tests."""
    global DEVICE
    return DEVICE if DEVICE is not None else torch.device("cpu")

class TestSuite:
    """Main test coordinator with temporary file management."""
    
    def __init__(self):
        self.temp_dir = None
        self.results = {}
        self.verbose = True
        
    def setup(self):
        """Create temporary test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="crystallm_test_")
        if self.verbose:
            print(f"Test environment: {self.temp_dir}")
            print(f"Device: {get_device()}")
        
    def cleanup(self):
        """Remove all test files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"Cleaned up: {self.temp_dir}")
    
    def run_test(self, test_name, test_func):
        """Run individual test with error handling."""
        try:
            test_func()
            self.results[test_name] = "PASS"
            if self.verbose:
                print(f"✓ {test_name}")
        except Exception as e:
            self.results[test_name] = f"FAIL: {str(e)}"
            if self.verbose:
                print(f"✗ {test_name}: {str(e)}")
                if self.verbose:
                    traceback.print_exc()
    
    def create_test_data(self):
        """Create minimal test CIF data."""
        test_cif = TEST_CIF_SIO2
        augmented_cif = AUGMENTED_CIF_SIO2

        partial_occ_valid_cif = PARTIAL_OCC_VALID_CIF
        partial_occ_invalid_cif = PARTIAL_OCC_INVALID_CIF

        test_df = build_test_dataframe(test_cif, include_al2o3_row=False)
        
        test_file = os.path.join(self.temp_dir, "test_data.parquet")
        test_df.to_parquet(test_file, index=False)
        
        return {
            'test_cif': test_cif,
            'augmented_cif': augmented_cif,
            'partial_occ_valid_cif': partial_occ_valid_cif,
            'partial_occ_invalid_cif': partial_occ_invalid_cif,
            'test_file': test_file,
            'test_df': test_df,
            'fixture_xrd_csv': get_fixture_xrd_csv(project_root),
        }
        
    def report_results(self):
        """Print final test results."""
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        passed = sum(1 for result in self.results.values() if result == "PASS")
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "✓" if result == "PASS" else "✗"
            print(f"{status} {test_name}: {result}")
        
        print(f"\nPassed: {passed}/{total} tests")
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"❌ {total - passed} test(s) failed")
            return False

