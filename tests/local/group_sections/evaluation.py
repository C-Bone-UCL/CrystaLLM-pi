"""Local test section: evaluation."""

import pandas as pd

class EvaluationTests:
    """Test evaluation and metrics components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_vun_metrics(self):
        """Test VUN metrics calculation."""
        from _utils._metrics_utils import is_valid

        valid_result = is_valid(self.test_data['test_cif'])
        assert isinstance(valid_result, bool), "is_valid should return a boolean"
        assert valid_result is True, "Known-good fixture CIF should validate"

        malformed = "data_invalid\n_invalid_field 123\ngarbage"
        try:
            invalid_result = is_valid(malformed)
        except Exception:
            # Invalid CIFs are allowed to raise in current metrics utility behavior.
            invalid_result = False
        assert invalid_result is False, "Invalid CIF should never validate"
    
    def test_validity_function(self):
        """Test is_valid function with various CIF inputs."""
        from _utils._metrics_utils import is_valid
        
        # Test with structurally valid CIF
        valid_result = is_valid(self.test_data['test_cif'], bond_length_acceptability_cutoff=0.5)
        assert isinstance(valid_result, bool), "is_valid should return boolean"
        assert valid_result is True, "Known-good fixture CIF should validate with cutoff"

        # Test with garbage and malformed input
        try:
            result_garbage = is_valid("random garbage text")
        except Exception:
            result_garbage = False
        assert result_garbage is False, "Garbage should be invalid"

        try:
            result_malformed = is_valid("data_\n_cell_length_a not_a_number")
        except Exception:
            result_malformed = False
        assert result_malformed is False, "Malformed CIF should be invalid"
    
    def test_uniqueness_function(self):
        """Test get_unique function for structure deduplication."""
        from _utils._metrics_utils import get_unique
        
        # Create dataframe with duplicates (same CIF twice)
        df_gen = pd.DataFrame({
            "Generated CIF": [self.test_data['test_cif'], self.test_data['test_cif']]
        })
        
        df_gen["is_valid"] = True
        df_out = get_unique(df_gen, workers=1)
        assert "is_unique" in df_out.columns, "Expected is_unique column after uniqueness pass"
        assert int(df_out["is_unique"].sum()) == 1, "Duplicate CIF rows should reduce to one unique row"
    
    def test_novelty_function(self):
        """Test get_novelty function for comparing against training set."""
        from _utils._metrics_utils import get_novelty
        
        # Generated CIFs dataframe
        df_gen = pd.DataFrame({
            "CIF": [self.test_data['test_cif']]
        })
        # Mock training set compositions
        base_comps = set()  # Empty for simplicity
        
        df_gen["is_unique"] = True
        structures = [None]
        df_out = get_novelty(
            df_gen,
            base_comps,
            ltol=0.2,
            stol=0.3,
            angle_tol=5,
            structures=structures,
            workers=1,
        )
        assert "is_novel" in df_out.columns, "Expected is_novel column after novelty pass"
        assert bool(df_out["is_novel"].iloc[0]) is False, "Rows without a parsed structure should not be novel"
    
    def test_density_calculation(self):
        """Test density calculation from CIF."""
        from _utils._metrics_utils import get_density

        density = get_density(self.test_data['test_cif'])
        assert density == density, "Density should not be NaN for valid test CIF"
        assert isinstance(density, (int, float)), "Density should be numeric"
        assert density > 0, "Density should be positive"

    def test_formula_consistency_partial_occupancy(self):
        """Test formula consistency with valid and invalid partial occupancy CIFs."""
        from _utils._metrics_utils import is_formula_consistent

        valid_result = is_formula_consistent(self.test_data['partial_occ_valid_cif'])
        mismatched_formula = self.test_data['test_cif'].replace(
            "_chemical_formula_sum   'Si4 O8'",
            "_chemical_formula_sum   'Si2 O3'",
        ).replace(
            "_chemical_formula_structural   SiO2",
            "_chemical_formula_structural   Si2O3",
        )
        invalid_result = is_formula_consistent(mismatched_formula)

        assert valid_result is True, "Valid partial-occupancy CIF should be consistent"
        assert invalid_result is False, "Mismatched formulas should be inconsistent"
    
    def test_basic_evaluation(self):
        """Test basic evaluation pipeline components."""
        from _utils import extract_volume
        from _utils._metrics_utils import is_valid, get_density

        assert callable(extract_volume), "extract_volume should be callable"
        assert callable(is_valid), "is_valid should be callable"
        assert callable(get_density), "get_density should be callable"
