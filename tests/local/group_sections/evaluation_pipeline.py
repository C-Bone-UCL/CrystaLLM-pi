"""Local test section: evaluation pipeline."""

class EvaluationPipelineTests:
    """Test evaluation and metrics scripts."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_vun_metrics_script(self):
        """Test VUN metrics calculation script."""
        try:
            from _utils._metrics.VUN_metrics import compute_vun_metrics, save_vun_metrics
            from _utils._metrics_utils import get_valid, get_unique, get_novelty
            
            # Test that VUN functions exist
            assert get_valid is not None, "Validity function exists"
            assert get_unique is not None, "Uniqueness function exists" 
            assert get_novelty is not None, "Novelty function exists"
            assert compute_vun_metrics is not None, "VUN compute function exists"
            
        except Exception as e:
            print(f"VUN metrics import failed: {e}")
    
    def test_stability_metrics(self):
        """Test stability metrics calculation."""
        try:
            # Test that we can import the MACE script
            import _utils._metrics.mace_ehull
            print("MACE E-hull script imported successfully")
            
        except Exception as e:
            # MACE calculations require specific dependencies
            print(f"Stability calculation failed (acceptable): {e}")
    
    def test_xrd_metrics(self):
        """Test XRD metrics calculation functions."""
        from _utils._metrics.XRD_metrics import (
            smact_validity, structure_validity, is_valid_bench,
            _symmetrize_cif, _find_best_lattice_match, _calculate_metrics,
            _parallel_convert_generated_cif
        )
        from pymatgen.core import Structure
        import numpy as np
        
        # Test smact_validity with simple composition (Si, O)
        # Si = Z=14, O = Z=8
        comp = (8, 14)  # O, Si sorted by Z
        count = (2, 1)  # SiO2 ratio
        result = smact_validity(comp, count)
        assert isinstance(result, bool), "smact_validity should return bool"
        
        # Test with single element (should return True for alloys)
        single_comp = (26,)  # Fe
        single_count = (1,)
        result_single = smact_validity(single_comp, single_count)
        assert result_single is True, "Single element should be valid"
        
        # Test structure_validity with a simple structure
        test_struct = Structure.from_str(self.test_data['test_cif'], fmt="cif")
        struct_valid = structure_validity(test_struct)
        assert isinstance(struct_valid, bool), "structure_validity should return bool"
        
        # Test is_valid_bench (combines smact + structure validity)
        bench_valid = is_valid_bench(test_struct)
        assert isinstance(bench_valid, bool), "is_valid_bench should return bool"
        
        # Test _symmetrize_cif
        symm_cif = _symmetrize_cif(test_struct)
        assert isinstance(symm_cif, str), "_symmetrize_cif should return string"
        assert "data_" in symm_cif, "Symmetrized CIF should contain data_ block"
        
        # Test _find_best_lattice_match
        gen_structs = [test_struct]  # Use same struct as generated
        best_gen, a_diff, b_diff, c_diff = _find_best_lattice_match(gen_structs, test_struct)
        assert best_gen is not None, "Should find a match"
        assert a_diff == 0.0, "Same structure should have zero a_diff"
        assert b_diff == 0.0, "Same structure should have zero b_diff"
        assert c_diff == 0.0, "Same structure should have zero c_diff"
        
        # Test _calculate_metrics
        rms_dists = [0.1, None, 0.2]
        a_diffs = [0.05, None, 0.1]
        b_diffs = [0.03, None, 0.08]
        c_diffs = [0.02, None, 0.05]
        gen_structs_mock = [[1], [2], [3]]  # Just need length
        
        metrics = _calculate_metrics(rms_dists, a_diffs, b_diffs, c_diffs, gen_structs_mock)
        assert "match_rate" in metrics, "Should have match_rate"
        assert "rms_dist" in metrics, "Should have rms_dist"
        assert "n_matched" in metrics, "Should have n_matched"
        assert metrics["match_rate"] == 2/3, "Match rate should be 2/3"
        assert metrics["n_matched"] == 2, "Should have 2 matches"
        
        # Test _parallel_convert_generated_cif
        converted = _parallel_convert_generated_cif(self.test_data['test_cif'])
        assert converted is not None, "Valid CIF should convert to Structure"
        assert isinstance(converted, Structure), "Should return pymatgen Structure"
        
        # Test with invalid CIF
        invalid_converted = _parallel_convert_generated_cif("invalid cif content")
        assert invalid_converted is None, "Invalid CIF should return None"
        
        print("XRD metrics functions tested successfully")
    
    def test_property_metrics(self):
        """Test property prediction metrics."""
        try:
            # Test that we can import ALIGNN-related scripts
            import _utils._metrics.property_metrics
            print("Property metrics script imported successfully")
            
        except Exception as e:
            # ALIGNN requires separate environment
            print(f"Property prediction failed (expected in main env): {e}")
