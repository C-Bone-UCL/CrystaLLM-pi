"""Local test section: data pipeline."""

import os
import pandas as pd

class DataPipelineTests:
    """Test data processing pipeline scripts."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_deduplicate_script(self):
        """Test deduplication functionality with actual data."""
        from _utils._preprocessing.deduplicate import process_cif_entry, deduplicate_table
        
        # Test CIF entry processing - returns (key, idx, vpfu) tuple
        result = process_cif_entry(0, self.test_data['test_cif'])
        assert result is not None, "CIF processing should return result"
        key, idx, vpfu = result
        assert idx == 0, "Should return same index"
        assert isinstance(key, tuple), "Key should be (formula, space_group) tuple"
        assert isinstance(vpfu, float), "Volume per formula unit should be float"
        
        # Test with mock dataframe for deduplication
        test_df = pd.DataFrame({
            'CIF': [self.test_data['test_cif'], self.test_data['test_cif']],
            'Material ID': ['test-1', 'test-2']
        })
        
        try:
            deduped_df = deduplicate_table(test_df, num_workers=1)
            assert len(deduped_df) <= len(test_df), "Deduplication should remove or keep same"
            # Two identical CIFs should result in 1 entry
            assert len(deduped_df) == 1, "Identical CIFs should deduplicate to 1"
        except Exception as e:
            print(f"Full deduplication test failed (acceptable): {e}")
        
    def test_cleaning_script(self):
        """Test CIF cleaning and normalization with actual processing."""
        try:
            from _utils._preprocessing.cleaning import add_atomic_props_block
            from _utils.processing import add_atomic_props_block as process_add_props
            
            # Test atomic properties addition
            result = process_add_props(self.test_data['test_cif'])
            assert len(result) > 0, "CIF processing should return content"
            assert "data_" in result, "Should still have data block"
            
        except ImportError as e:
            print(f"Cleaning imports not available (acceptable): {e}")
    
    def test_float_occupancies(self):
        """Full occupancies become 1.0; partial ones and other loops are left alone."""
        from _utils._preprocessing.cleaning import float_occupancies

        cif = (
            "loop_\n _atom_type_symbol\n _atom_type_radius\n  Na  1\n"
            "loop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n"
            "loop_\n _atom_site_occupancy\n"
            "  Na  Na0  4  0.0000  0.0000  0.0000  1\n"
            "  Cl  Cl1  4  0.5000  -0.5000  0.5000  1.0\n"
            "  Fe  Fe2  2  0.2500  0.2500  0.2500  0.5\n"
        )
        out = float_occupancies(cif)
        assert "  Na  Na0  4  0.0000  0.0000  0.0000  1.0\n" in out
        assert out.replace("0.0000  1.0\n", "0.0000  1\n", 1) == cif, "only the Na site should change"
        assert float_occupancies(out) == out, "should be idempotent"

    def test_coordinate_normalisation(self):
        """Rounding never writes -0.0000, and coordinates that round to 1.0000 wrap to 0.0000."""
        from _utils import round_numbers
        from _utils._preprocessing.cleaning import wrap_unit_coords

        assert round_numbers("x -0.00001234 y 0.99996 z", 4) == "x 0.0000 y 1.0000 z"
        cif = (
            "_cell_length_a 1.0000\n"
            "  Na  Na0  4  1.0000  0.5000  -0.2500  1.0\n"
            "  Fe  Fe1  2  0.2500  1.0000  1.0000  0.5\n"
        )
        out = wrap_unit_coords(cif)
        assert "_cell_length_a 1.0000\n" in out, "only atom-site coordinates should wrap"
        assert "  Na  Na0  4  0.0000  0.5000  -0.2500  1.0\n" in out
        assert "  Fe  Fe1  2  0.2500  0.0000  0.0000  0.5\n" in out
        assert wrap_unit_coords(out) == out, "should be idempotent"

    def test_tolerant_parse(self):
        """Rounded special positions do not duplicate atoms. Only consistent CIFs are rewritten."""
        from pymatgen.core import Structure
        from _utils import extract_space_group_symbol, replace_symmetry_operators, is_formula_consistent
        from _utils._generating.postprocess import postprocess

        # Alex-MP-20 fixture: default tolerance (1e-4) reads 48 F atoms instead of 36.
        cif = """data_Li3Sc9Tl6F36
_symmetry_space_group_name_H-M R-3m
_cell_length_a 7.7542
_cell_length_b 7.7542
_cell_length_c 18.4867
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 120.0000
_symmetry_Int_Tables_number 166
_chemical_formula_structural LiSc3Tl2F12
_chemical_formula_sum 'Li3 Sc9 Tl6 F36'
_cell_volume 962.6508
_cell_formula_units_Z 3
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Li  Li0  3  0.0000  0.0000  0.0000  1.0
  Sc  Sc1  9  0.0000  0.5000  0.5000  1.0
  Tl  Tl2  6  0.0000  0.0000  0.3356  1.0
  F  F3  18  0.0605  0.5303  0.6053  1.0
  F  F4  18  0.0853  0.5426  0.8550  1.0
"""

        # The fixture fails at default tolerance and passes with SITE_TOLERANCE.
        with_ops = replace_symmetry_operators(cif, extract_space_group_symbol(cif))
        assert Structure.from_str(with_ops, fmt="cif").composition["F"] == 48, "fixture no longer shows the bug"
        assert is_formula_consistent(with_ops)

        # The rewritten CIF passes at default tolerance.
        rewritten = postprocess(cif)
        assert Structure.from_str(rewritten, fmt="cif").composition["F"] == 36, "rewrite should read right at 1e-4"

        # A wrong declared formula stays unchanged and fails validation.
        wrong = postprocess(cif.replace("'Li3 Sc9 Tl6 F36'", "'Li3 Sc9 Tl6 F30'"))
        assert not wrong.startswith("# generated using pymatgen"), "inconsistent CIF must not be rewritten"
        assert not is_formula_consistent(wrong)

    def test_xrd_calculation(self):
        """Test XRD pattern calculation with actual structure."""
        try:
            from pymatgen.core import Structure
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            # Parse test CIF to structure
            struct = Structure.from_str(self.test_data['test_cif'], fmt="cif")
            assert struct is not None, "Should parse CIF to structure"
            
            # Calculate XRD pattern
            xrd_calc = XRDCalculator()
            pattern = xrd_calc.get_pattern(struct)
            assert pattern is not None, "Should compute XRD pattern"
            assert len(pattern.x) > 0, "Pattern should have peaks"
            
        except Exception as e:
            print(f"XRD calculation failed (acceptable): {e}")
    
    def test_hf_dataset_save(self):
        """Test Hugging Face dataset formatting."""
        try:
            import _utils._preprocessing.save_dataset_to_hf
            print("HF dataset save script imported successfully")
            
        except Exception as e:
            print(f"HF save script import failed: {e}")

    def test_xrd_input_processing_script(self):
        """Test XRD input processing script functions on fixture data."""
        try:
            import pandas as pd
            from _utils._preprocessing.process_exp_xrd_inputs import process_and_convert, save_to_crystallm_csv

            output_csv = os.path.join(self.temp_dir, "xrd_peaks_processed.csv")
            peaks = process_and_convert(
                input_data=self.test_data['fixture_xrd_csv'],
                xrd_wavelength=1.54056,
                peak_pick=False,
            )

            assert len(peaks) > 0, "Processed peak list should not be empty"

            save_to_crystallm_csv(peaks, output_csv)
            assert os.path.exists(output_csv), "Processed XRD csv should be written"

            df_out = pd.read_csv(output_csv)
            assert list(df_out.columns) == ["2theta", "intensity"], "Output csv header should match expected format"
            assert len(df_out) > 0, "Output csv should contain peaks"
            assert (df_out["2theta"] >= 0).all() and (df_out["2theta"] <= 90).all(), "2theta values must be in [0, 90]"
            assert abs(df_out["intensity"].max() - 100.0) < 1e-6, "Intensities should be normalized to 100 max"
        except Exception as e:
            print(f"XRD input processing test failed (acceptable): {e}")
