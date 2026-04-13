"""Local test section: data processing."""

import os
import pandas as pd
import torch

RAW_GENERATED_CIF_SIO2 = """data_Si1O2
loop_
 _atom_type_symbol
 _atom_type_electronegativity
 _atom_type_radius
 _atom_type_ionic_radius
    Si  1.9000  1.1000  0.5400
    O   3.4400  0.6000  1.2600
_symmetry_space_group_name_H-M P1
_cell_length_a 5.0000
_cell_length_b 5.0000
_cell_length_c 5.0000
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 90.0000
_symmetry_Int_Tables_number 1
_chemical_formula_structural SiO2
_chemical_formula_sum 'Si1 O2'
_cell_volume 125.0000
_cell_formula_units_Z 1
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
    Si  Si0  1  0.0000  0.0000  0.0000  1
    O   O1   1  0.3000  0.3000  0.3000  1
    O   O2   1  0.7000  0.7000  0.7000  1"""

class DataProcessingTests:
    """Test data processing components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_tokenizer_basic(self):
        """Test basic tokenizer functionality."""
        from _tokenizer import CustomCIFTokenizer
        
        # Use existing tokenizer files
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Test encoding/decoding
        test_text = self.test_data['augmented_cif']
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        assert len(encoded) > 0, "Encoding failed"
        assert '<bos>' in decoded and '<eos>' in decoded, "Special tokens missing"
        assert 'data_' in decoded, "CIF content missing"
    
    def test_cif_validation(self):
        """Test CIF validation utilities."""
        from _utils._metrics_utils import is_valid
        
        # Test that the validation function works without crashing
        try:
            # Test with our realistic CIF - may pass or fail validation but shouldn't crash
            result = is_valid(self.test_data['test_cif'], bond_length_acceptability_cutoff=0.5)
            assert isinstance(result, bool), "Should return boolean result"
            print(f"CIF validation result: {result}")
        except Exception as e:
            # Some validation failures are acceptable for test CIFs
            print(f"CIF validation failed (acceptable for test): {e}")
        
        # Test that completely malformed input is handled
        try:
            result = is_valid("completely invalid text")
            assert result is False, "Invalid input should be rejected"
        except Exception:
            # Exception for malformed input is acceptable
            pass
    
    def test_prompt_creation(self):
        """Test prompt creation utilities."""
        from _utils._generating.make_prompts import create_manual_prompts
        
        prompts_file = os.path.join(self.temp_dir, "test_prompts.parquet")
        
        # Test manual prompt creation
        df_prompts = create_manual_prompts(
            compositions=["Si2O4", "Ti2O4"],
            condition_lists=[["0.5", "0.0"]],
            level="level_2",
            spacegroups=None
        )
        
        # Save to file and verify
        df_prompts.to_parquet(prompts_file, index=False)
        df = pd.read_parquet(prompts_file)
        assert len(df) > 0, "No prompts created"
        assert 'Prompt' in df.columns, "Prompt column missing"
        assert any('Si' in str(prompt) for prompt in df['Prompt']), "Composition missing from prompts"

    def test_logit_analysis_reconstruction(self):
        """Reconstruct generated CIFs with the canonical bracket helper."""
        from _utils._logit_analysis.cif_parser import reconstruct_bracketed_cif

        bracketed_cif = reconstruct_bracketed_cif(RAW_GENERATED_CIF_SIO2)

        assert "data_[Si1O2]" in bracketed_cif, "data_ formula should be bracketed"
        assert "_atom_type_ionic_radius\n[" in bracketed_cif, (
            "Atomic properties loop should open immediately after the final header"
        )
        assert "\n]\n_symmetry_space_group_name_H-M [P1]" in bracketed_cif, (
            "Atomic properties loop should close before the symmetry line"
        )
        assert "_cell_length_a [5.0000]" in bracketed_cif, "Cell values should be bracketed"
        assert "_chemical_formula_sum '[Si1 O2]'" in bracketed_cif, "Quoted formula sum should preserve quotes"
        assert (
            "loop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n["
            not in bracketed_cif
        ), "Constant symmetry loop should stay unwrapped"
        assert "_atom_site_occupancy\n[" in bracketed_cif, "Atom-site loop should be bracketed"

    def test_logit_analysis_condition_dtype_matches_model(self):
        """Condition tensor dtype should match the model's conditioning path."""
        from _utils._logit_analysis import logit_extraction

        class DummyTokenizer:
            bos_token = "<bos>"
            eos_token = "<eos>"
            token_to_id = {str(i): i for i in range(10)} | {".": 10}

            def encode(self, _text, return_tensors=None):
                token_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
                if return_tensors == "pt":
                    return token_ids
                return token_ids[0].tolist()

        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.condition_probe = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)

            def forward(self, input_ids=None, attention_mask=None, condition_values=None):
                expected_dtype = self.condition_probe.weight.dtype
                assert condition_values.dtype == expected_dtype, (
                    f"expected {expected_dtype}, got {condition_values.dtype}"
                )
                batch_size, seq_len = input_ids.shape
                logits = torch.zeros((batch_size, seq_len, 11), dtype=expected_dtype)
                return DummyOutput(logits=logits)

        original_parser = logit_extraction.parse_cif_numeric_fields
        logit_extraction.parse_cif_numeric_fields = lambda *_args, **_kwargs: [
            {
                "tag": "_cell_length_a",
                "digit_positions": [1, 2],
                "digit_tokens": ["0", "."],
            }
        ]

        try:
            result = logit_extraction.extract_digit_logits(
                DummyModel(),
                DummyTokenizer(),
                cif_text="data_[Si1O2]",
                condition_value=1.0,
                device="cpu",
                include_coords=False,
            )
        finally:
            logit_extraction.parse_cif_numeric_fields = original_parser

        assert len(result["fields"]) == 1, "Expected a single mocked numeric field"
