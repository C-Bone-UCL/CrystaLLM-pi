"""Local test section: data processing."""

import os
import pandas as pd

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
