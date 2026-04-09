"""Local test section: data utils."""

from datasets import Dataset, DatasetDict

class DataUtilsTests:
    """Test data utility functions from _utils/_data_utils.py."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_filter_long_cifs(self):
        """Test filtering CIFs that exceed context length."""
        from _utils._data_utils import filter_long_CIFs
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Create mock dataset with varying lengths
        short_ids = list(range(50))
        long_ids = list(range(200))
        
        mock_data = {
            "train": Dataset.from_dict({
                "input_ids": [short_ids, long_ids, short_ids]
            })
        }
        dataset = DatasetDict(mock_data)
        
        # Filter with context_length=100
        filtered = filter_long_CIFs(dataset, context_length=100)
        
        # Should keep only entries with length <= 100
        assert len(filtered["train"]) == 2, "Should filter out long entries"
    
    def test_filter_cifs_with_unk(self):
        """Test filtering CIFs with unknown tokens."""
        from _utils._data_utils import filter_CIFs_with_unk
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        unk_id = tokenizer.unk_token_id
        
        # Create mock dataset: one with unk, one without
        mock_data = {
            "train": Dataset.from_dict({
                "input_ids": [
                    [1, 2, 3, 4, 5],  # No unk
                    [1, unk_id, 3, 4, 5],  # Has unk
                    [1, 2, 3, unk_id, 5],  # Has unk
                ]
            })
        }
        dataset = DatasetDict(mock_data)
        
        filtered = filter_CIFs_with_unk(dataset, tokenizer)
        
        assert len(filtered["train"]) == 1, "Should filter entries with unk tokens"
    
    def test_tokenize_function_unconditional(self):
        """Test tokenize_function in unconditional mode."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif'], self.test_data['test_cif']]
        }
        
        result = tokenize_function(examples, tokenizer, mode="unconditional")
        
        assert "input_ids" in result, "Should have input_ids"
        assert "fixed_mask" in result, "Should have fixed_mask"
        assert len(result["input_ids"]) == 2, "Should have 2 tokenized samples"
    
    def test_tokenize_function_conditional(self):
        """Test tokenize_function in conditional mode."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif'], self.test_data['test_cif']],
            "bandgap": [0.5, 0.7],
            "density": [0.3, 0.4]
        }
        
        result = tokenize_function(
            examples, 
            tokenizer, 
            condition_columns="['bandgap', 'density']",
            mode="conditional"
        )
        
        assert "input_ids" in result, "Should have input_ids"
        assert "condition_values" in result, "Should have condition_values"
        assert len(result["condition_values"]) == 2, "Should have 2 condition value sets"
        assert len(result["condition_values"][0]) == 2, "Each should have 2 conditions"
    
    def test_tokenize_function_raw(self):
        """Test tokenize_function in raw mode (text conditioning)."""
        from _utils._data_utils import tokenize_function
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        examples = {
            "CIF": [self.test_data['augmented_cif']],
            "bandgap": [0.5]
        }
        
        result = tokenize_function(
            examples, 
            tokenizer, 
            condition_columns="['bandgap']",
            mode="raw"
        )
        
        assert "input_ids" in result, "Should have input_ids"
        # Raw mode embeds conditions as text, so no separate condition_values
        assert "condition_values" not in result, "Raw mode should not have condition_values"
    
    def test_create_fixed_format_mask(self):
        """Test fixed format mask creation for variable tokens."""
        from _utils._data_utils import create_fixed_format_mask
        from _tokenizer import CustomCIFTokenizer
        
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Text with variable regions in brackets
        text = "data_test\n_cell_length_a [5.0]\n_cell_length_b [4.0]"
        
        mask = create_fixed_format_mask(text, tokenizer, full_length=100)
        
        assert len(mask) > 0, "Mask should not be empty"
        assert 0 in mask, "Should have variable tokens (0s)"
        assert 1 in mask, "Should have fixed tokens (1s)"
    
    def test_parse_condition_value(self):
        """Test condition value parsing helper."""
        from _utils._data_utils import _parse_condition_value
        
        # Test float
        assert _parse_condition_value(0.5) == [0.5]
        
        # Test int
        assert _parse_condition_value(1) == [1.0]
        
        # Test string number
        assert _parse_condition_value("0.7") == [0.7]
        
        # Test string list
        result = _parse_condition_value("[0.5, 0.3]")
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-6
        
        # Test list
        assert _parse_condition_value([0.1, 0.2]) == [0.1, 0.2]
