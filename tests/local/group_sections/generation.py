"""Local test section: generation."""

import pandas as pd

class GenerationTests:
    """Test generation pipeline components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_generation_basic(self):
        """Test basic generation utilities."""
        from _utils._generating.generate_CIFs import init_tokenizer, setup_device, build_generation_kwargs
        
        # Test tokenizer init
        tokenizer = init_tokenizer("HF-cif-tokenizer")
        assert tokenizer is not None, "Tokenizer initialization failed"
        assert tokenizer.pad_token is not None, "Pad token should be set"
        
        # Test device setup (CPU fallback)
        device = setup_device(0)
        assert device is not None, "Device setup failed"
        assert device.type in ['cuda', 'cpu'], "Device type should be cuda or cpu"
        
        # Test generation kwargs - create mock args
        class MockArgs:
            def __init__(self):
                self.do_sample = True
                self.top_k = 15
                self.temperature = 1.0
                self.top_p = 0.95
                self.num_beams = 1
                self.num_return_sequences = 1
                self.gen_max_length = 512
                self.repetition_penalty = 1.0
                self.length_penalty = 1.0
                
        args = MockArgs()
        kwargs = build_generation_kwargs(args, tokenizer, 512)
        assert 'do_sample' in kwargs, "Generation kwargs missing do_sample"
        assert 'max_length' in kwargs, "Generation kwargs missing max_length"
        assert 'pad_token_id' in kwargs, "Generation kwargs missing pad_token_id"
    
    def test_generation_conditional(self):
        """Test conditional generation setup."""
        from _utils._generating.generate_CIFs import parse_condition_vector
        
        # Test condition parsing with comma-separated values
        condition_str = "0.5,0.3"
        parsed = parse_condition_vector(condition_str)
        assert len(parsed) == 2, "Condition parsing failed"
        assert abs(parsed[0] - 0.5) < 1e-6, "Condition value incorrect"
        assert abs(parsed[1] - 0.3) < 1e-6, "Second condition value incorrect"
        
        # Test single value
        single = parse_condition_vector("1.5")
        assert len(single) == 1, "Single value parsing failed"
        assert abs(single[0] - 1.5) < 1e-6, "Single value incorrect"
        
        # Test None handling
        assert parse_condition_vector(None) is None, "None should return None"
        assert parse_condition_vector("None") is None, "String 'None' should return None"
    
    def test_check_cif(self):
        """Test CIF validation function."""
        from _utils._generating.generate_CIFs import check_cif
        
        # Test with valid CIF (from test data)
        valid_cif = self.test_data['test_cif']
        result = check_cif(valid_cif)
        assert isinstance(result, bool), "check_cif should return boolean"
        
        # Test with invalid/malformed CIF
        invalid_cif = "data_invalid\n_cell_length_a garbage\nrandom text"
        result_invalid = check_cif(invalid_cif)
        assert result_invalid is False, "Invalid CIF should return False"
        
        # Test with empty string
        assert check_cif("") is False, "Empty string should return False"
        
        # Test exception handling
        assert check_cif(None) is False, "None should be handled gracefully"
    
    def test_get_model_class(self):
        """Test model class selection."""
        from _utils._generating.generate_CIFs import get_model_class
        from _models import PKVGPT, PrependGPT, SliderGPT
        from transformers import GPT2LMHeadModel
        
        # Test each conditionality type
        assert get_model_class("PKV") == PKVGPT, "PKV should return PKVGPT"
        assert get_model_class("Prepend") == PrependGPT, "Prepend should return PrependGPT"
        assert get_model_class("Slider") == SliderGPT, "Slider should return SliderGPT"
        
        # Test default/unconditional cases
        assert get_model_class(None) == GPT2LMHeadModel, "None should return GPT2LMHeadModel"
        assert get_model_class("Raw") == GPT2LMHeadModel, "Raw should return GPT2LMHeadModel"
        assert get_model_class("unconditional") == GPT2LMHeadModel, "Unknown type should return GPT2LMHeadModel"
    
    def test_build_generation_kwargs_modes(self):
        """Test build_generation_kwargs with different sampling modes."""
        from _utils._generating.generate_CIFs import init_tokenizer, build_generation_kwargs
        
        tokenizer = init_tokenizer("HF-cif-tokenizer")
        
        class MockArgs:
            def __init__(self, do_sample):
                self.do_sample = do_sample
                self.top_k = 15
                self.temperature = 0.8
                self.top_p = 0.95
                self.num_return_sequences = 5
                self.gen_max_length = 512
        
        # Test sampling mode (do_sample=True)
        args_sample = MockArgs(do_sample=True)
        kwargs_sample = build_generation_kwargs(args_sample, tokenizer, 1024)
        assert kwargs_sample['do_sample'] is True, "do_sample should be True"
        assert kwargs_sample['top_k'] == 15, "top_k should be 15"
        assert kwargs_sample['temperature'] == 0.8, "temperature should be 0.8"
        assert kwargs_sample['num_return_sequences'] == 5, "num_return_sequences should be 5"
        
        # Test greedy mode (do_sample=False)
        args_greedy = MockArgs(do_sample=False)
        kwargs_greedy = build_generation_kwargs(args_greedy, tokenizer, 1024)
        assert kwargs_greedy['do_sample'] is False, "do_sample should be False"
        assert kwargs_greedy['num_return_sequences'] == 1, "Greedy should have 1 sequence"
        assert kwargs_greedy['top_k'] == 0, "Greedy should have top_k=0"
        
        # Test beam search mode
        args_beam = MockArgs(do_sample="beam")
        kwargs_beam = build_generation_kwargs(args_beam, tokenizer, 1024)
        assert kwargs_beam['do_sample'] is False, "Beam should not sample"
        assert kwargs_beam['num_beams'] == 5, "num_beams should match num_return_sequences"
        
        # Test max_length capping
        args_long = MockArgs(do_sample=True)
        args_long.gen_max_length = 2048
        kwargs_capped = build_generation_kwargs(args_long, tokenizer, 1024)
        assert kwargs_capped['max_length'] == 1024, "max_length should be capped to model max"
    
    def test_remove_conditionality(self):
        """Test removal of conditioning comments from CIF."""
        from _utils._generating.generate_CIFs import remove_conditionality
        
        # Test with comments before data_ block
        cif_with_comments = """# Bandgap: 2.5 eV
# Density: 3.2 g/cm3
data_Si1O2
_cell_length_a 5.0
loop_
 _atom_site_label
  Si0"""
        result = remove_conditionality(cif_with_comments)
        assert result.startswith("data_"), "Should start with data_"
        assert "Bandgap" not in result, "Comments should be removed"
        
        # Test with no comments
        cif_clean = "data_Ti1O2\n_cell_length_a 4.5"
        result_clean = remove_conditionality(cif_clean)
        assert result_clean == cif_clean, "Clean CIF should be unchanged"
        
        # Test with no data_ block (edge case)
        no_data = "# Just comments\n_cell_length_a 5.0"
        result_no_data = remove_conditionality(no_data)
        assert result_no_data == no_data, "No data_ block should return original"
    
    def test_get_material_id(self):
        """Test material ID extraction/generation."""
        from _utils._generating.generate_CIFs import get_material_id
        
        # Test with Material ID in row - now expects unique counter suffix
        row_with_id = pd.Series({"Material ID": "mp-1234", "Formula": "Si1O2"})
        assert get_material_id(row_with_id, 0) == "mp-1234_1", "Should use Material ID with counter"
        
        # Test with only Formula - now expects unique counter suffix
        row_formula = pd.Series({"Formula": "Ti1O2"})
        assert get_material_id(row_formula, 0) == "Ti1O2_1", "Should use Formula with counter"
        
        # Test with neither - should generate ID
        row_empty = pd.Series({"Prompt": "test"})
        generated_id = get_material_id(row_empty, 5, offset=10)
        assert generated_id == "Generated_16", "Should generate ID with count+offset+1"
        
        # Test count and offset
        assert get_material_id(row_empty, 0, offset=0) == "Generated_1"
        assert get_material_id(row_empty, 2, offset=5) == "Generated_8"
    
    def test_build_output_df(self):
        """Test output dataframe construction."""
        from _utils._generating.generate_CIFs import build_output_df
        
        # Create mock generated data
        generated_data = [
            {"Material ID": "mp-1", "Prompt": "test", "Generated CIF": "data_1", "condition_vector": "0.5"},
            {"Material ID": "mp-2", "Prompt": "test2", "Generated CIF": "data_2", "condition_vector": "0.6"},
        ]
        
        class MockArgs:
            def __init__(self):
                self.input_parquet = None
        
        # Test without True CIF merge
        df_prompts = pd.DataFrame({"Material ID": ["mp-1", "mp-2"], "Prompt": ["test", "test2"]})
        result = build_output_df(generated_data, MockArgs(), df_prompts)
        assert len(result) == 2, "Should have 2 rows"
        assert "Generated CIF" in result.columns, "Should have Generated CIF column"
        
        # Test with True CIF merge
        class MockArgsWithInput:
            def __init__(self):
                self.input_parquet = "test.parquet"
        
        df_prompts_with_cif = pd.DataFrame({
            "Material ID": ["mp-1", "mp-2"],
            "Prompt": ["test", "test2"],
            "True CIF": ["true_1", "true_2"]
        })
        result_merged = build_output_df(generated_data, MockArgsWithInput(), df_prompts_with_cif)
        assert "True CIF" in result_merged.columns, "Should have True CIF after merge"