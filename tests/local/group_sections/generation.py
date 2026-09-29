"""Local test section: generation."""

import pandas as pd

class GenerationTests:
    """Test generation pipeline components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_generation_basic(self):
        """Test basic generation utilities."""
        from _utils._generating.generate_cifs import init_tokenizer, build_generation_kwargs
        from _utils._generating.workers import setup_device
        
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
        from _utils._generating.generate_cifs import parse_condition_vector
        
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
        from _utils._generating.generate_cifs import check_cif
        
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
    
    def test_screening_profiles(self):
        """Only the two profile names are accepted, and this repo defaults to application."""
        import sys
        from _args import parse_args

        def profile(argv):
            saved, sys.argv = sys.argv, ["generate"] + argv
            try:
                return parse_args().screening_profile
            finally:
                sys.argv = saved

        assert profile([]) == "application", "The zoo screens at full strength by default"
        assert profile(["--screening_profile", "benchmark"]) == "benchmark"

    def test_formula_consistency_tolerates_partial_occupancy(self):
        """Fractional occupancies must survive screening, virtualiser output above all.

        `reduced_composition` divides out only an integer factor, so this used to fail on scale.
        """
        from _utils.validity import is_formula_consistent

        assert is_formula_consistent(self.test_data['partial_occ_valid_cif']) is True, \
            "Disordered CIF must pass the formula-consistency check"

    def test_formula_consistency_catches_ratio_mismatch(self):
        """Tolerating fractional occupancies must not weaken the check itself."""
        from _utils.validity import _compositions_match
        from pymatgen.core import Composition

        # Scale must cancel, both for supercells and for partial occupancy.
        assert _compositions_match(Composition({"Si": 1, "O": 2}),
                                   Composition({"Si": 4, "O": 8})) is True
        assert _compositions_match(Composition({"Hf": 1, "Ta": 1, "Mo": 1, "W": 1}),
                                   Composition({"Hf": .5, "Ta": .5, "Mo": .5, "W": .5})) is True
        # Ratio mismatches must fail. The first is a real CHILI case the old 0.1 accepted.
        assert _compositions_match(Composition({"Fe": 12, "C": 4}),
                                   Composition({"Fe": 2, "C": 1})) is False
        assert _compositions_match(Composition({"Eu": 3, "Au": 1, "O": 6}),
                                   Composition({"Eu": 2, "Au": 1, "O": 3})) is False
        assert _compositions_match(Composition({"Hf": 1, "Ta": 1, "Mo": 1, "W": 1}),
                                   Composition({"Hf": .5, "Ta": .5, "Mo": .5, "W": .25})) is False

    def test_get_model_class(self):
        """Test strict model class selection."""
        from _utils._generating.generate_cifs import get_model_class
        from _models import PKVGPT, SliderGPT, PrefixGPT, ResidualGPT
        from transformers import GPT2LMHeadModel

        # Test each conditionality type
        assert get_model_class("PKV") == PKVGPT, "PKV should return PKVGPT"
        assert get_model_class("Slider") == SliderGPT, "Slider should return SliderGPT"
        assert get_model_class("Prefix") == PrefixGPT, "Prefix should return PrefixGPT"
        assert get_model_class("Residual") == ResidualGPT, "Residual should return ResidualGPT"

        # Base/None aliases still map to plain GPT2
        assert get_model_class(None) == GPT2LMHeadModel, "None should return GPT2LMHeadModel"
        assert get_model_class("Base") == GPT2LMHeadModel, "Base should return GPT2LMHeadModel"

        # Unknown names now raise instead of silently falling back to GPT2
        try:
            get_model_class("unconditional")
            assert False, "Unknown model type should raise ValueError"
        except ValueError as err:
            assert "Unknown model type" in str(err), f"Unexpected error: {err}"

    def test_parse_condition_vector_nested(self):
        """Nested condition vectors (continuous XRD) survive parsing, and flat strings stay unchanged."""
        from _utils._generating.generate_cifs import parse_condition_vector

        # Nested string form (parquet round-trip) and native nested lists preserve 2D shape
        assert parse_condition_vector("[[0.0, 0.1], [0.01, 0.2]]") == [[0.0, 0.1], [0.01, 0.2]]
        profile = [[round(0.01 * i, 2), 0.5] for i in range(1000)]
        parsed = parse_condition_vector(profile)
        assert len(parsed) == 1000 and parsed == profile

        # Legacy flat forms are unchanged
        assert parse_condition_vector("1.0, 2.0") == [1.0, 2.0]
        assert parse_condition_vector("0.5") == [0.5]
        assert parse_condition_vector(None) is None
        assert parse_condition_vector("None") is None

    def test_build_generation_kwargs_modes(self):
        """Test build_generation_kwargs with different sampling modes."""
        from _utils._generating.generate_cifs import init_tokenizer, build_generation_kwargs
        
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
    
    def test_get_material_id(self):
        """Test material ID extraction/generation."""
        from _utils._generating.generate_cifs import get_material_id
        
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
        from _utils._generating.generate_cifs import build_output_df
        
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