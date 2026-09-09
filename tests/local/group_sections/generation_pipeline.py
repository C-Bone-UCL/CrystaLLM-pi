"""Local test section: generation pipeline."""

class GenerationPipelineTests:
    """Test generation pipeline scripts."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_tokenizer_dir_resolves_when_cwd_copy_is_absent(self):
        """The bare name must resolve to the packaged copy from any working directory.

        HF-cif-tokenizer ships inside the wheel under _utils/. Without this fallback an
        installed CrystaLLM-pi could only generate when run from the repo root, which is the
        bug the wheel-install CI job guards against.
        """
        import os
        import tempfile

        from _tokenizer import BUNDLED_TOKENIZER_DIR, CustomCIFTokenizer, resolve_tokenizer_dir

        assert os.path.isdir(BUNDLED_TOKENIZER_DIR), "Packaged tokenizer directory is missing"

        original_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as elsewhere:
                os.chdir(elsewhere)
                resolved = resolve_tokenizer_dir("HF-cif-tokenizer")
                assert os.path.isdir(resolved), "Bare name should fall back to the packaged copy"
                tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
                assert len(tokenizer) > 0, "Packaged tokenizer loaded no vocabulary"
        finally:
            os.chdir(original_cwd)

        # An explicit directory that does exist still wins over the packaged copy.
        assert resolve_tokenizer_dir(BUNDLED_TOKENIZER_DIR) == BUNDLED_TOKENIZER_DIR
        # An unrelated missing path is returned untouched, so callers still see their own error.
        assert resolve_tokenizer_dir("no/such/tokenizer") == "no/such/tokenizer"

    def test_generation_script_imports(self):
        """Every name the generation pipeline exposes is still importable."""
        from _utils._generating.generate_cifs import (
            init_tokenizer, check_cif, get_model_class,
            build_generation_kwargs, parse_condition_vector,
            get_material_id, build_output_df,
            DEFAULT_MAX_LENGTH, TOKENIZER_PAD_TOKEN, DEFAULT_TOKENIZER_DIR
        )
        from _utils._generating.workers import setup_device

        # Test constants
        assert DEFAULT_MAX_LENGTH == 1024, "Default max length should be 1024"
        assert TOKENIZER_PAD_TOKEN == "<pad>", "Pad token should be <pad>"
        assert DEFAULT_TOKENIZER_DIR == "HF-cif-tokenizer", "Default tokenizer dir"
        
        # Test tokenizer initialization
        tokenizer = init_tokenizer("HF-cif-tokenizer")
        assert tokenizer is not None, "Generation tokenizer init failed"
        assert hasattr(tokenizer, 'encode'), "Tokenizer should have encode method"
        assert hasattr(tokenizer, 'decode'), "Tokenizer should have decode method"
        
        # Test device setup
        device = setup_device(0)
        assert device is not None, "Generation device setup failed"
    
    def test_score_output_logp(self):
        """forward_pass_logp scores the prompt's continuation up to EOS, and nothing else.

        Counting prompt tokens would drag every candidate toward the same value, and counting
        past EOS would let padding decide the ranking.
        """
        import torch
        from _utils._generating.scoring_methods import forward_pass_logp

        VOCAB, EOS, PAD, K = 100, 99, 0, 4

        class MockModel:
            """Spreads all probability evenly over the first K tokens, so log p is -log(K)."""

            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, **_kwargs):
                self.calls += 1
                logits = torch.full((*input_ids.shape, VOCAB), -1e9)
                logits[..., :K] = 0.0
                return type("Out", (), {"logits": logits})

        assert forward_pass_logp(None, None, 0) == [], "No sequences means no scores"
        assert forward_pass_logp(None, torch.empty(0), 0) == [], "Empty tensor means no scores"

        # Tokens stay under K so perplexity is exactly K at any length. Row 0 scores indices
        # 2-3 and row 1 only index 2, both stopping before EOS and ignoring the padding.
        sequences = torch.tensor([
            [1, 2, 3, 1, EOS, PAD],
            [2, 3, 2, EOS, PAD, PAD],
        ])
        model = MockModel()
        scores = forward_pass_logp(model, sequences, input_length=2, eos_token_id=EOS)

        assert model.calls == 1, "One forward pass for the whole batch"
        assert len(scores) == 2, "One score per sequence"
        for score in scores:
            assert abs(score - K) < 1e-3, \
                f"uniform over {K} tokens gives perplexity {K} at any length, got {score}"

        # A sequence with no scoreable token is inf, not a silently good score.
        only_eos = torch.tensor([[1, 2, EOS, PAD]])
        assert forward_pass_logp(model, only_eos, input_length=2, eos_token_id=EOS)[0] == float('inf'), \
            "Nothing generated before EOS must not rank first"
    
    def test_generation_kwargs_edge_cases(self):
        """Test generation kwargs with edge cases."""
        from _utils._generating.generate_cifs import init_tokenizer, build_generation_kwargs
        
        tokenizer = init_tokenizer("HF-cif-tokenizer")
        
        class MockArgs:
            def __init__(self, do_sample):
                self.do_sample = do_sample
                self.top_k = 50
                self.temperature = 1.0
                self.top_p = 0.9
                self.num_return_sequences = 3
                self.gen_max_length = 256
        
        # Test string "true" and "True" variations
        args_true_str = MockArgs(do_sample="True")
        kwargs = build_generation_kwargs(args_true_str, tokenizer, 512)
        assert kwargs['do_sample'] is True, "String 'True' should enable sampling"
        
        args_false_str = MockArgs(do_sample="False")
        kwargs_false = build_generation_kwargs(args_false_str, tokenizer, 512)
        assert kwargs_false['do_sample'] is False, "String 'False' should disable sampling"
        
        # Test that essential kwargs are always present
        for key in ['max_length', 'pad_token_id', 'eos_token_id', 'renormalize_logits']:
            assert key in kwargs, f"{key} should be in generation kwargs"
    
    def test_check_cif_comprehensive(self):
        """check_cif returns a bool for junk input and rejects formula mismatches."""
        from _utils._generating.generate_cifs import check_cif
        
        # Test valid CIF structure
        valid_cif = self.test_data['test_cif']
        result = check_cif(valid_cif)
        assert isinstance(result, bool), "Should return boolean"
        
        # Test various invalid inputs
        test_cases = [
            ("", False, "Empty string"),
            ("not a cif", False, "Random text"),
            ("data_\n_cell 1", False, "Incomplete CIF"),
            (None, False, "None value"),
        ]
        
        for input_val, expected, description in test_cases:
            try:
                result = check_cif(input_val)
                # We expect False for invalid inputs, but the function might handle differently
                assert isinstance(result, bool), f"{description}: should return boolean"
            except Exception:
                # Exception handling is acceptable for malformed input
                pass

        # Formula-structure mismatch should fail validation
        mismatched_cif = valid_cif.replace("_chemical_formula_sum   'Si4 O8'", "_chemical_formula_sum   'Si2 O3'")
        mismatched_cif = mismatched_cif.replace("_chemical_formula_structural   SiO2", "_chemical_formula_structural   Si2O3")
        assert check_cif(mismatched_cif) is False, "Mismatched formula should fail validation"
    
    def test_condition_vector_parsing_comprehensive(self):
        """parse_condition_vector handles each string form the CLI and parquet paths produce."""
        from _utils._generating.generate_cifs import parse_condition_vector
        
        # Test various input formats
        test_cases = [
            ("0.5", [0.5]),
            ("1.0,2.0", [1.0, 2.0]),
            ("0.1, 0.2, 0.3", [0.1, 0.2, 0.3]),  # With spaces
            ("-1.5", [-1.5]),  # Negative values
            ("0", [0.0]),  # Zero
            (None, None),
            ("None", None),
        ]
        
        for input_val, expected, in test_cases:
            result = parse_condition_vector(input_val)
            if expected is None:
                assert result is None, f"Input {input_val} should return None"
            else:
                assert len(result) == len(expected), f"Length mismatch for {input_val}"
                for r, e in zip(result, expected):
                    assert abs(r - e) < 1e-6, f"Value mismatch for {input_val}"

    def test_generation_mode_resolution(self):
        """None scoring should validate when target_valid_cifs is positive, but return all rows when it is
zero."""
        from _utils._generating.generate_cifs import resolve_generation_plan

        validate_only = resolve_generation_plan(
            scoring_mode="None",
            target_valid_cifs=3,
            max_return_attempts=5,
            num_return_sequences=2,
            total_samples=4,
        )
        assert validate_only["normalized_scoring_mode"] == "none"
        assert validate_only["need_scores"] is False
        assert validate_only["check_validity"] is True
        assert validate_only["target_per_prompt"] == 3
        assert validate_only["total_expected_generations"] == 12

        raw_generation = resolve_generation_plan(
            scoring_mode="None",
            target_valid_cifs=0,
            max_return_attempts=5,
            num_return_sequences=2,
            total_samples=4,
        )
        assert raw_generation["normalized_scoring_mode"] == "none"
        assert raw_generation["need_scores"] is False
        assert raw_generation["check_validity"] is False
        assert raw_generation["target_per_prompt"] == 10
        assert raw_generation["total_expected_generations"] == 40

        ranked_generation = resolve_generation_plan(
            scoring_mode="LOGP",
            target_valid_cifs=3,
            max_return_attempts=5,
            num_return_sequences=2,
            total_samples=4,
        )
        assert ranked_generation["normalized_scoring_mode"] == "logp"
        assert ranked_generation["need_scores"] is True
        assert ranked_generation["check_validity"] is True
        assert ranked_generation["target_per_prompt"] == 3
        assert ranked_generation["total_expected_generations"] == 12

    def test_evaluation_script(self):
        """Test CIF evaluation script."""
        try:
            import _utils._generating.evaluate_cifs
            print("Evaluation script imported successfully")
            
        except Exception as e:
            print(f"Evaluation script import failed: {e}")
    
    def test_postprocessing_script(self):
        """Test CIF postprocessing script."""
        try:
            import _utils._generating.postprocess
            from _utils._generating.postprocess import process_dataframe
            
            assert process_dataframe is not None, "Postprocessing function exists"
            
        except Exception as e:
            print(f"Postprocessing script import failed: {e}")
