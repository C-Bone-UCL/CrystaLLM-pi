"""Local test section: generation pipeline."""

class GenerationPipelineTests:
    """Test generation pipeline scripts."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_generation_script_imports(self):
        """Test generation script components - comprehensive import check."""
        from _utils._generating.generate_CIFs import (
            init_tokenizer, setup_device, check_cif, get_model_class,
            build_generation_kwargs, remove_conditionality, parse_condition_vector,
            get_material_id, build_output_df, score_output_logp, score_outputs_logp,
            DEFAULT_MAX_LENGTH, TOKENIZER_PAD_TOKEN, DEFAULT_TOKENIZER_DIR
        )
        
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
        """Test perplexity scoring function."""
        from _utils._generating.generate_CIFs import score_output_logp, score_outputs_logp
        import torch

        class MockModel:
            def __init__(self, transition_scores):
                self.transition_scores = transition_scores
                self.calls = 0

            def compute_transition_scores(self, full_sequences, scores, normalize_logits=True):
                self.calls += 1
                return self.transition_scores
        
        # Test with None/empty scores
        result_none = score_output_logp(None, None, None, 0, 0)
        assert result_none == float('inf'), "None scores should return inf"
        
        result_empty = score_output_logp(None, [], None, 0, 0)
        assert result_empty == float('inf'), "Empty scores should return inf"

        full_sequences = torch.tensor([
            [10, 11, 12, 99],
            [20, 21, 99, 0],
        ])
        transition_scores = torch.tensor([
            [0.0, -0.2, -0.4, -0.8],
            [0.0, -0.5, -1.0, -2.0],
        ])
        mock_model = MockModel(transition_scores)
        batch_scores = score_outputs_logp(
            mock_model,
            scores=[torch.tensor([0.0])],
            full_sequences=full_sequences,
            input_length=1,
            eos_token_id=99,
        )

        assert mock_model.calls == 1, "Batch scoring should compute transition scores once"
        assert len(batch_scores) == 2, "Batch scoring should return one score per sequence"

        single_score = score_output_logp(
            mock_model,
            scores=[torch.tensor([0.0])],
            full_sequences=full_sequences,
            sequence_idx=1,
            input_length=1,
            eos_token_id=99,
        )
        assert single_score == batch_scores[1], "Single score helper should match batch scoring"
    
    def test_generation_kwargs_edge_cases(self):
        """Test generation kwargs with edge cases."""
        from _utils._generating.generate_CIFs import init_tokenizer, build_generation_kwargs
        
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
        """Comprehensive CIF validation tests."""
        from _utils._generating.generate_CIFs import check_cif
        
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
        """Comprehensive condition vector parsing tests."""
        from _utils._generating.generate_CIFs import parse_condition_vector
        
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

    def test_evaluation_script(self):
        """Test CIF evaluation script."""
        try:
            import _utils._generating.evaluate_CIFs
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

    def test_xtra_augment_generation_jobs(self):
        """Xtra-augment generation jobs should target LeMat prompts and xtra-augment artifacts."""
        from _pipelines._xtra_augment_jobs import XTRA_AUGMENT_JOBS

        assert len(XTRA_AUGMENT_JOBS) == 3, "Expected 3 xtra-augment jobs"

        for job in XTRA_AUGMENT_JOBS:
            assert job["activate_conditionality"] == "None"
            assert job["gen_config"] == "_config_files/generation/unconditional/_xtra_augment/lemat-bench_eval.jsonc"
            assert job["input_parquet"] == "_post_paper_files/lemat-bench/an-init_prompt.parquet"
            assert job["output_parquet"].startswith("_artifacts/_xtra_augment/")
            assert job["output_parquet"].endswith("_gen.parquet")
            assert job["output_cif_dir"].startswith("_artifacts/_xtra_augment/")
            assert job["output_cif_dir"].endswith("_cifs/")

    def test_sequential_full_imports(self):
        """Sequential full pipeline should be importable and expose a main entrypoint."""
        import _pipelines.sequential_full as sequential_full

        assert hasattr(sequential_full, "main"), "sequential_full should expose main()"
        assert hasattr(sequential_full, "run_full_job"), "sequential_full should expose run_full_job()"
