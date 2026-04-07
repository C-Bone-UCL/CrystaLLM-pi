"""Runner for local CrystaLLM test suites."""

import argparse
import os
import traceback
import torch

from tests.local.core import TestSuite
from tests.local.groups import (
    DataProcessingTests,
    DataLoaderTests,
    DataUtilsTests,
    ModelTests,
    GenerationTests,
    EvaluationTests,
    DataPipelineTests,
    TrainingTests,
    GenerationPipelineTests,
    EvaluationPipelineTests,
    IntegrationTests,
    LoadAndGenerateTests,
    VirtualiserTests,
)

DEVICE = None

def get_device():
    global DEVICE
    return DEVICE if DEVICE is not None else torch.device("cpu")

def main():
    """Run the complete test suite."""
    global DEVICE
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="CrystaLLM-pi Test Suite")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--gpu", action="store_true", help="Force GPU execution")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated substrings to filter tests, e.g. 'dataloader,cleaning'")
    args = parser.parse_args()
    
    # Set device
    if args.cpu:
        DEVICE = torch.device("cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs
        print(f"🖥️  Running tests on CPU (forced via --cpu)")
    elif args.gpu:
        if not torch.cuda.is_available():
            print("ERROR: --gpu specified but CUDA is not available")
            return 1
        DEVICE = torch.device("cuda")
        print(f"🚀 Running tests on GPU (forced via --gpu)")
    else:
        # Auto-detect
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Running tests on {DEVICE.type} (auto-detected)")
    
    suite = TestSuite()
    suite.setup()
    
    try:
        # Create test data
        test_data = suite.create_test_data()
        
        # Initialize test classes
        data_tests = DataProcessingTests(suite.temp_dir, test_data)
        dataloader_tests = DataLoaderTests(suite.temp_dir, test_data)
        data_utils_tests = DataUtilsTests(suite.temp_dir, test_data)
        model_tests = ModelTests(suite.temp_dir, test_data)
        gen_tests = GenerationTests(suite.temp_dir, test_data)
        eval_tests = EvaluationTests(suite.temp_dir, test_data)
        
        # Pipeline test classes
        pipeline_tests = DataPipelineTests(suite.temp_dir, test_data)
        training_tests = TrainingTests(suite.temp_dir, test_data)
        gen_pipeline_tests = GenerationPipelineTests(suite.temp_dir, test_data)
        eval_pipeline_tests = EvaluationPipelineTests(suite.temp_dir, test_data)
        load_gen_tests = LoadAndGenerateTests(suite.temp_dir, test_data)
        integration_tests = IntegrationTests(suite.temp_dir, test_data)
        virtualiser_tests = VirtualiserTests(suite.temp_dir, test_data)
        
        # Build optional name filter
        _only = [s.strip() for s in args.only.split(",")] if args.only else None

        def run(name, fn):
            if _only and not any(f in name for f in _only):
                return
            suite.run_test(name, fn)

        # Execute tests
        print("Running CrystaLLM-pi Comprehensive Test Suite...")
        if _only:
            print(f"Filter: {_only}")
        print("-" * 50)
        
        # Core component tests
        print("\n🔧 Core Component Tests:")
        run("tokenizer_basic", data_tests.test_tokenizer_basic)
        run("cif_validation", data_tests.test_cif_validation)
        run("prompt_creation", data_tests.test_prompt_creation)
        run("model_loading", model_tests.test_model_loading)
        run("model_forward", model_tests.test_model_forward)
        run("pkv_model_forward", model_tests.test_pkv_model_forward)
        run("prepend_model_forward", model_tests.test_prepend_model_forward)
        run("slider_model_forward", model_tests.test_slider_model_forward)
        run("conditional_model_with_labels", model_tests.test_conditional_model_with_labels)
        run("generation_basic", gen_tests.test_generation_basic)
        run("generation_conditional", gen_tests.test_generation_conditional)
        run("check_cif", gen_tests.test_check_cif)
        run("get_model_class", gen_tests.test_get_model_class)
        run("build_generation_kwargs_modes", gen_tests.test_build_generation_kwargs_modes)
        run("remove_conditionality", gen_tests.test_remove_conditionality)
        run("get_material_id", gen_tests.test_get_material_id)
        run("build_output_df", gen_tests.test_build_output_df)
        
        # Dataloader tests
        print("\n📦 Dataloader Tests:")
        run("data_collator_unconditional", dataloader_tests.test_data_collator_unconditional)
        run("data_collator_conditional", dataloader_tests.test_data_collator_conditional)
        run("data_collator_round_robin", dataloader_tests.test_data_collator_round_robin)
        run("data_collator_long_cif_slicing", dataloader_tests.test_data_collator_long_cif_slicing)
        run("load_data_unconditional", dataloader_tests.test_load_data_unconditional)
        run("load_data_conditional", dataloader_tests.test_load_data_conditional)
        run("load_data_train_uses_all_cif_variants", dataloader_tests.test_load_data_train_uses_all_cif_variants)
        run("data_collator_samples_unique_variant_content", dataloader_tests.test_data_collator_samples_unique_variant_content)
        
        # Data utils tests
        print("\n🔢 Data Utils Tests:")
        run("filter_long_cifs", data_utils_tests.test_filter_long_cifs)
        run("filter_cifs_with_unk", data_utils_tests.test_filter_cifs_with_unk)
        run("tokenize_function_unconditional", data_utils_tests.test_tokenize_function_unconditional)
        run("tokenize_function_conditional", data_utils_tests.test_tokenize_function_conditional)
        run("tokenize_function_raw", data_utils_tests.test_tokenize_function_raw)
        run("create_fixed_format_mask", data_utils_tests.test_create_fixed_format_mask)
        run("parse_condition_value", data_utils_tests.test_parse_condition_value)
        run("get_cif_candidate_columns", data_utils_tests.test_get_cif_candidate_columns)
        run("build_train_cif_variant_texts_with_fallback", data_utils_tests.test_build_train_cif_variant_texts_with_fallback)
        run("tokenize_function_train_variants", data_utils_tests.test_tokenize_function_train_variants)
        run("tokenize_function_train_variants_blank_fallback", data_utils_tests.test_tokenize_function_train_variants_blank_fallback)
        run("tokenize_function_train_variants_avoids_mask_retokenization", data_utils_tests.test_tokenize_function_train_variants_avoids_mask_retokenization)
        
        # Evaluation tests
        print("\n📊 Evaluation Tests:")
        run("vun_metrics", eval_tests.test_vun_metrics)
        run("validity_function", eval_tests.test_validity_function)
        run("uniqueness_function", eval_tests.test_uniqueness_function)
        run("novelty_function", eval_tests.test_novelty_function)
        run("density_calculation", eval_tests.test_density_calculation)
        run("formula_consistency_partial_occupancy", eval_tests.test_formula_consistency_partial_occupancy)
        run("basic_evaluation", eval_tests.test_basic_evaluation)
        
        # Data processing pipeline tests
        print("\n🗂️ Data Processing Pipeline Tests:")
        run("deduplicate_script", pipeline_tests.test_deduplicate_script)
        run("cleaning_script", pipeline_tests.test_cleaning_script)
        run("xrd_calculation", pipeline_tests.test_xrd_calculation)
        run("xrd_input_processing_script", pipeline_tests.test_xrd_input_processing_script)
        run("hf_dataset_save", pipeline_tests.test_hf_dataset_save)
        run("cleaning_variant_normalization", pipeline_tests.test_cleaning_variant_normalization)
        run("cleaning_precount_variant_row_normalization", pipeline_tests.test_cleaning_precount_variant_row_normalization)
        
        # Training pipeline tests
        print("\n🚀 Training Pipeline Tests:")
        run("training_setup", training_tests.test_training_setup)
        run("model_initialization", training_tests.test_model_initialization)
        run("xtra_augment_training_jobs", training_tests.test_xtra_augment_training_jobs)
        
        # Generation pipeline tests
        print("\n🔮 Generation Pipeline Tests:")
        run("generation_script_imports", gen_pipeline_tests.test_generation_script_imports)
        run("score_output_logp", gen_pipeline_tests.test_score_output_logp)
        run("generation_kwargs_edge_cases", gen_pipeline_tests.test_generation_kwargs_edge_cases)
        run("check_cif_comprehensive", gen_pipeline_tests.test_check_cif_comprehensive)
        run("condition_vector_parsing_comprehensive", gen_pipeline_tests.test_condition_vector_parsing_comprehensive)
        run("evaluation_script", gen_pipeline_tests.test_evaluation_script)
        run("postprocessing_script", gen_pipeline_tests.test_postprocessing_script)
        run("xtra_augment_generation_jobs", gen_pipeline_tests.test_xtra_augment_generation_jobs)
        run("sequential_full_imports", gen_pipeline_tests.test_sequential_full_imports)
        
        # Evaluation pipeline tests
        print("\n📈 Evaluation Pipeline Tests:")
        run("vun_metrics_script", eval_pipeline_tests.test_vun_metrics_script)
        run("stability_metrics", eval_pipeline_tests.test_stability_metrics)
        run("xrd_metrics", eval_pipeline_tests.test_xrd_metrics)
        run("property_metrics", eval_pipeline_tests.test_property_metrics)
        
        # Load and generate tests
        print("\n🤗 HF Load & Generate Tests:")
        run("hf_model_loading", load_gen_tests.test_hf_model_loading)
        run("prompt_generation_from_args", load_gen_tests.test_prompt_generation_from_args)
        run("mattergen_xrd_generation_smoke", load_gen_tests.test_mattergen_xrd_generation_smoke)
        run("direct_generation_logp_smoke", load_gen_tests.test_direct_generation_logp_smoke)
        run("multi_gpu_single_prompt_worker_resolution", load_gen_tests.test_multi_gpu_single_prompt_worker_resolution)
        run("scoring_mode_normalization_helper", load_gen_tests.test_scoring_mode_normalization_helper)
        run("reduced_formula_prompt_expansion", load_gen_tests.test_reduced_formula_prompt_expansion)
        run("reduced_formula_selection_modes", load_gen_tests.test_reduced_formula_selection_modes)
        run("reduced_formula_selection_uses_provided_cif_text", load_gen_tests.test_reduced_formula_selection_uses_provided_cif_text)
        run("search_zs_zero_target_keeps_all_generated_rows", load_gen_tests.test_search_zs_zero_target_keeps_all_generated_rows)
        run("xrd_raw_file_parsing_and_conversion", load_gen_tests.test_xrd_raw_file_parsing_and_conversion)
        
        # Integration tests
        print("\n🔗 Integration Tests:")
        run("minimal_training_loop", integration_tests.test_minimal_training_loop)
        run("conditional_training_loop", integration_tests.test_conditional_training_loop)
        run("generation_after_training", integration_tests.test_generation_after_training)

        # Virtualiser tests
        print("\n🔬 Virtualiser Tests:")
        run("virtualiser_import", virtualiser_tests.test_import)
        run("virtualiser_pair_fractions", virtualiser_tests.test_compute_pair_fractions)
        run("virtualiser_pair_fractions_absent", virtualiser_tests.test_compute_pair_fractions_absent_element)
        run("virtualiser_virtualise_structure", virtualiser_tests.test_virtualise_structure)
        run("virtualiser_preserves_composition", virtualiser_tests.test_virtualise_structure_preserves_composition)
        run("virtualiser_promote_symmetry", virtualiser_tests.test_promote_symmetry)
        run("virtualiser_load_config", virtualiser_tests.test_load_config)
        run("virtualiser_full_pipeline_to_cif", virtualiser_tests.test_full_pipeline_to_cif)

        # Report results
        success = suite.report_results()
        return 0 if success else 1
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        suite.cleanup()


if __name__ == "__main__":
    exit(main())
