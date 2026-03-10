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
        
        # Execute tests
        print("Running CrystaLLM-pi Comprehensive Test Suite...")
        print("-" * 50)
        
        # Core component tests
        print("\n🔧 Core Component Tests:")
        suite.run_test("tokenizer_basic", data_tests.test_tokenizer_basic)
        suite.run_test("cif_validation", data_tests.test_cif_validation)
        suite.run_test("prompt_creation", data_tests.test_prompt_creation)
        suite.run_test("model_loading", model_tests.test_model_loading)
        suite.run_test("model_forward", model_tests.test_model_forward)
        suite.run_test("pkv_model_forward", model_tests.test_pkv_model_forward)
        suite.run_test("prepend_model_forward", model_tests.test_prepend_model_forward)
        suite.run_test("slider_model_forward", model_tests.test_slider_model_forward)
        suite.run_test("conditional_model_with_labels", model_tests.test_conditional_model_with_labels)
        suite.run_test("generation_basic", gen_tests.test_generation_basic)
        suite.run_test("generation_conditional", gen_tests.test_generation_conditional)
        suite.run_test("check_cif", gen_tests.test_check_cif)
        suite.run_test("get_model_class", gen_tests.test_get_model_class)
        suite.run_test("build_generation_kwargs_modes", gen_tests.test_build_generation_kwargs_modes)
        suite.run_test("remove_conditionality", gen_tests.test_remove_conditionality)
        suite.run_test("get_material_id", gen_tests.test_get_material_id)
        suite.run_test("build_output_df", gen_tests.test_build_output_df)
        
        # Dataloader tests
        print("\n📦 Dataloader Tests:")
        suite.run_test("data_collator_unconditional", dataloader_tests.test_data_collator_unconditional)
        suite.run_test("data_collator_conditional", dataloader_tests.test_data_collator_conditional)
        suite.run_test("data_collator_round_robin", dataloader_tests.test_data_collator_round_robin)
        suite.run_test("data_collator_long_cif_slicing", dataloader_tests.test_data_collator_long_cif_slicing)
        suite.run_test("load_data_unconditional", dataloader_tests.test_load_data_unconditional)
        suite.run_test("load_data_conditional", dataloader_tests.test_load_data_conditional)
        
        # Data utils tests
        print("\n🔢 Data Utils Tests:")
        suite.run_test("filter_long_cifs", data_utils_tests.test_filter_long_cifs)
        suite.run_test("filter_cifs_with_unk", data_utils_tests.test_filter_cifs_with_unk)
        suite.run_test("tokenize_function_unconditional", data_utils_tests.test_tokenize_function_unconditional)
        suite.run_test("tokenize_function_conditional", data_utils_tests.test_tokenize_function_conditional)
        suite.run_test("tokenize_function_raw", data_utils_tests.test_tokenize_function_raw)
        suite.run_test("create_fixed_format_mask", data_utils_tests.test_create_fixed_format_mask)
        suite.run_test("parse_condition_value", data_utils_tests.test_parse_condition_value)
        
        # Evaluation tests
        print("\n📊 Evaluation Tests:")
        suite.run_test("vun_metrics", eval_tests.test_vun_metrics)
        suite.run_test("validity_function", eval_tests.test_validity_function)
        suite.run_test("uniqueness_function", eval_tests.test_uniqueness_function)
        suite.run_test("novelty_function", eval_tests.test_novelty_function)
        suite.run_test("density_calculation", eval_tests.test_density_calculation)
        suite.run_test("formula_consistency_partial_occupancy", eval_tests.test_formula_consistency_partial_occupancy)
        suite.run_test("basic_evaluation", eval_tests.test_basic_evaluation)
        
        # Data processing pipeline tests
        print("\n🗂️ Data Processing Pipeline Tests:")
        suite.run_test("deduplicate_script", pipeline_tests.test_deduplicate_script)
        suite.run_test("cleaning_script", pipeline_tests.test_cleaning_script)
        suite.run_test("xrd_calculation", pipeline_tests.test_xrd_calculation)
        suite.run_test("xrd_input_processing_script", pipeline_tests.test_xrd_input_processing_script)
        suite.run_test("hf_dataset_save", pipeline_tests.test_hf_dataset_save)
        
        # Training pipeline tests
        print("\n🚀 Training Pipeline Tests:")
        suite.run_test("training_setup", training_tests.test_training_setup)
        suite.run_test("model_initialization", training_tests.test_model_initialization)
        
        # Generation pipeline tests
        print("\n🔮 Generation Pipeline Tests:")
        suite.run_test("generation_script_imports", gen_pipeline_tests.test_generation_script_imports)
        suite.run_test("score_output_logp", gen_pipeline_tests.test_score_output_logp)
        suite.run_test("generation_kwargs_edge_cases", gen_pipeline_tests.test_generation_kwargs_edge_cases)
        suite.run_test("check_cif_comprehensive", gen_pipeline_tests.test_check_cif_comprehensive)
        suite.run_test("condition_vector_parsing_comprehensive", gen_pipeline_tests.test_condition_vector_parsing_comprehensive)
        suite.run_test("evaluation_script", gen_pipeline_tests.test_evaluation_script)
        suite.run_test("postprocessing_script", gen_pipeline_tests.test_postprocessing_script)
        
        # Evaluation pipeline tests
        print("\n📈 Evaluation Pipeline Tests:")
        suite.run_test("vun_metrics_script", eval_pipeline_tests.test_vun_metrics_script)
        suite.run_test("stability_metrics", eval_pipeline_tests.test_stability_metrics)
        suite.run_test("xrd_metrics", eval_pipeline_tests.test_xrd_metrics)
        suite.run_test("property_metrics", eval_pipeline_tests.test_property_metrics)
        
        # Load and generate tests
        print("\n🤗 HF Load & Generate Tests:")
        suite.run_test("hf_model_loading", load_gen_tests.test_hf_model_loading)
        suite.run_test("prompt_generation_from_args", load_gen_tests.test_prompt_generation_from_args)
        suite.run_test("mattergen_xrd_generation_smoke", load_gen_tests.test_mattergen_xrd_generation_smoke)
        suite.run_test("direct_generation_logp_smoke", load_gen_tests.test_direct_generation_logp_smoke)
        suite.run_test("multi_gpu_single_prompt_worker_resolution", load_gen_tests.test_multi_gpu_single_prompt_worker_resolution)
        suite.run_test("scoring_mode_normalization_helper", load_gen_tests.test_scoring_mode_normalization_helper)
        suite.run_test("reduced_formula_prompt_expansion", load_gen_tests.test_reduced_formula_prompt_expansion)
        suite.run_test("reduced_formula_selection_modes", load_gen_tests.test_reduced_formula_selection_modes)
        suite.run_test("reduced_formula_selection_uses_provided_cif_text", load_gen_tests.test_reduced_formula_selection_uses_provided_cif_text)
        suite.run_test("search_zs_zero_target_keeps_all_generated_rows", load_gen_tests.test_search_zs_zero_target_keeps_all_generated_rows)
        suite.run_test("xrd_raw_file_parsing_and_conversion", load_gen_tests.test_xrd_raw_file_parsing_and_conversion)
        
        # Integration tests
        print("\n🔗 Integration Tests:")
        suite.run_test("minimal_training_loop", integration_tests.test_minimal_training_loop)
        suite.run_test("conditional_training_loop", integration_tests.test_conditional_training_loop)
        suite.run_test("generation_after_training", integration_tests.test_generation_after_training)
        
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
