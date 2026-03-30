"""Runner for API test suites."""

import argparse
import traceback
from datetime import datetime

from tests.api.core import APITestSuite
from tests.api.endpoints import (
    RootEndpointTests,
    JobManagementTests,
    PreprocessingEndpointTests,
    TrainingEndpointTests,
    GenerationEndpointTests,
    MetricsEndpointTests,
    CommandConstructionTests,
    APIGapTests,
    PipelineIntegrationTests,
    VirtualiserEndpointTests,
)

def run_all_tests(suite: APITestSuite, run_integration: bool = False, verbose: bool = False):
    """Run all test categories."""
    test_data = suite.create_test_data()
    
    # Root endpoint tests
    root_tests = RootEndpointTests(suite.client, suite.temp_dir)
    suite.run_test("root_returns_api_info", root_tests.test_root_returns_api_info)
    suite.run_test("root_lists_all_endpoint_categories", root_tests.test_root_lists_all_endpoint_categories)
    suite.run_test("health_endpoint", root_tests.test_health_endpoint)
    
    # Job management tests
    job_tests = JobManagementTests(suite.client, suite.temp_dir)
    suite.run_test("list_jobs_empty", job_tests.test_list_jobs_empty)
    suite.run_test("get_nonexistent_job_returns_404", job_tests.test_get_nonexistent_job_returns_404)
    suite.run_test("job_creation_returns_pending_status", job_tests.test_job_creation_returns_pending_status)
    suite.run_test("job_can_be_retrieved_after_creation", job_tests.test_job_can_be_retrieved_after_creation)
    
    # Preprocessing endpoint tests
    preproc_tests = PreprocessingEndpointTests(suite.client, suite.temp_dir, test_data, mode="smoke")
    suite.run_test("deduplicate_valid_request", preproc_tests.test_deduplicate_valid_request)
    suite.run_test("deduplicate_with_all_optional_params", preproc_tests.test_deduplicate_with_all_optional_params)
    suite.run_test("deduplicate_missing_required_field", preproc_tests.test_deduplicate_missing_required_field)
    suite.run_test("calc_theor_xrd_smoke", preproc_tests.test_calc_theor_xrd_smoke)
    suite.run_test("clean_valid_request", preproc_tests.test_clean_valid_request)
    suite.run_test("clean_with_normalizers", preproc_tests.test_clean_with_normalizers)
    suite.run_test("clean_invalid_normalizer", preproc_tests.test_clean_invalid_normalizer)
    suite.run_test("save_dataset_valid_request", preproc_tests.test_save_dataset_valid_request)
    suite.run_test("save_dataset_with_splits", preproc_tests.test_save_dataset_with_splits)
    suite.run_test("xrd_preprocessing_valid_request", preproc_tests.test_xrd_preprocessing_valid_request)
    suite.run_test("xrd_preprocessing_missing_required_field", preproc_tests.test_xrd_preprocessing_missing_required_field)
    suite.run_test("cifs_zip_to_parquet_smoke", preproc_tests.test_cifs_zip_to_parquet_smoke)
    
    # Training endpoint tests
    train_tests = TrainingEndpointTests(suite.client, suite.temp_dir)
    suite.run_test("train_single_gpu", train_tests.test_train_single_gpu)
    suite.run_test("train_multi_gpu", train_tests.test_train_multi_gpu)
    
    # Generation endpoint tests
    gen_tests = GenerationEndpointTests(suite.client, suite.temp_dir, test_data, mode="smoke")
    suite.run_test("generate_base_explicit_z", gen_tests.test_generate_base_explicit_z)
    suite.run_test("direct_generation_mapped_lists", gen_tests.test_direct_generation_mapped_lists)
    suite.run_test("direct_generation_slme_level_1", gen_tests.test_direct_generation_slme_level_1)
    suite.run_test("direct_generation_cod_xrd_early_stop", gen_tests.test_direct_generation_cod_xrd_early_stop)
    suite.run_test("direct_generation_mattergen_xrd_logp", gen_tests.test_direct_generation_mattergen_xrd_logp)
    suite.run_test("direct_generation_input_parquet_mode", gen_tests.test_direct_generation_input_parquet_mode)
    suite.run_test("direct_generation_reduced_formula_conflict", gen_tests.test_direct_generation_reduced_formula_conflict)
    suite.run_test("make_prompts_manual", gen_tests.test_make_prompts_manual)
    suite.run_test("make_prompts_automatic", gen_tests.test_make_prompts_automatic)
    suite.run_test("generate_cifs", gen_tests.test_generate_cifs)
    suite.run_test("evaluate_cifs", gen_tests.test_evaluate_cifs)
    suite.run_test("postprocess", gen_tests.test_postprocess)
    suite.run_test("direct_generation_raw_xrd_conversion", gen_tests.test_direct_generation_raw_xrd_conversion)
    suite.run_test("direct_generation_search_zs_all_rows_mode", gen_tests.test_direct_generation_search_zs_all_rows_mode)
    
    # Metrics endpoint tests
    metrics_tests = MetricsEndpointTests(suite.client, suite.temp_dir, test_data, mode="smoke")
    suite.run_test("vun_metrics", metrics_tests.test_vun_metrics)
    suite.run_test("ehull_metrics", metrics_tests.test_ehull_metrics)

    # Virtualiser endpoint tests
    virtualiser_tests = VirtualiserEndpointTests(suite.client, suite.temp_dir, test_data, mode="smoke")
    suite.run_test("virtualise_inline_pairs", virtualiser_tests.test_virtualise_with_inline_pairs)
    suite.run_test("virtualise_config_file", virtualiser_tests.test_virtualise_with_config_file)
    suite.run_test("virtualise_missing_pairs_and_config", virtualiser_tests.test_virtualise_missing_pairs_and_config)
    suite.run_test("virtualise_missing_required_fields", virtualiser_tests.test_virtualise_missing_required_fields)
    
    # Command construction tests
    cmd_tests = CommandConstructionTests(suite.client, suite.temp_dir)
    suite.run_test("deduplicate_command_structure", cmd_tests.test_deduplicate_command_structure)
    suite.run_test("direct_generation_condition_lists_format", cmd_tests.test_direct_generation_condition_lists_format)
    suite.run_test("direct_generation_all_params_in_command", cmd_tests.test_direct_generation_all_params_in_command)
    suite.run_test("train_torchrun_format", cmd_tests.test_train_torchrun_format)
    
    # API gap tests (document missing features)
    gap_tests = APIGapTests(suite.client, suite.temp_dir)
    suite.run_test("xrd_preprocessing_endpoint_now_supported", gap_tests.test_xrd_preprocessing_endpoint_now_supported)
    suite.run_test("missing_xrd_metrics_endpoint", gap_tests.test_missing_xrd_metrics_endpoint)
    suite.run_test("missing_property_metrics_endpoint", gap_tests.test_missing_property_metrics_endpoint)
    
    # Integration tests (optional, slower)
    if run_integration:
        print("\n" + "="*60)
        print("INTEGRATION TESTS (Unified execution)")
        print("="*60)
        integration_tag = datetime.now().strftime("api_itest_%Y%m%d_%H%M%S")
        docker_mode = suite.docker_url is not None

        int_preproc_tests = PreprocessingEndpointTests(
            suite.client,
            suite.temp_dir,
            test_data,
            mode="integration",
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )
        int_gen_tests = GenerationEndpointTests(
            suite.client,
            suite.temp_dir,
            test_data,
            mode="integration",
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )
        int_metrics_tests = MetricsEndpointTests(
            suite.client,
            suite.temp_dir,
            test_data,
            mode="integration",
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )
        pipeline_tests = PipelineIntegrationTests(
            suite.client,
            suite.temp_dir,
            test_data,
            mode="integration",
            docker_mode=docker_mode,
            verbose=verbose,
            integration_tag=integration_tag,
        )

        try:
            print("\nPreprocessing integration tests")
            suite.run_test("integration_deduplicate", int_preproc_tests.test_deduplicate_valid_request)
            suite.run_test("integration_calc_theor_xrd", int_preproc_tests.test_calc_theor_xrd_smoke)
            suite.run_test("integration_clean", int_preproc_tests.test_clean_valid_request)
            suite.run_test("integration_save_dataset", int_preproc_tests.test_save_dataset_valid_request)
            suite.run_test("integration_xrd_preprocessing", int_preproc_tests.test_xrd_preprocessing_valid_request)
            
            print("\nGeneration integration tests")
            # Run make_prompts FIRST so the files exist for downstream tests!
            suite.run_test("integration_make_prompts_manual", int_gen_tests.test_make_prompts_manual)
            suite.run_test("integration_make_prompts_automatic", int_gen_tests.test_make_prompts_automatic)

            # Base and mapped generation
            suite.run_test("integration_generate_base_explicit_z", int_gen_tests.test_generate_base_explicit_z)
            suite.run_test("integration_direct_generation_mapped_lists", int_gen_tests.test_direct_generation_mapped_lists)
            suite.run_test("integration_direct_generation_slme_level_1", int_gen_tests.test_direct_generation_slme_level_1)
            
            # XRD Generation tests
            suite.run_test("integration_direct_generation_cod_xrd_early_stop", int_gen_tests.test_direct_generation_cod_xrd_early_stop)
            suite.run_test("integration_direct_generation_mattergen_xrd_logp", int_gen_tests.test_direct_generation_mattergen_xrd_logp)
            suite.run_test("integration_direct_generation_raw_xrd_conversion", int_gen_tests.test_direct_generation_raw_xrd_conversion)
            
            # Parquet, Config, and conflict handling (Now safe to run)
            suite.run_test("integration_direct_generation_input_parquet_mode", int_gen_tests.test_direct_generation_input_parquet_mode)
            suite.run_test("integration_direct_generation_reduced_formula_conflict", int_gen_tests.test_direct_generation_reduced_formula_conflict)
            # integration_generate_cifs skipped: requires a locally-available model checkpoint,
            # and the full pipeline test already covers config-driven generation end-to-end.
            
            # Downstream evaluation
            suite.run_test("integration_evaluate_cifs", int_gen_tests.test_evaluate_cifs)
            suite.run_test("integration_postprocess", int_gen_tests.test_postprocess)
            
            print("\nMetrics integration tests")
            suite.run_test("integration_vun_metrics", int_metrics_tests.test_vun_metrics)
            suite.run_test("integration_ehull_metrics", int_metrics_tests.test_ehull_metrics)
            
            print("\nFull pipeline integration test")
            suite.run_test("integration_full_pipeline", pipeline_tests.test_full_generation_pipeline)
        finally:
            int_preproc_tests.cleanup_artifacts()

def main():
    """Run the API test suite."""
    parser = argparse.ArgumentParser(description="CrystaLLM-pi API Test Suite")
    parser.add_argument("--hf_key", type=str, default="", help="HuggingFace API key (not needed for Docker testing)")
    parser.add_argument("--wandb_key", type=str, default="", help="Weights & Biases API key (not needed for Docker testing)")
    parser.add_argument("--docker_url", type=str, default=None, 
                        help="URL of running Docker container (e.g., http://localhost:8000)")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests (slower, requires actual execution)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output: sample CIFs, validity stats, E-hull values")
    args = parser.parse_args()
    
    print("="*60)
    print("CrystaLLM-pi API Test Suite")
    print("="*60)
    
    if args.docker_url:
        print(f"Testing against Docker: {args.docker_url}")
    else:
        if not args.hf_key or not args.wandb_key:
            print("Warning: --hf_key and --wandb_key not provided.")
            print("For local testing, these are recommended.")
            print("For Docker testing, use --docker_url instead.")
        print("Testing locally with FastAPI TestClient")
    
    if args.verbose:
        print("Verbose mode: Will show sample outputs from integration tests")
    
    suite = APITestSuite(args.hf_key, args.wandb_key, args.docker_url)
    suite.setup()
    fatal_error = None
    
    try:
        run_all_tests(suite, run_integration=args.integration, verbose=args.verbose)
    except Exception as e:
        print(f"\nFatal error during tests: {e}")
        import traceback
        traceback.print_exc()
        fatal_error = e
    finally:
        suite.cleanup()

    exit_code = suite.report_results()
    if fatal_error is not None:
        return 1
    return exit_code


if __name__ == "__main__":
    exit(main())
