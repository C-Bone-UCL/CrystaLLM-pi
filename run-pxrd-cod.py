"""
PXRD pipeline script for generating, postprocessing, and computing XRD metrics.
"""

import subprocess
import datetime
import os
import commentjson

def run_command_with_logging(cmd, step_name, timestamp):
    """Helper function to run commands with logging"""
    os.makedirs("__logs__", exist_ok=True)
    
    with open(f"__logs__/{step_name}_output_{timestamp}.log", "w") as out_file, \
         open(f"__logs__/{step_name}_error_{timestamp}.log", "w") as err_file:
        subprocess.check_call(cmd, stdout=out_file, stderr=err_file)
    
    print(f"{step_name} completed. Logs saved to logs/{step_name}_*_{timestamp}.log")

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    gen_config = "_config_files/generation/conditional/xrd_studies/cod-xrd_eval.jsonc"
    gen_parquet = "_artifacts/cod-xrd/cod-ft-20perp-test_gen.parquet"
    post_parquet = "_artifacts/cod-xrd/cod-ft-20perp-test_post.parquet"
    ref_parquet = "_artifacts/cod-xrd/cod-test_ref.parquet"
    metrics_parquet = "_artifacts/cod-xrd/cod-ft-20perp-test_metrics.parquet"
    metrics_1gen_parquet = "_artifacts/cod-xrd/cod-ft-1perp-test_metrics.parquet"

    # make sure ref_parquet exists and gen_confif exists
    assert os.path.exists(ref_parquet), f"Reference parquet file not found: {ref_parquet}"
    assert os.path.exists(gen_config), f"Generation config file not found: {gen_config}"

    with open(gen_config, "r") as f:
        config = commentjson.load(f)
    assert config.get("output_parquet", "") == gen_parquet, \
        f"Output parquet in config ({config.get('output_parquet', '')}) does not match gen_parquet ({gen_parquet})"

    # assert 1 == 0, "Pipeline paused for verification. Remove this line to continue."


    print("Starting PXRD pipeline")
    
    # Step 1: Generate CIFs
    print("\nStep 1: Generating CIFs...")
    try:
        cmd_1 = [
            "python", "_utils/_generating/generate_CIFs.py",
            "--config", gen_config,
        ]
        run_command_with_logging(cmd_1, "generate_cifs", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during CIF generation: {e}")
        return

    # Step 2: Postprocess generated CIFs  
    print("\nStep 2: Postprocessing generated CIFs...")
    try:
        cmd_2 = [
            "python", "_utils/_generating/postprocess.py",
            "--input_parquet", gen_parquet,
            "--output_parquet", post_parquet, 
            "--num_workers", "32",
            "--column_name", "Generated CIF"
        ]
        run_command_with_logging(cmd_2, "postprocess", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during postprocessing: {e}")
        return

    # Step 3: Compute XRD metrics
    print("\nStep 3: Computing XRD metrics...")
    try:
        cmd_3 = [
            "python", "_utils/_metrics/XRD_metrics.py",
            "--input_parquet", post_parquet,
            "--num_gens", "20",
            "--ref_parquet", ref_parquet,
            "--output_parquet", metrics_parquet,
            "--num_workers", "16",
            "--validity_check", "none"
        ]
        run_command_with_logging(cmd_3, "xrd_metrics", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during XRD metrics computation: {e}")
        return
    
    print("\nStep 4: Computing XRD metrics for 1 gen...")
    try:
        cmd_4 = [
            "python", "_utils/_metrics/XRD_metrics.py",
            "--input_parquet", post_parquet,
            "--num_gens", "1",
            "--ref_parquet", ref_parquet,
            "--output_parquet", metrics_1gen_parquet,
            "--num_workers", "16",
            "--validity_check", "none"
        ]
        run_command_with_logging(cmd_4, "xrd_metrics_1gen", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during XRD metrics computation: {e}")
        return

    print(f"\nPXRD pipeline completed successfully! All logs saved with timestamp {timestamp}")

if __name__ == "__main__":
    main()