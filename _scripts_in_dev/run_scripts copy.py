import subprocess
import argparse
import datetime
import csv
import re
import os
from huggingface_hub import login
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


    print("Thermo eleectrics train, gen and process results")
    try:
        config_1 = '_config_files/cg_train/thermo_study/ricci_full_ft-PKV.jsonc'
        cmd_1 = [
            "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
            "torchrun", "--nproc_per_node=2", "_train.py",
            "--config", config_1,
        ]

        run_command_with_logging(cmd_1, "training", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")

    try:
        config_2 = '_config_files/cg_eval/Thermo_study/ricci_thermo_eval.jsonc'
        cmd_2 = [
            "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
            "python", "_utils/_evaluation_conditional/generate_CIFs.py",
            "--config", config_2,
        ]

        run_command_with_logging(cmd_2, "generation", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")

    try:
        input_file_3 = '_utils/_evaluation_conditional/evaluation_files/ricci-full-ovlp-test_gen.parquet'
        output_file_3 = '_utils/_evaluation_conditional/evaluation_files/ricci-full-ovlp-test_proc.parquet'
        cmd_3 = [
            "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
            "python", "_utils/_evaluation_og/postprocess.py",
            "--input_file", input_file_3,
            "--output_file", output_file_3,
            "--num_workers", "64",
            "--column_name", "Generated CIF"
        ]
        run_command_with_logging(cmd_3, "postprocessing", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during postprocessing: {e}")


    try:
        input_parquet_4 = '_utils/_evaluation_conditional/evaluation_files/ricci-full-ovlp-test_proc.parquet'
        path_to_db_4 = '_utils/_evaluation_conditional/evaluation_files/ricci-full-ovlp-test_prompts.parquet'
        output_parquet_4 = '_utils/_evaluation_conditional/evaluation_files/ricci-full-ovlp-test_metrics.parquet'
        cmd_4 = [
            "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
            "python", "_utils/_evaluation_og/compare_gen_true.py",
            "--input_parquet", input_parquet_4,
            "--path_to_db", path_to_db_4,
            "--num_gens", "20",
            "--num_workers", "64",
            "--output_parquet", output_parquet_4,
            "--group_by_condition"
        ]
        run_command_with_logging(cmd_4, "comparison", timestamp)
    except subprocess.CalledProcessError as e:
        print(f"Error during comparison of generated and true CIFs: {e}")


    # print("Unconditional SiO2 generation")
    # try:
    #     config_5 = '_config_files/cg_eval/SiO2_study/mgen_SiO2-uncond-10T15K_eval.jsonc'
    #     cmd_5 = [
    #         "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
    #         "python", "_utils/_evaluation_og/generate_CIFs.py",
    #         "--config", config_5,
    #     ]
    #     run_command_with_logging(cmd_5, "SiO2_generation", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during SiO2 generation: {e}")

    # try:
    #     cmd_6 = [
    #         "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #         "python", "_utils/_evaluation_conditional/metrics_CG.py",
    #         "--gen_data", "_utils/_evaluation_conditional/evaluation_files/SiO2-uncond-10T15K_gen.parquet",
    #         "--huggingface_dataset", "c-bone/mattergen_den_ehull",
    #         "--parquet_out", "_utils/_evaluation_conditional/evaluation_files/SiO2-uncond-10T15K_post.parquet",
    #         "--metrics_out", "_utils/_evaluation_conditional/evaluation_files/SiO2-uncond-10T15K_metrics.csv",
    #         "--sort_metrics_by", "both",
    #         "--num_workers", "8",
    #         "--property_targets", "['Density (g/cm^3)', 'energy_above_hull']",
    #         "--property1_normaliser", "linear",
    #         "--min_property1", "0.0",
    #         "--max_property1", "25.494",
    #         "--property2_normaliser", "linear",
    #         "--min_property2", "0.0",
    #         "--max_property2", "0.1",
    #         "--load_processed_data", "HF-databases/mattergen_dev/mgen_den_ehull_proc.parquet"
    #     ]
    #     run_command_with_logging(cmd_6, "SiO2_metrics", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during SiO2 metrics computation: {e}")

    # print("TiO2 Discovery")

    # try:
    #     config_7 = '_config_files/cg_eval/TiO2_study/mgen_uncond-TiO2-10T15K_eval.jsonc'
    #     cmd_7 = [
    #         "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
    #         "python", "_utils/_evaluation_og/generate_CIFs.py",
    #         "--config", config_7,
    #     ]
    #     run_command_with_logging(cmd_7, "TiO2_generation", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 generation: {e}")


    # try:
    #     cmd_8 = [
    #         "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #         "python", "_utils/_evaluation_conditional/metrics_CG.py",
    #         "--gen_data", "_utils/_evaluation_conditional/evaluation_files/TiO2-uncond-10T15K_gen.parquet",
    #         "--huggingface_dataset", "c-bone/mattergen_bg_ehull",
    #         "--parquet_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-uncond-10T15K_post.parquet",
    #         "--metrics_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-uncond-10T15K_metrics.csv",
    #         "--sort_metrics_by", "both",
    #         "--num_workers", "8",
    #         "--property_targets", "['ALIGNN_BG', 'energy_above_hull']",
    #         "--property1_normaliser", "power_log",
    #         "--max_property1", "9.242",
    #         "--property2_normaliser", "linear",
    #         "--max_property2", "0.1",
    #         "--min_property2", "0.0",
    #         "--load_processed_data", "HF-databases/mattergen_dev/mgen_bg_ehull_proc.parquet"
    #     ]
    #     run_command_with_logging(cmd_8, "TiO2_metrics", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 metrics computation: {e}")

    # print("TiO2 Recovery")
    # try:
    #     config_9 = '_config_files/cg_eval/TiO2_study/mgen_uncond-rutile-TiO2-10T15K_eval.jsonc'
    #     cmd_9 = [
    #         "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
    #         "python", "_utils/_evaluation_og/generate_CIFs.py",
    #         "--config", config_9,
    #     ]
    #     run_command_with_logging(cmd_9, "TiO2_rutile_generation", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 rutile generation: {e}")

    # try:
    #     config_10 = '_config_files/cg_eval/TiO2_study/mgen_uncond-anatase-TiO2-10T15K_eval.jsonc'
    #     cmd_10 = [
    #         "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
    #         "python", "_utils/_evaluation_og/generate_CIFs.py",
    #         "--config", config_10,
    #     ]
    #     run_command_with_logging(cmd_10, "TiO2_anatase_generation", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 anatase generation: {e}")

    # try:
    #     cmd_11 = [
    #         "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #         "python", "_utils/_evaluation_conditional/metrics_CG.py",
    #         "--gen_data", "_utils/_evaluation_conditional/evaluation_files/TiO2-anatase-uncond-10T15K_gen.parquet",
    #         "--huggingface_dataset", "c-bone/mattergen_bg_ehull",
    #         "--parquet_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-anatase-uncond-10T15K_post.parquet",
    #         "--metrics_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-anatase-uncond-10T15K_metrics.csv",
    #         "--sort_metrics_by", "both",
    #         "--num_workers", "8",
    #         "--property_targets", "['ALIGNN_BG', 'energy_above_hull']",
    #         "--property1_normaliser", "power_log",
    #         "--max_property1", "9.242",
    #         "--property2_normaliser", "linear",
    #         "--max_property2", "0.1",
    #         "--min_property2", "0.0",
    #         "--load_processed_data", "HF-databases/mattergen_dev/mgen_bg_ehull_proc.parquet"
    #     ]
    #     run_command_with_logging(cmd_11, "TiO2_anatase_metrics", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 anatase metrics computation: {e}")

    # try:
    #     cmd_12 = [
    #         "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #         "python", "_utils/_evaluation_conditional/metrics_CG.py",
    #         "--gen_data", "_utils/_evaluation_conditional/evaluation_files/TiO2-rutile-uncond-10T15K_gen.parquet",
    #         "--huggingface_dataset", "c-bone/mattergen_bg_ehull",
    #         "--parquet_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-rutile-uncond-10T15K_post.parquet",
    #         "--metrics_out", "_utils/_evaluation_conditional/evaluation_files/TiO2-rutile-uncond-10T15K_metrics.csv",
    #         "--sort_metrics_by", "both",
    #         "--num_workers", "8",
    #         "--property_targets", "['ALIGNN_BG', 'energy_above_hull']",
    #         "--property1_normaliser", "power_log",
    #         "--max_property1", "9.242",
    #         "--property2_normaliser", "linear",
    #         "--max_property2", "0.1",
    #         "--min_property2", "0.0",
    #         "--load_processed_data", "HF-databases/mattergen_dev/mgen_bg_ehull_proc.parquet"
    #     ]
    #     run_command_with_logging(cmd_12, "TiO2_rutile_metrics", timestamp)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error during TiO2 rutile metrics computation: {e}")
    

if __name__ == "__main__":
    main()
