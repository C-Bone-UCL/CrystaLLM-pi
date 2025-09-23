import subprocess
import argparse
import datetime
import csv
import re
import os
from huggingface_hub import login
import commentjson
import pandas as pd

def run_command_with_logging(cmd, step_name, timestamp):
    """Helper function to run commands with logging"""
    os.makedirs("logs", exist_ok=True)
    
    with open(f"logs/{step_name}_output_{timestamp}.log", "w") as out_file, \
         open(f"logs/{step_name}_error_{timestamp}.log", "w") as err_file:
        subprocess.check_call(cmd, stdout=out_file, stderr=err_file)
    
    print(f"{step_name} completed. Logs saved to logs/{step_name}_*_{timestamp}.log")

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df1 = "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-slider_post.parquet"

    df2 = "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-slider_post.parquet"

    df3 = "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-PKV_post.parquet"

    df4 = "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-PKV_post.parquet"

    df5 = "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-prepend_post.parquet"

    df6 = "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-prepend_post.parquet"

    df7 = "_utils/_evaluation_conditional/evaluation_files/raw_tests/mpdb_scratch-raw_post.parquet"

    df8 = "_utils/_evaluation_conditional/evaluation_files/SiO2/mgen_PKV-SiO2-den-ehull-10T15K_post-s.parquet"

    df9 = "_utils/_evaluation_conditional/evaluation_files/SiO2/mgen_PKV-SiO2-den-ehull-uncond-10T15K_post-s.parquet"

    df10 = "_utils/_evaluation_conditional/evaluation_files/TiO2/mgen_PKV-TiO2-10T15K_post-s.parquet"

    try:
        for df in [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]:
            if not os.path.exists(df):
                raise FileNotFoundError(f"File {df} does not exist.")
 
            cmd_3 = [
                "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv", 
                "python", "_utils/_evaluation_conditional/mace_ehull.py",
                "--post_parquet", df,
                "--output_parquet", df
            ]
            run_command_with_logging(cmd_3, "ehull_mace", timestamp)

    except subprocess.CalledProcessError as e:
        print(f"Error during postprocessing: {e}")


if __name__ == "__main__":
    main()
