import subprocess
import argparse
import datetime
import csv
import re
import os
from huggingface_hub import login
import commentjson

def main():
    with open('API_keys.jsonc', 'r') as hf_key_path:
        data = commentjson.load(hf_key_path)
    hf_key_json = str(data['HF_key'])
    
    login(hf_key_json)

    parser = argparse.ArgumentParser(description="Automate generation, postprocessing, and benchmarking.")
    parser.add_argument("--train_config", default=None,
                        help="Path to JSONC config file for training.")
    parser.add_argument("--eval_config", default=None, required=True,
                        help="Path to JSONC config file.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes to use.")
    
    args = parser.parse_args()

    print ("\n==========================")
    print(
        "Reminder of things to set:\
          \n\nshould have [chosen_name].jsonc for train and [chosen_name]_eval.jsonc for gen params\
          \nIf dataset has been changed, remember to set new targets and max property values for scaling\
          \nIf you want to change the number of workers, set it in the command line (rn is 8)"
    )
    print ("==========================\n")
    
    print("#######################")
    print("Generate is on conditional")
    print("#######################\n")

    # model_name = re.search(r'\/([^\/]+).jsonc', args.train_config).group(1)

    if args.train_config is not None:
        # Check if the config file exists
        if not os.path.exists(args.train_config):
            raise FileNotFoundError(f"Training config file {args.train_config} does not exist.")
        # Check if the config file is a JSONC file
        if not args.train_config.endswith('.jsonc'):
            raise ValueError(f"Training config file {args.train_config} is not a JSONC file.")
    if args.eval_config is not None:
        # Check if the config file exists
        if not os.path.exists(args.eval_config):
            raise FileNotFoundError(f"Evaluation config file {args.eval_config} does not exist.")
        # Check if the config file is a JSONC file
        if not args.eval_config.endswith('.jsonc'):
            raise ValueError(f"Evaluation config file {args.eval_config} is not a JSONC file.")

    print(f"Training config: {args.train_config}, training model now")
    if args.train_config is not None:
        train_cmd = [
            "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
            "torchrun", "--nproc_per_node=2", "_train.py",
            "--config", args.train_config
        ]
        # If only on one GPU
        # train_cmd = [
        #     "python", "_train.py",
        #     "--config", args.train_config
        # ]

        # Uncomment to run the training command
        # subprocess.check_call(train_cmd)

    # if config path config_files/benchmark_configs/large_configs/MP-20-large_eval.jsonc, store the everything between the last / and the last _eval.jsonc as the model name
    model_name = re.search(r'\/([^\/]+)_eval.jsonc', args.eval_config).group(1)

    print(f"Model name: {model_name}")
    # args.eval_config = f"_config_files/cg_eval/scratch/{model_name}_eval.jsonc"
    gen_input_file = f"_utils/_evaluation_conditional/evaluation_files/{model_name}_gen.parquet"
    print(f"Generated input file: {gen_input_file}")

    # 1) Generate CIFs
    print("\n=== Generating CIFs ===")
    gen_cmd = [
        "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
        "python", "_utils/_evaluation_conditional/generate_CIFs.py",
        "--config", args.eval_config,
        "--parquet_out", gen_input_file,
    ]
    subprocess.check_call(gen_cmd)

    # make args.gen_input_file the path to the generated parquet file it should be 'model_evaluation/evaluation_files/benchmark/' + model_name + '_gen.parquet'
    postprocess_output_file = f"_utils/_evaluation_conditional/evaluation_files/{model_name}_post.parquet"
    metrics_out_file = f"_utils/_evaluation_conditional/evaluation_files/{model_name}_metrics.csv"
    stability_output_file = f"_utils/_evaluation_conditional/evaluation_files/{model_name}_post-s.parquet"
    print(f"Postprocess output file: {postprocess_output_file}")
    print(f"Metrics output file: {metrics_out_file}")
    print(f"Stability output file: {stability_output_file}")


    # Depennding on the ref dataset, we need to change the args of the metrics script
    # 3) metrics
    # print("\n=== Metrics ===")
    # metrics_cmd = [
    #     "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #     "python", "-u", "_utils/_evaluation_conditional/metrics_CG.py",
    #     "--gen_data", gen_input_file,
    #     "--huggingface_dataset", "c-bone/mattergen_bg_ehull",
    #     "--parquet_out", postprocess_output_file,
    #     "--metrics_out", metrics_out_file,
    #     "--sort_metrics_by", "both",
    #     "--num_workers", str(args.num_workers),
    #     "--property_targets", "['ALIGNN_BG', 'energy_above_hull']",
    #     "--property1_normaliser", "power_log",
    #     "--max_property1", "9.242",
    #     "--property2_normaliser", "linear",
    #     "--max_property2", "0.1",
    #     "--min_property2", "0.0",
    # ]

    # 3) metrics
    print("\n=== Metrics ===")
    metrics_cmd = [
        "conda", "run", "--no-capture-output", "-n", "my_alignn",
        "python", "-u", "_utils/_evaluation_conditional/metrics_CG.py",
        "--gen_data", gen_input_file,
        "--huggingface_dataset", "c-bone/mattergen_den_ehull",
        "--parquet_out", postprocess_output_file,
        "--metrics_out", metrics_out_file,
        "--sort_metrics_by", "both",
        "--num_workers", str(args.num_workers),
        "--property_targets", "['Density (g/cm^3)', 'energy_above_hull']",
        "--property1_normaliser", "linear",
        "--min_property1", "0.0",
        "--max_property1", "25.494",
        "--property2_normaliser", "linear",
        "--min_property2", "0.0",
        "--max_property2", "0.1",
        "--load_processed_data",
    ]

    # # # # 3) metrics
    # print("\n=== Metrics ===")
    # metrics_cmd = [
    #     "conda", "run", "--no-capture-output", "-n", "my_alignn",
    #     "python", "-u", "_utils/_evaluation_conditional/metrics_CG.py",
    #     "--gen_data", gen_input_file,
    #     "--huggingface_dataset", "c-bone/mpdb-2prop_clean",
    #     "--parquet_out", postprocess_output_file,
    #     "--metrics_out", metrics_out_file,
    #     "--sort_metrics_by", "both",
    #     "--num_workers", str(args.num_workers),
    #     "--property_targets", "['Bandgap (eV)', 'Energy Above Hull (eV)']",
    #     "--property1_normaliser", "power_log",
    #     "--max_property1", "17.891",
    #     "--property2_normaliser", "power_log",
    #     "--max_property2", "5.418",
    # ]

    subprocess.check_call(metrics_cmd)
    
    # 4) orb 
    # _utils/_evaluation_conditional/orb_ehull.py
    print("\n=== Orbital Energy Above Hull ===")
    orb_cmd = [
        "conda", "run", "--no-capture-output", "-n", "crystallmv2_venv",
        "python", "-u", "_utils/_evaluation_conditional/mace_ehull.py",
        "--post_parquet", postprocess_output_file,
        "--output_parquet", stability_output_file,
    ]

    subprocess.check_call(orb_cmd)


if __name__ == "__main__":
    main()
