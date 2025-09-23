##############################
# Script to execute multiple runs of a training script with different configs
##############################

import os
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="configs")
    args = parser.parse_args()

    # Load all config files
    config_files = os.listdir(args.config_dir)

    # keep only config files that end with "_post.parquet"
    # config_files = [f for f in config_files if f.endswith("_post.parquet")]

    for config_file in config_files:
        try:
            ## Path variables
            config_path = os.path.join(args.config_dir, config_file)
            # gen_config_file = config_file.replace(".jsonc", "_eval.jsonc")
            # gen_dir = "_config_files/cg_eval/raw"
            # gen_config_path = os.path.join(gen_dir, gen_config_file)
            # parquet_out = config_path.replace("_gen.parquet", "_gen_post.parquet")
            # metrics_out = config_path.replace("_gen.parquet", "_metrics.csv")
            # config_out = config_path.replace("_post.parquet", "_post_orb.parquet")

            ## Script execution
            print(f"Running script with --config {config_path}")
            # print(f"Running script with --gen_config {gen_config_path}")
            # Run training script with config
            os.system(f"torchrun --nproc_per_node=2 _train.py --config '{config_path}'")
            # os.system(f"python _train.py --config '{config_path}'")
            # os.system(f"python _utils/_evaluation_og/bench_pipeline.py --config '{config_path}'")
            # os.system(f"python _utils/_evaluation_conditional/generate_CIFs.py --config '{config_path}'")
            # os.system(f"python _utils/_evaluation_og/generate_CIFs.py --config '{config_path}'")
            # os.system(f"python _utils/_evaluation_conditional/CG_pipeline.py --train_config '{config_path}' --eval_config '{gen_config_path}'")
            # os.system(f"python _utils/_evaluation_conditional/CG_pipeline.py --eval_config '{config_path}'")
            # os.system(f"python _utils/_evaluation_conditional/orb_ehull.py --input_parquet '{config_path}' --output_parquet '{config_out}'")
        
        except Exception as e:
            print(f"Error running training with config {config_path}: {e}")
            continue

if __name__ == "__main__":
    main()