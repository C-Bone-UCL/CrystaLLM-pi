import subprocess
import argparse
import datetime
import csv
import re
import os
from huggingface_hub import login
import commentjson
import pandas as pd
from _utils._metrics_utils import get_novelty
from _utils._evaluation_og.postprocess import process_dataframe
from pymatgen.core import Structure
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
from tqdm import tqdm

def load_structures_from_df(df: pd.DataFrame, cif_column: str = "CIF"):

    structures = []
    print(f"Loading structures from column '{cif_column}' …")

    for _, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True, desc="Parse CIFs"):
        cif_str = row.get(cif_column, None)

        # Missing or blank CIF → None
        if pd.isna(cif_str) or str(cif_str).strip() == "":
            structures.append(None)
            continue

        # Parse with warnings suppressed (to match tolerant behavior)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = Structure.from_str(str(cif_str), fmt="cif")
            structures.append(s)
        except Exception:
            structures.append(None)

    valid_count = sum(1 for s in structures if s is not None)
    print(f"Successfully loaded {valid_count}/{len(structures)} structures")
    return structures

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define all dataframe paths
    dataframes = {
        # "mpdb_bg_scratch_slider": "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-slider_post.parquet",
        # "mpdb_bg_ft_ehull_slider": "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-slider_post.parquet",
        # "mpdb_bg_scratch_PKV": "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-PKV_post.parquet",
        # "mpdb_bg_ft_ehull_PKV": "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-PKV_post.parquet",
        # "mpdb_bg_scratch_prepend": "_utils/_evaluation_conditional/evaluation_files/scratch-methods/mpdb_scratch-prepend_post.parquet",
        # "mpdb_bg_ft_ehull_prepend": "_utils/_evaluation_conditional/evaluation_files/ft-method-reruns/mpdb-bg-ehull-prepend_post.parquet",
        # "mpdb_bg_scratch_raw": "_utils/_evaluation_conditional/evaluation_files/raw_tests/mpdb_scratch-raw_post.parquet",
        # "mgen_den_SiO2_cond": "_utils/_evaluation_conditional/evaluation_files/SiO2/mgen_PKV-SiO2-den-ehull-10T15K_post-s.parquet",
        "mgen_bg_TiO2": "_utils/_evaluation_conditional/evaluation_files/TiO2/mgen_PKV-TiO2-10T15K_post-s.parquet",
        "mgen_den_SiO2_uncond": "_utils/_evaluation_conditional/evaluation_files/SiO2/mgen_PKV-SiO2-den-ehull-uncond-10T15K_post-s.parquet",
    }

    # Reference dataset path (you'll need to specify this)
    mpdb_bg_path = "c-bone/mpdb-2prop_clean"
    mgen_den_path = "c-bone/mattergen_den_ehull"
    mgen_bg_path = "c-bone/mattergen_bg_ehull"

    # Processed reference dataset path
    mpdb_bg_processed_ref_path = "HF-databases/mpdb-2prop_clean/mpdb_bg_ehull_proc.parquet"
    mgen_bg_processed_ref_path = "HF-databases/mattergen_dev/mgen_bg_ehull_proc.parquet"
    mgen_den_processed_ref_path = "HF-databases/mattergen_dev/mgen_den_ehull_proc.parquet"


    try:
        # Process each dataframe for novelty
        for name, df_path in dataframes.items():
            if "mpdb_bg" in name:
                ref_dataset_path = mpdb_bg_path
                ref_processed_path = mpdb_bg_processed_ref_path
            elif "mgen_den" in name:
                ref_dataset_path = mgen_den_path
                ref_processed_path = mgen_den_processed_ref_path
            elif "mgen_bg" in name:
                ref_dataset_path = mgen_bg_path
                ref_processed_path = mgen_bg_processed_ref_path
            if not os.path.exists(df_path):
                raise FileNotFoundError(f"File {df_path} does not exist.")


            print(f"\n{'='*50}")
            print(f"Processing {name}: {df_path}")
            print(f"{'='*50}")

            print(f"Loading training dataset from Hugging Face: {ref_dataset_path} (train split)")


            train_dataset = load_dataset(ref_dataset_path, split="train")
            try:
                train_df = train_dataset.to_pandas()
            except Exception:
                train_df = train_dataset.data.to_pandas()

            if "CIF" not in train_df.columns:
                raise KeyError(
                    f"Training dataset is missing CIF column 'CIF', "
                    f"Available columns: {list(train_df.columns)}"
                )
            
            # Load the generated dataframe
            df_gen = pd.read_parquet(df_path)
            print(f"Loaded {len(df_gen)} generated structures")
            
            # Load structures from CIF strings
            print("Loading structures from CIF strings...")
            structures = load_structures_from_df(df_gen, cif_column="Generated CIF")
            valid_structures = sum(1 for s in structures if s is not None)
            print(f"Successfully loaded {valid_structures}/{len(structures)} structures")
            
            # Calculate novelty
            df_gen_with_novelty = get_novelty(
                df_gen=df_gen,
                df_ref=train_df,
                load_processed_data=ref_processed_path,
                ref_cif_column='CIF',
                ltol=0.2,
                stol=0.3,
                angle_tol=5.0,
                structures=structures,
                workers=48,
                is_unique_filter=True
            )
            
            # Save the updated dataframe
            output_path = df_path
            df_gen_with_novelty.to_parquet(output_path)
            print(f"Saved updated dataframe with novelty to: {output_path}")
            
            # Print novelty statistics
            novelty_rate = df_gen_with_novelty['is_novel'].mean()
            print(f"Novelty rate for {name}: {novelty_rate:.3f} ({df_gen_with_novelty['is_novel'].sum()}/{len(df_gen_with_novelty)})")

        print(f"\n{'='*50}")
        print("All dataframes processed successfully!")
        print(f"{'='*50}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()