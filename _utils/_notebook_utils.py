"""
Random utility functions used in the jupyter notebooks to avoid overcluttering them with code.
"""

import os
from pathlib import Path
import pandas as pd

from _tokenizer import CustomCIFTokenizer

def filter_df_to_context(
    df: pd.DataFrame,
    context: int = 1024,
    cif_column: str = "CIF"
) -> pd.DataFrame:
    
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir='HF-cif-tokenizer',
        pad_token="<pad>"
        )

    mask = df[cif_column].fillna("").apply(lambda x: len(tokenizer.tokenize(x)) <= context)
    return df.loc[mask].reset_index(drop=True)

def build_challenge_dataframe(input_folder: str, output_parquet: str = "materials.parquet"):
    records = []
    input_path = Path(input_folder)
    
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            # Find .cif file
            cif_files = list(subfolder.glob("*.cif"))
            if not cif_files:
                continue
            cif_path = cif_files[0]
            material_id = cif_path.stem
            
            # Read CIF content
            true_cif = cif_path.read_text()
            
            records.append({
                "Material ID": material_id,
                "CIF": true_cif
            })
    
    df = pd.DataFrame.from_records(records, columns=["Material ID", "CIF"])
    
    df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(df)} records to {output_parquet}")