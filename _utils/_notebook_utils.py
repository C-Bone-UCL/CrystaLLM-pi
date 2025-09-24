"""
Random utility functions used in the jupyter notebooks to avoid overcluttering them with code.
"""

import os
from pathlib import Path
import pandas as pd

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

def get_metrics(df, n_test, only_matched=False):
    mean_rmsd = df['RMS-d'].mean()
    n_matches = df['RMS-d'].notna().sum()
    percent_match = n_matches / n_test * 100
    # a_mae is MAE of True a vs Gen a (absolute values)
    # abs(true_a - gen_a) is diff
    if only_matched:
        df = df[df['RMS-d'].notna()]
        
    a_mae = df['True a'].sub(df['Gen a']).abs().mean()
    b_mae = df['True b'].sub(df['Gen b']).abs().mean()
    c_mae = df['True c'].sub(df['Gen c']).abs().mean()
    vol_mae = df['True volume'].sub(df['Gen volume']).abs().mean()
    # also compute pearsons correlation
    a_corr = df[['True a', 'Gen a']].corr().iloc[0, 1]
    b_corr = df[['True b', 'Gen b']].corr().iloc[0, 1]
    c_corr = df[['True c', 'Gen c']].corr().iloc[0, 1]
    vol_corr = df[['True volume', 'Gen volume']].corr().iloc[0, 1]
    
    metrics = {
        'Number of matched structures': n_matches,
        'Total number of structures': n_test,
        'Mean RMS-d': mean_rmsd,
        'Percent Matched (%)': percent_match,
        'a MAE': a_mae,
        'b MAE': b_mae,
        'c MAE': c_mae,
        'Volume MAE': vol_mae,
        'a R^2': a_corr,
        'b R^2': b_corr,
        'c R^2': c_corr,
        'Volume R^2': vol_corr
    }

    # print metrics nicely
    print(F"Number of matched structures: {n_matches} / {n_test}")
    print(f"Mean RMS-d: {mean_rmsd:.4f}")
    print(f"Percent Matched (%): {percent_match:.2f}% ({n_matches}/{n_test})")
    print(f"a MAE: {a_mae:.4f}")
    print(f"b MAE: {b_mae:.4f}")
    print(f"c MAE: {c_mae:.4f}")
    print(f"Volume MAE: {vol_mae:.4f}")
    print(f"a R^2: {a_corr:.4f}")
    print(f"b R^2: {b_corr:.4f}")
    print(f"c R^2: {c_corr:.4f}")
    print(f"Volume R^2: {vol_corr:.4f}")
    return metrics
