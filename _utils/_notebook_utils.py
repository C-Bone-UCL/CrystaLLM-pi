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

def get_metrics_xrd(df, n_test, only_matched=False, verbose=True):
    mean_rmsd = df['RMS-d'].mean()
    n_matches = df['RMS-d'].notna().sum()
    percent_match = n_matches / n_test * 100
    # a_mae is MAE of True a vs Gen a (absolute values)
    # abs(true_a - gen_a) is diff
    
    if only_matched:
        df = df[df['RMS-d'].notna()]
        if verbose:
            print(f"Computing metrics only on matched structures ({len(df)} entries)")

    else:
        if verbose:
            print(f"Computing metrics on all (also unmatched) structures ({len(df)} entries, {n_matches} matched)")

    a_mae = df['True a'].sub(df['Gen a']).abs().mean()
    b_mae = df['True b'].sub(df['Gen b']).abs().mean()
    c_mae = df['True c'].sub(df['Gen c']).abs().mean()
    vol_mae = df['True volume'].sub(df['Gen volume']).abs().mean()
    # also compute pearsons correlation
    a_corr = df[['True a', 'Gen a']].corr().iloc[0, 1]
    b_corr = df[['True b', 'Gen b']].corr().iloc[0, 1]
    c_corr = df[['True c', 'Gen c']].corr().iloc[0, 1]
    vol_corr = df[['True volume', 'Gen volume']].corr().iloc[0, 1]

    average_score = df['Score'].mean() if 'Score' in df.columns else float('nan')
    
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
        'Volume R^2': vol_corr,
        'Average Score': average_score
    }

    if verbose:
        print(f"Number of matched structures: {n_matches} / {n_test}")
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
        if 'Score' in df.columns:
            print(f"Average Score: {average_score:.4f}")

    return metrics

def get_metrics_ptnd_vs_scratch(
    df_dict, *,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    target_bg_col="target_Bandgap (eV)",
    cond_col="target_Bandgap (eV)",
    hit_tol_eV=0.2
):
    """Extract metrics from pretraining vs scratch comparison analysis."""
    import numpy as np
    from scipy.stats import gaussian_kde
    
    def _method_pretrain(lbl):
        l = lbl.lower()
        if   "slider"  in l: m = "Slider"
        elif "pkv"     in l: m = "PKV"
        elif "prepend" in l: m = "Prepend"
        else:                m = "Raw"
        return m, ("scratch" not in l)
    
    # Gather all condition values
    conds = set()
    for d in df_dict.values():
        if cond_col in d.columns:
            conds.update(d[cond_col].dropna().unique())
    conds = sorted(map(float, conds))
    
    # Build summary rows
    rows = []
    for lbl, df in df_dict.items():
        meth, pre = _method_pretrain(lbl)
        if cond_col not in df.columns:
            df = df.assign(**{cond_col: [np.nan]*len(df)})
        
        tau = 0.157
        eh = pd.to_numeric(df["ehull_mace_mp"], errors="coerce")
        df["is_stable_mace"] = eh.le(tau)
        
        for c in conds or [np.nan]:
            sub = df[df[cond_col] == c].copy() if not pd.isna(c) else df.copy()
            n = len(sub)
            if n == 0:
                rows.append({"m":meth,"p":pre,"x":c,"n":0,"hit":np.nan,"valid":np.nan,"q":np.nan})
                continue
            
            # Hit rate calculation
            hit = np.nan
            if {pred_bg_col,target_bg_col}.issubset(sub.columns):
                pred = pd.to_numeric(sub[pred_bg_col],errors="coerce")
                true = pd.to_numeric(sub[target_bg_col],errors="coerce")
                mask = (pred-true).abs() <= hit_tol_eV
                hit = mask.mean()
            
            # Valid fraction
            valid = sub["is_valid"].mean() if "is_valid" in sub.columns else np.nan
            
            # Quality metric Q
            q = np.nan
            req = {"is_valid","is_unique","is_novel","is_stable_mace"}
            if req.issubset(sub.columns):
                mask_q = sub["is_valid"] & sub["is_unique"] & sub["is_novel"] & sub["is_stable_mace"]
                q = mask_q.mean()
            rows.append({"m":meth,"p":pre,"x":c,"n":n,"hit":hit,"valid":valid,"q":q})
    
    met = pd.DataFrame(rows)
    
    # Compute KDE density if training data available
    dens_interp = None
    if train_df is not None and train_bg_col in train_df.columns and len(train_df)>50:
        vals = pd.to_numeric(train_df[train_bg_col],errors="coerce").dropna()
        kde = gaussian_kde(vals,bw_method="scott")
        xs = np.linspace(vals.min(),vals.max(),400)
        dens = kde(xs)
        dens_norm = dens/dens.max()
        dens_interp = np.interp(met["x"].astype(float), xs, dens_norm)
        met["density"] = dens_interp
    
    # Compute metrics
    met = met[met["n"] > 0].copy()
    met["hit_rate"] = met["hit"]
    
    metrics = {}
    
    # Hit-rate vs density correlation
    if "density" in met.columns:
        d = met["density"]
        a_h, b_h = np.polyfit(d, met["hit_rate"], 1)
        r_hit_density = np.corrcoef(d, met["hit_rate"])[0,1]
        metrics["hit_rate_vs_density_slope"] = a_h
        metrics["hit_rate_vs_density_intercept"] = b_h
        metrics["hit_rate_density_correlation"] = r_hit_density
    
    # Average differences (pretrained - scratch)
    diffs_valid = []
    diffs_q = []
    diffs_hit = []
    
    for m in met.m.unique():
        grp = met[met.m == m]
        
        # Valid differences
        vp = grp[grp.p]["valid"].mean()
        vs = grp[~grp.p]["valid"].mean()
        if not np.isnan(vp) and not np.isnan(vs):
            diffs_valid.append(vp - vs)
        
        # Quality differences
        qp = grp[grp.p]["q"].mean()
        qs = grp[~grp.p]["q"].mean()
        if not np.isnan(qp) and not np.isnan(qs):
            diffs_q.append(qp - qs)
        
        # Hit rate differences
        hp = grp[grp.p]["hit_rate"].mean()
        hs = grp[~grp.p]["hit_rate"].mean()
        if not np.isnan(hp) and not np.isnan(hs):
            diffs_hit.append(hp - hs)
    
    if diffs_valid:
        metrics["avg_delta_validity"] = np.mean(diffs_valid)
        metrics["sem_delta_validity"] = np.std(diffs_valid, ddof=1) / np.sqrt(len(diffs_valid))
    
    if diffs_q:
        metrics["avg_delta_quality"] = np.mean(diffs_q)
        metrics["sem_delta_quality"] = np.std(diffs_q, ddof=1) / np.sqrt(len(diffs_q))
    
    if diffs_hit:
        metrics["avg_delta_hit_rate"] = np.mean(diffs_hit)
        metrics["sem_delta_hit_rate"] = np.std(diffs_hit, ddof=1) / np.sqrt(len(diffs_hit))
    
    # Best/worst ratio at bandgap around 6-7 eV
    candidates = [c for c in conds if 6.0 <= c < 7.0]
    best_ratio = None
    best_c = None
    for c in candidates:
        vals = met[met.x == c]["hit_rate"].dropna()
        if len(vals) > 1 and vals.min() > 0:
            ratio = vals.max() / vals.min()
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_c = c
    
    if best_ratio is not None:
        metrics["best_worst_ratio_6_7eV"] = best_ratio
        metrics["best_worst_ratio_bandgap"] = best_c
    
    # Density correlations for valid and quality
    if "density" in met.columns:
        d = met["density"]
        a_v, b_v = np.polyfit(d, met["valid"], 1)
        a_q2, b_q2 = np.polyfit(d, met["q"], 1)
        r_v = np.corrcoef(d, met["valid"])[0,1]
        r_q = np.corrcoef(d, met["q"])[0,1]
        
        metrics["valid_vs_density_slope"] = a_v
        metrics["valid_vs_density_intercept"] = b_v
        metrics["valid_density_correlation"] = r_v
        metrics["quality_vs_density_slope"] = a_q2
        metrics["quality_vs_density_intercept"] = b_q2
        metrics["quality_density_correlation"] = r_q
    
    # Best performing methods
    for metric in ("hit_rate", "valid", "q"):
        means = met.groupby(["m", "p"])[metric].mean().dropna()
        if len(means) >= 2:
            top = means.nlargest(2)
            (m1, p1), (m2, p2) = top.index[0], top.index[1]
            v1, v2 = top.iloc[0], top.iloc[1]
            
            metrics[f"best_{metric}_method"] = m1
            metrics[f"best_{metric}_pretrained"] = p1
            metrics[f"best_{metric}_value"] = v1
            metrics[f"second_best_{metric}_method"] = m2
            metrics[f"second_best_{metric}_pretrained"] = p2
            metrics[f"second_best_{metric}_value"] = v2
            
            pct_improvement = (v1 - v2) * 100 if v2 > 0 else np.nan
            metrics[f"{metric}_improvement_percent"] = pct_improvement
    
    # print metrics nicely
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    # Additional print statements to match plotting.py output format
    if "hit_rate_vs_density_slope" in metrics:
        print(f"Hit-rate vs density: hit_rate = {metrics['hit_rate_vs_density_slope']:.3f}·density + {metrics['hit_rate_vs_density_intercept']:.3f}")
        print(f"Pearson's r = {metrics['hit_rate_density_correlation']:.3f} (density vs hit_rate)")
    
    if "avg_delta_validity" in metrics:
        print(f"Avg Δ validity (pretrained-scratch): {metrics['avg_delta_validity']:.3f} ± {metrics['sem_delta_validity']:.3f} (SEM)")
    
    if "avg_delta_quality" in metrics:
        print(f"Avg Δ quality  (pretrained-scratch): {metrics['avg_delta_quality']:.3f} ± {metrics['sem_delta_quality']:.3f} (SEM)")
    
    if "avg_delta_hit_rate" in metrics:
        print(f"Avg Δ hit_rate (pretrained-scratch): {metrics['avg_delta_hit_rate']:.3f} ± {metrics['sem_delta_hit_rate']:.3f} (SEM)")
    
    if "best_worst_ratio_6_7eV" in metrics:
        print(f"At {metrics['best_worst_ratio_bandgap']:.2f} eV, best/worst hit_rate = {metrics['best_worst_ratio_6_7eV']:.2f}×")
    
    if "valid_vs_density_slope" in metrics:
        print(f"Valid vs density: valid = {metrics['valid_vs_density_slope']:.3f}·density + {metrics['valid_vs_density_intercept']:.3f}")
        print(f"Quality vs density: Q = {metrics['quality_vs_density_slope']:.3f}·density + {metrics['quality_vs_density_intercept']:.3f}")
        print(f"Pearson's r = {metrics['valid_density_correlation']:.3f} (density vs valid)")
        print(f"Pearson's r = {metrics['quality_density_correlation']:.3f} (density vs quality)")
    
    # Best run outputs
    for metric in ("hit_rate", "valid", "q"):
        if f"best_{metric}_method" in metrics:
            m1 = metrics[f"best_{metric}_method"]
            p1 = metrics[f"best_{metric}_pretrained"]
            v1 = metrics[f"best_{metric}_value"]
            m2 = metrics[f"second_best_{metric}_method"]
            p2 = metrics[f"second_best_{metric}_pretrained"]
            v2 = metrics[f"second_best_{metric}_value"]
            pct = metrics[f"{metric}_improvement_percent"]
            name1 = f"{m1} ({'pretrained' if p1 else 'scratch'})"
            name2 = f"{m2} ({'pretrained' if p2 else 'scratch'})"
            print(f"Best run for mean {metric}: {name1} ({v1:.3f}), {pct:.1f}% above {name2} ({v2:.3f})")
    
    return metrics

def get_metrics_dataset_size_study(
    dfs_dict,
    train_df,
    *,
    train_col="Density (g/cm^3)", 
    target_col="target_Density (g/cm^3)",
    gen_col="gen_density (g/cm3)",
    targets=(1.2747, 15.30, 22.94)
):
    """Extract metrics from dataset size study analysis."""
    import numpy as np
    from scipy.stats import pearsonr, ttest_rel
    
    def _model(k: str) -> str:
        k = k.lower()
        if "pkv" in k:     return "PKV"
        if "slider" in k:  return "Slider"
        return "Prepend"
    
    def _size(k: str) -> str:
        for s in ("100k", "10k", "1k"):
            if f"-{s}" in k: return s
        return "full"
    
    def _subset(df: pd.DataFrame, tgt_col: str, tgt: float, gen_col: str):
        m = np.isclose(df[tgt_col], tgt, atol=1e-2) & df["is_valid"] & df[gen_col].notna()
        return pd.to_numeric(df.loc[m, gen_col], errors="coerce").dropna()
    
    # Build metrics for each target and dataset combination
    metrics_list = []
    for tgt in targets:
        for key, df in dfs_dict.items():
            m = _model(key)
            s = _size(key)
            vals = _subset(df, target_col, tgt, gen_col)
            if vals.empty:
                continue
            count = len(vals)
            mean_x = float(np.mean(vals.values))
            mae = np.mean(np.abs(vals.values - tgt))
            std = np.std(vals.values, ddof=0)
            metrics_list.append({
                "method": m, "size": s, "target": tgt,
                "count": count, "mean": mean_x, "mae": mae, "std": std
            })
    
    met_df = pd.DataFrame(metrics_list)
    
    # Clean size data
    if 'size' in met_df.columns:
        met_df['size'] = met_df['size'].astype(str).str.strip()
    
    size_map = {"full": len(train_df), "100k": 1e5, "10k": 1e4, "1k": 1e3}
    met_df["size_numeric"] = met_df["size"].map(size_map)
    
    # Calculate all metrics
    metrics = {}
    
    # Pearson correlations between valid count and dataset size
    for method in met_df.method.unique():
        sub = met_df[met_df.method == method]
        if len(sub) >= 2:
            r, p_val = pearsonr(sub["count"], sub["size_numeric"])
            metrics[f"{method}_count_vs_size_correlation"] = r
            metrics[f"{method}_count_vs_size_pvalue"] = p_val
    
    # Average MAE and SEM for each method
    for method in met_df.method.unique():
        vals = met_df[met_df.method == method]["mae"].values
        if len(vals) > 0:
            mean_mae = vals.mean()
            sem_mae = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            metrics[f"{method}_avg_mae"] = mean_mae
            metrics[f"{method}_sem_mae"] = sem_mae
    
    # Average standard deviation for each method
    for method in met_df.method.unique():
        vals = met_df[met_df.method == method]["std"].values
        if len(vals) > 0:
            mean_std = vals.mean()
            sem_std = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            metrics[f"{method}_avg_std"] = mean_std
            metrics[f"{method}_sem_std"] = sem_std
    
    # 1k dataset specific metrics
    df_1k = met_df[met_df['size'] == '1k']
    for method in ['Slider', 'PKV', 'Prepend']:
        method_counts = df_1k[df_1k['method'] == method]['count']
        if not method_counts.empty:
            mean_count = method_counts.mean()
            sem_count = method_counts.std(ddof=1) / np.sqrt(len(method_counts)) if len(method_counts) > 1 else 0.0
            metrics[f"{method}_1k_avg_valid_count"] = mean_count
            metrics[f"{method}_1k_sem_valid_count"] = sem_count
    
    # MAE for each method-size combination
    for method in met_df.method.unique():
        for size in ("1k", "10k", "100k", "full"):
            sub = met_df[(met_df.method == method) & (met_df.size == size)]
            if not sub.empty:
                vals = sub['mae'].values
                mean_mae = vals.mean()
                sem_mae = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
                metrics[f"{method}_{size}_mae"] = mean_mae
                metrics[f"{method}_{size}_mae_sem"] = sem_mae
    
    # T-tests comparing methods
    df_wide = met_df.pivot_table(index=["target", "size"], columns="method", values="mae").dropna()
    
    if {"Slider", "PKV"}.issubset(df_wide.columns):
        t_stat, p_val = ttest_rel(df_wide["Slider"], df_wide["PKV"])
        metrics["ttest_slider_vs_pkv_t"] = t_stat
        metrics["ttest_slider_vs_pkv_p"] = p_val
    
    if {"Prepend", "PKV"}.issubset(df_wide.columns):
        t_stat, p_val = ttest_rel(df_wide["Prepend"], df_wide["PKV"])
        metrics["ttest_prepend_vs_pkv_t"] = t_stat
        metrics["ttest_prepend_vs_pkv_p"] = p_val
    
    # Print statements to match plotting.py output format
    for method in met_df.method.unique():
        sub = met_df[met_df.method==method]
        if len(sub)>=2:
            r = metrics[f"{method}_count_vs_size_correlation"]
            print(f"{method}: Pearson r (valid count vs size) = {r:.3f}")
    
    for method in met_df.method.unique():
        if f"{method}_avg_mae" in metrics:
            mean_mae = metrics[f"{method}_avg_mae"]
            sem_mae = metrics[f"{method}_sem_mae"]
            print(f"{method}: Avg MAE = {mean_mae:.3f} ± {sem_mae:.3f}")
    
    for method in met_df.method.unique():
        if f"{method}_avg_std" in metrics:
            mean_std = metrics[f"{method}_avg_std"]
            sem_std = metrics[f"{method}_sem_std"]
            print(f"{method}: Avg std = {mean_std:.3f} ± {sem_std:.3f}")
    
    for method in ['Slider', 'PKV', 'Prepend']:
        if f"{method}_1k_avg_valid_count" in metrics:
            mean_count = metrics[f"{method}_1k_avg_valid_count"]
            sem_count = metrics[f"{method}_1k_sem_valid_count"]
            print(f"{method} (1k): Avg valid count = {mean_count:.1f} ± {sem_count:.1f}")
    
    for method in met_df.method.unique():
        for size in ("1k","10k","100k","full"):
            if f"{method}_{size}_mae" in metrics:
                mean_mae = metrics[f"{method}_{size}_mae"]
                sem_mae = metrics[f"{method}_{size}_mae_sem"]
                print(f"{method} ({size}): Avg MAE = {mean_mae:.3f} ± {sem_mae:.3f}")
    
    if "ttest_slider_vs_pkv_t" in metrics:
        t = metrics["ttest_slider_vs_pkv_t"]
        p = metrics["ttest_slider_vs_pkv_p"]
        print(f"t-test Slider vs PKV MAE: t={t:.3f}, p={p:.3f}")
    
    if "ttest_prepend_vs_pkv_t" in metrics:
        t = metrics["ttest_prepend_vs_pkv_t"]
        p = metrics["ttest_prepend_vs_pkv_p"]
        print(f"t-test Prepend vs PKV MAE: t={t:.3f}, p={p:.3f}")
    
    # Add raw dataframe for further analysis
    metrics["raw_dataframe"] = met_df
    
    return metrics

def process_xrd_to_condition_vector(file_content):
    # Parse the text data
    lines = file_content.strip().split('\n')
    
    # Skip header line
    data_lines = lines[1:] if len(lines) > 1 else lines
    
    peaks = []
    for line in data_lines:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    two_theta = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append({'two_theta': two_theta, 'intensity': intensity})
                except ValueError:
                    continue
    
    # Sort by intensity (descending) and take top 20
    top_k = 20
    peaks = sorted(peaks, key=lambda d: d['intensity'], reverse=True)[:top_k]
    
    # Extract theta and intensity values
    thetas = [d['two_theta'] for d in peaks]
    ints = [d['intensity'] for d in peaks]
    
    # Pad with -100 if fewer than top_k peaks
    thetas += [-100] * (top_k - len(thetas))
    ints += [-100] * (top_k - len(ints))
    
    # Define scaling ranges
    theta_min, theta_max = 0, 90
    
    # Find max intensity for normalization
    valid_intensities = [i for i in ints if i != -100]
    intensity_max = max(valid_intensities) if valid_intensities else 1
    
    # Scale theta values
    scaled_theta = [(t - theta_min) / (theta_max - theta_min) if t != -100 else -100 for t in thetas]
    scaled_theta = [round(t, 3) for t in scaled_theta]
    
    # Scale intensity values relative to max intensity
    scaled_int = [i / intensity_max if i != -100 else -100 for i in ints]
    scaled_int = [round(i, 3) for i in scaled_int]
    
    # Combine vectors (theta + intensity = 40 total values)
    vec = scaled_theta + scaled_int
    
    # Format as string
    vector_str = ",".join(map(str, vec))
    
    print("Theta scaled to [0,1] (0 to 90), Intensity scaled to [0,1] (relative to max in pattern), -100 for padding")
    
    return vector_str