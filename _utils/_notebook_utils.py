import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _utils._processing_utils import extract_reduced_formula as _shared_extract_reduced_formula
from pymatgen.core import Composition
from pymatgen.core import Element
from smact.data_loader import lookup_element_hhis


_XRD_PROPERTY_SPECS = (
    ("a", "True a", "Gen a"),
    ("b", "True b", "Gen b"),
    ("c", "True c", "Gen c"),
)


def _mean_abs_error(
    frame: pd.DataFrame,
    true_col: str,
    gen_col: str,
    *,
    allow_missing: bool = False,
):
    if allow_missing and (true_col not in frame.columns or gen_col not in frame.columns):
        return np.nan
    return frame[true_col].sub(frame[gen_col]).abs().mean()


def _r_squared(
    frame: pd.DataFrame,
    true_col: str,
    gen_col: str,
    *,
    allow_missing: bool = False,
):
    if allow_missing and (true_col not in frame.columns or gen_col not in frame.columns):
        return np.nan

    valid_pairs = frame[[true_col, gen_col]].dropna()
    if len(valid_pairs) <= 1:
        return np.nan
    return valid_pairs.corr().iloc[0, 1] ** 2


def _atom_count_stats(frame: pd.DataFrame) -> dict:
    if "atom_counts" not in frame.columns:
        return {}

    matched_counts = frame.loc[frame["RMS-d"].notna(), "atom_counts"].dropna()
    unmatched_counts = frame.loc[frame["RMS-d"].isna(), "atom_counts"].dropna()
    mean_matched = matched_counts.mean() if len(matched_counts) > 0 else np.nan
    mean_unmatched = unmatched_counts.mean() if len(unmatched_counts) > 0 else np.nan
    matched_se = matched_counts.sem() if len(matched_counts) > 1 else np.nan
    unmatched_se = unmatched_counts.sem() if len(unmatched_counts) > 1 else np.nan

    return {
        'Atom Count (matched mean)': mean_matched,
        'Atom Count (matched SE)': matched_se,
        'Atom Count (unmatched mean)': mean_unmatched,
        'Atom Count (unmatched SE)': unmatched_se,
        'Delta Atom Count (match - unmatch)': (
            mean_matched - mean_unmatched
            if not (np.isnan(mean_matched) or np.isnan(mean_unmatched))
            else np.nan
        ),
    }


def _build_xrd_property_metrics(
    frame: pd.DataFrame,
    *,
    allow_missing: bool,
    volume_label: str,
) -> dict:
    metrics = {}

    for label, true_col, gen_col in _XRD_PROPERTY_SPECS:
        metrics[f"{label} MAE"] = _mean_abs_error(
            frame,
            true_col,
            gen_col,
            allow_missing=allow_missing,
        )
    metrics[f"{volume_label} MAE"] = _mean_abs_error(
        frame,
        "True volume",
        "Gen volume",
        allow_missing=allow_missing,
    )

    for label, true_col, gen_col in _XRD_PROPERTY_SPECS:
        metrics[f"{label} R^2"] = _r_squared(
            frame,
            true_col,
            gen_col,
            allow_missing=allow_missing,
        )
    metrics[f"{volume_label} R^2"] = _r_squared(
        frame,
        "True volume",
        "Gen volume",
        allow_missing=allow_missing,
    )

    return metrics


def _material_summary_record(
    row: pd.Series,
    *,
    formula: str,
    position: int,
    metric_name: str,
    struct_nov: str,
    comp_nov: str,
) -> dict:
    return {
        'Reduced Formula': formula,
        'Position': position,
        'Metric': metric_name,
        'Pred. SLME': row['predicted_slme'],
        'HHI_p': row['HHI_p'],
        'HHI_r': row['HHI_r'],
        'HHI_dist': row['HHI_distance_to_0'],
        'E_hull_mace (eV/atom)': row['ehull_mace_mp'],
        'Structure Nov.': struct_nov,
        'Composition Nov.': comp_nov,
    }


def _sem(values, *, single_value=np.nan) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) > 1:
        return values.std(ddof=1) / np.sqrt(len(values))
    return single_value


def _fit_line_and_corr(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    return slope, intercept, corr


def _top_two_means(means: pd.Series):
    top = means.nlargest(2)
    if len(top) < 2:
        return None
    return (top.index[0], top.iloc[0]), (top.index[1], top.iloc[1])


def _draw_hist_series(ax, series, *, bins):
    for item in sorted(series, key=lambda entry: len(entry[0]), reverse=True):
        if len(item) == 3:
            data, color, alpha = item
            label = None
        else:
            data, color, alpha, label = item

        if len(data):
            hist_kwargs = {
                'bins': bins,
                'color': color,
                'edgecolor': 'none',
                'alpha': alpha,
            }
            if label is not None:
                hist_kwargs['label'] = label
            ax.hist(data, **hist_kwargs)
def build_challenge_dataframe(input_folder: str, output_csv: str = "materials.parquet"):
    records = []
    input_path = Path(input_folder)
    
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            cif_files = list(subfolder.glob("*.cif"))
            if not cif_files:
                continue
            cif_path = cif_files[0]
            material_id = cif_path.stem

            true_cif = cif_path.read_text()
            
            records.append({
                "Material ID": material_id,
                "CIF": true_cif
            })
    
    df = pd.DataFrame.from_records(records, columns=["Material ID", "CIF"])
    
    df.to_parquet(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

def get_metrics_xrd(df, n_test, only_matched=False, verbose=True):
    mean_rmsd = df['RMS-d'].mean()
    n_matches = df['RMS-d'].notna().sum()
    percent_match = n_matches / n_test * 100

    metrics_df = df
    
    if only_matched:
        metrics_df = metrics_df[metrics_df['RMS-d'].notna()]
        if verbose:
            print(f"Computing metrics only on matched, valid-system structures ({len(metrics_df)} entries)")

    else:
        if verbose:
            matched_in_subset = metrics_df['RMS-d'].notna().sum()
            print(f"Computing metrics on all valid-system structures ({len(metrics_df)} entries, {matched_in_subset} matched)")

    property_metrics = _build_xrd_property_metrics(
        metrics_df,
        allow_missing=False,
        volume_label='Volume',
    )

    average_score = metrics_df['Score'].mean() if 'Score' in metrics_df.columns else float('nan')
    
    metrics = {
        'Number of matched structures': n_matches,
        'Total number of structures': n_test,
        'Mean RMS-d': mean_rmsd,
        'Percent Matched (%)': percent_match,
        **property_metrics,
        'Average Score': average_score
    }

    if verbose:
        print("\n--- Top-Level Stats (Full Dataset) ---")
        print(f"Number of matched structures: {n_matches} / {n_test}")
        print(f"Mean RMS-d: {mean_rmsd:.4f}")
        print(f"Percent Matched (%): {percent_match:.2f}% ({n_matches}/{n_test})")
        
        print("\n--- Property Metrics (Valid-System Data) ---")
        for key, value in property_metrics.items():
            print(f"{key}: {value:.4f}")
        if 'Score' in metrics_df.columns:
            print(f"Average Score: {average_score:.4f}")

    return metrics

def get_stratified_metrics_xrd(df_metrics: pd.DataFrame, df_test: pd.DataFrame = None, only_matched: bool = False, verbose=True, n_test=None) -> pd.DataFrame:
    if df_test is not None:
        df = df_test.copy()
        if 'Material ID' in df.columns and 'Material ID' in df_metrics.columns:
            cols_to_use = df_metrics.columns.difference(df.columns).tolist() + ['Material ID']
            df = df.merge(df_metrics[cols_to_use], on='Material ID', how='left')
        else:
            cols_to_use = df_metrics.columns.difference(df.columns)
            df = df.join(df_metrics[cols_to_use], how='left')
    else:
        missing = [c for c in ('is_novel', 'is_comp_novel') if c not in df_metrics.columns]
        if missing:
            raise ValueError(
                f"df_test was not provided but df_metrics is missing columns: {missing}. "
                "Pass df_test, or run XRD_metrics.py with a ref_parquet that contains these columns."
            )
        df = df_metrics.copy()

    masks = {
        'Overall': pd.Series(True, index=df.index),
        'Memorized (Seen Comp & Struct)': ~df['is_novel'],
        'Structurally Novel (Seen Comp)': df['is_novel'] & ~df['is_comp_novel'],
        'Compositionally Novel (Unseen Comp)': df['is_comp_novel']
    }

    results = {}

    for tier_name, mask in masks.items():
        subset = df[mask].copy()
        tier_size = len(subset)
        
        if tier_size == 0:
            continue

        if isinstance(n_test, dict):
            denom = n_test.get(tier_name, tier_size)
        elif tier_name == 'Overall' and n_test is not None:
            denom = n_test
        else:
            denom = tier_size

        n_matches = subset['RMS-d'].notna().sum()
        percent_match = (n_matches / denom) * 100
        mean_rmsd = subset['RMS-d'].mean()
        average_score = subset['Score'].mean() if 'Score' in subset.columns else np.nan

        prop_df = subset[subset['RMS-d'].notna()] if only_matched else subset
        atom_stats = _atom_count_stats(subset)
        property_metrics = _build_xrd_property_metrics(
            prop_df,
            allow_missing=True,
            volume_label='Vol',
        )

        results[tier_name] = {
            'Total Structures': denom,
            'Matched': n_matches,
            'Match Rate (%)': percent_match,
            'Mean RMS-d': mean_rmsd,
            **property_metrics,
            'Avg Score': average_score,
            **atom_stats,
        }

    if verbose:
        for tier, metrics in results.items():
            print(f"\n--- Metrics for {tier} ---")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

    return pd.DataFrame(results).T

def get_metrics_ptnd_vs_scratch(
    df_dict, *,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    target_bg_col="target_Bandgap (eV)",
    cond_col="target_Bandgap (eV)",
    hit_tol_eV=0.2
):
    from scipy.stats import gaussian_kde
    
    def _method_pretrain(lbl):
        l = lbl.lower()
        if   "slider"  in l: m = "Slider"
        elif "pkv"     in l: m = "PKV"
        elif "prepend" in l: m = "Prepend"
        else:                m = "Raw"
        return m, ("scratch" not in l)
    
    conds = set()
    for d in df_dict.values():
        if cond_col in d.columns:
            conds.update(d[cond_col].dropna().unique())
    conds = sorted(map(float, conds))

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

            hit = np.nan
            if {pred_bg_col,target_bg_col}.issubset(sub.columns):
                pred = pd.to_numeric(sub[pred_bg_col],errors="coerce")
                true = pd.to_numeric(sub[target_bg_col],errors="coerce")
                mask = (pred-true).abs() <= hit_tol_eV
                hit = mask.mean()

            valid = sub["is_valid"].mean() if "is_valid" in sub.columns else np.nan

            q = np.nan
            req = {"is_valid","is_unique","is_novel","is_stable_mace"}
            if req.issubset(sub.columns):
                mask_q = sub["is_valid"] & sub["is_unique"] & sub["is_novel"] & sub["is_stable_mace"]
                q = mask_q.mean()
            rows.append({"m":meth,"p":pre,"x":c,"n":n,"hit":hit,"valid":valid,"q":q})
    
    met = pd.DataFrame(rows)

    dens_interp = None
    if train_df is not None and train_bg_col in train_df.columns and len(train_df)>50:
        vals = pd.to_numeric(train_df[train_bg_col],errors="coerce").dropna()
        kde = gaussian_kde(vals,bw_method="scott")
        xs = np.linspace(vals.min(),vals.max(),400)
        dens = kde(xs)
        dens_norm = dens/dens.max()
        dens_interp = np.interp(met["x"].astype(float), xs, dens_norm)
        met["density"] = dens_interp

    met = met[met["n"] > 0].copy()
    met["hit_rate"] = met["hit"]
    
    metrics = {}

    if "density" in met.columns:
        d = met["density"]
        a_h, b_h, r_hit_density = _fit_line_and_corr(d, met["hit_rate"])
        metrics["hit_rate_vs_density_slope"] = a_h
        metrics["hit_rate_vs_density_intercept"] = b_h
        metrics["hit_rate_density_correlation"] = r_hit_density

    diffs_valid = []
    diffs_q = []
    diffs_hit = []
    
    for m in met.m.unique():
        grp = met[met.m == m]

        vp = grp[grp.p]["valid"].mean()
        vs = grp[~grp.p]["valid"].mean()
        if not np.isnan(vp) and not np.isnan(vs):
            diffs_valid.append(vp - vs)

        qp = grp[grp.p]["q"].mean()
        qs = grp[~grp.p]["q"].mean()
        if not np.isnan(qp) and not np.isnan(qs):
            diffs_q.append(qp - qs)

        hp = grp[grp.p]["hit_rate"].mean()
        hs = grp[~grp.p]["hit_rate"].mean()
        if not np.isnan(hp) and not np.isnan(hs):
            diffs_hit.append(hp - hs)
    
    if diffs_valid:
        metrics["avg_delta_validity"] = np.mean(diffs_valid)
        metrics["sem_delta_validity"] = _sem(diffs_valid)
    
    if diffs_q:
        metrics["avg_delta_quality"] = np.mean(diffs_q)
        metrics["sem_delta_quality"] = _sem(diffs_q)
    
    if diffs_hit:
        metrics["avg_delta_hit_rate"] = np.mean(diffs_hit)
        metrics["sem_delta_hit_rate"] = _sem(diffs_hit)

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

    if "density" in met.columns:
        d = met["density"]
        a_v, b_v, r_v = _fit_line_and_corr(d, met["valid"])
        a_q2, b_q2, r_q = _fit_line_and_corr(d, met["q"])
        
        metrics["valid_vs_density_slope"] = a_v
        metrics["valid_vs_density_intercept"] = b_v
        metrics["valid_density_correlation"] = r_v
        metrics["quality_vs_density_slope"] = a_q2
        metrics["quality_vs_density_intercept"] = b_q2
        metrics["quality_density_correlation"] = r_q

    for metric in ("hit_rate", "valid", "q"):
        means = met.groupby(["m", "p"])[metric].mean().dropna()
        top_two = _top_two_means(means)
        if top_two is not None:
            ((m1, p1), v1), ((m2, p2), v2) = top_two
            
            metrics[f"best_{metric}_method"] = m1
            metrics[f"best_{metric}_pretrained"] = p1
            metrics[f"best_{metric}_value"] = v1
            metrics[f"second_best_{metric}_method"] = m2
            metrics[f"second_best_{metric}_pretrained"] = p2
            metrics[f"second_best_{metric}_value"] = v2
            
            pct_improvement = (v1 - v2) * 100 if v2 > 0 else np.nan
            metrics[f"{metric}_improvement_percent"] = pct_improvement

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    extra_line_groups = (
        (
            "hit_rate_vs_density_slope",
            (
                "Hit-rate vs density: hit_rate = {hit_rate_vs_density_slope:.3f}*density + {hit_rate_vs_density_intercept:.3f}",
                "Pearson's r = {hit_rate_density_correlation:.3f} (density vs hit_rate)",
            ),
        ),
        (
            "avg_delta_validity",
            ("Avg delta validity (pretrained-scratch): {avg_delta_validity:.3f} +/- {sem_delta_validity:.3f} (SEM)",),
        ),
        (
            "avg_delta_quality",
            ("Avg delta quality  (pretrained-scratch): {avg_delta_quality:.3f} +/- {sem_delta_quality:.3f} (SEM)",),
        ),
        (
            "avg_delta_hit_rate",
            ("Avg delta hit_rate (pretrained-scratch): {avg_delta_hit_rate:.3f} +/- {sem_delta_hit_rate:.3f} (SEM)",),
        ),
        (
            "best_worst_ratio_6_7eV",
            ("At {best_worst_ratio_bandgap:.2f} eV, best/worst hit_rate = {best_worst_ratio_6_7eV:.2f}x",),
        ),
        (
            "valid_vs_density_slope",
            (
                "Valid vs density: valid = {valid_vs_density_slope:.3f}*density + {valid_vs_density_intercept:.3f}",
                "Quality vs density: Q = {quality_vs_density_slope:.3f}*density + {quality_vs_density_intercept:.3f}",
                "Pearson's r = {valid_density_correlation:.3f} (density vs valid)",
                "Pearson's r = {quality_density_correlation:.3f} (density vs quality)",
            ),
        ),
    )

    for gate_key, lines in extra_line_groups:
        if gate_key in metrics:
            for line in lines:
                print(line.format(**metrics))

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

    if 'size' in met_df.columns:
        met_df['size'] = met_df['size'].astype(str).str.strip()
    
    size_map = {"full": len(train_df), "100k": 1e5, "10k": 1e4, "1k": 1e3}
    met_df["size_numeric"] = met_df["size"].map(size_map)

    metrics = {}

    for method in met_df.method.unique():
        sub = met_df[met_df.method == method]
        if len(sub) >= 2:
            r, p_val = pearsonr(sub["count"], sub["size_numeric"])
            metrics[f"{method}_count_vs_size_correlation"] = r
            metrics[f"{method}_count_vs_size_pvalue"] = p_val

    for method in met_df.method.unique():
        vals = met_df[met_df.method == method]["mae"].values
        if len(vals) > 0:
            mean_mae = vals.mean()
            sem_mae = _sem(vals)
            metrics[f"{method}_avg_mae"] = mean_mae
            metrics[f"{method}_sem_mae"] = sem_mae

    for method in met_df.method.unique():
        vals = met_df[met_df.method == method]["std"].values
        if len(vals) > 0:
            mean_std = vals.mean()
            sem_std = _sem(vals)
            metrics[f"{method}_avg_std"] = mean_std
            metrics[f"{method}_sem_std"] = sem_std

    df_1k = met_df[met_df['size'] == '1k']
    for method in ['Slider', 'PKV', 'Prepend']:
        method_counts = df_1k[df_1k['method'] == method]['count']
        if not method_counts.empty:
            mean_count = method_counts.mean()
            sem_count = _sem(method_counts.values, single_value=0.0)
            metrics[f"{method}_1k_avg_valid_count"] = mean_count
            metrics[f"{method}_1k_sem_valid_count"] = sem_count

    for method in met_df.method.unique():
        for size in ("1k", "10k", "100k", "full"):
            sub = met_df[(met_df.method == method) & (met_df.size == size)]
            if not sub.empty:
                vals = sub['mae'].values
                mean_mae = vals.mean()
                sem_mae = _sem(vals)
                metrics[f"{method}_{size}_mae"] = mean_mae
                metrics[f"{method}_{size}_mae_sem"] = sem_mae

    df_wide = met_df.pivot_table(index=["target", "size"], columns="method", values="mae").dropna()
    
    if {"Slider", "PKV"}.issubset(df_wide.columns):
        t_stat, p_val = ttest_rel(df_wide["Slider"], df_wide["PKV"])
        metrics["ttest_slider_vs_pkv_t"] = t_stat
        metrics["ttest_slider_vs_pkv_p"] = p_val
    
    if {"Prepend", "PKV"}.issubset(df_wide.columns):
        t_stat, p_val = ttest_rel(df_wide["Prepend"], df_wide["PKV"])
        metrics["ttest_prepend_vs_pkv_t"] = t_stat
        metrics["ttest_prepend_vs_pkv_p"] = p_val

    for method in met_df.method.unique():
        sub = met_df[met_df.method==method]
        if len(sub)>=2:
            r = metrics[f"{method}_count_vs_size_correlation"]
            print(f"{method}: Pearson r (valid count vs size) = {r:.3f}")

    for label, mean_suffix, sem_suffix in (
        ("Avg MAE", "avg_mae", "sem_mae"),
        ("Avg std", "avg_std", "sem_std"),
    ):
        for method in met_df.method.unique():
            mean_key = f"{method}_{mean_suffix}"
            sem_key = f"{method}_{sem_suffix}"
            if mean_key in metrics:
                print(f"{method}: {label} = {metrics[mean_key]:.3f} +/- {metrics[sem_key]:.3f}")
    
    for method in ['Slider', 'PKV', 'Prepend']:
        if f"{method}_1k_avg_valid_count" in metrics:
            mean_count = metrics[f"{method}_1k_avg_valid_count"]
            sem_count = metrics[f"{method}_1k_sem_valid_count"]
            print(f"{method} (1k): Avg valid count = {mean_count:.1f} +/- {sem_count:.1f}")
    
    for method in met_df.method.unique():
        for size in ("1k","10k","100k","full"):
            if f"{method}_{size}_mae" in metrics:
                mean_mae = metrics[f"{method}_{size}_mae"]
                sem_mae = metrics[f"{method}_{size}_mae_sem"]
                print(f"{method} ({size}): Avg MAE = {mean_mae:.3f} +/- {sem_mae:.3f}")
    
    if "ttest_slider_vs_pkv_t" in metrics:
        t = metrics["ttest_slider_vs_pkv_t"]
        p = metrics["ttest_slider_vs_pkv_p"]
        print(f"t-test Slider vs PKV MAE: t={t:.3f}, p={p:.3f}")
    
    if "ttest_prepend_vs_pkv_t" in metrics:
        t = metrics["ttest_prepend_vs_pkv_t"]
        p = metrics["ttest_prepend_vs_pkv_p"]
        print(f"t-test Prepend vs PKV MAE: t={t:.3f}, p={p:.3f}")

    metrics["raw_dataframe"] = met_df
    
    return metrics

def process_xrd_to_condition_vector(file_content):
    lines = file_content.strip().split('\n')
    data_lines = lines[1:] if len(lines) > 1 else lines

    peaks = []
    for line in data_lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    two_theta = float(parts[0])
                    intensity = float(parts[1])
                    peaks.append({'two_theta': two_theta, 'intensity': intensity})
                except ValueError:
                    continue

    top_k = 20
    peaks = sorted(peaks, key=lambda d: d['intensity'], reverse=True)[:top_k]

    thetas = [d['two_theta'] for d in peaks]
    ints = [d['intensity'] for d in peaks]

    thetas += [-100] * (top_k - len(thetas))
    ints += [-100] * (top_k - len(ints))

    theta_min, theta_max = 0, 90

    valid_intensities = [i for i in ints if i != -100]
    intensity_max = max(valid_intensities) if valid_intensities else 1

    scaled_theta = [(t - theta_min) / (theta_max - theta_min) if t != -100 else -100 for t in thetas]
    scaled_theta = [round(t, 3) for t in scaled_theta]

    scaled_int = [i / intensity_max if i != -100 else -100 for i in ints]
    scaled_int = [round(i, 3) for i in scaled_int]

    vec = scaled_theta + scaled_int

    vector_str = ",".join(map(str, vec))

    print("Theta scaled to [0,1] (0 to 90), Intensity scaled to [0,1] (relative to max in pattern), -100 for padding")

    return vector_str

def extract_reduced_formula(cif_str):
    return _shared_extract_reduced_formula(cif_str)


def parse_formula(formula_string: str) -> dict:
    comp = Composition(formula_string)
    return comp.get_el_amt_dict()


def hhi_scores_from_formula(formula: dict):
    masses = []
    hhi_p_vals = []
    hhi_r_vals = []

    for symbol, count in formula.items():
        atomic_weight = Element(symbol).atomic_mass
        hhi_data = lookup_element_hhis(symbol)

        if hhi_data is None:
            print(f"Warning: No HHI data for element {symbol} (excluded by source paper). "
                  f"Cannot calculate HHI for material '{formula}'.", file=sys.stderr)
            return None, None

        hhi_p, hhi_r = hhi_data
        
        if hhi_p is None or hhi_r is None:
            print(f"Warning: Incomplete HHI data for element {symbol}. "
                  f"Cannot calculate HHI for material '{formula}'.", file=sys.stderr)
            return None, None

        masses.append(count * atomic_weight)
        hhi_p_vals.append(hhi_p)
        hhi_r_vals.append(hhi_r)

    total_mass = sum(masses)
    
    if total_mass == 0:
        return None, None
        
    hhi_p_material = sum((m / total_mass) * h for m, h in zip(masses, hhi_p_vals))
    hhi_r_material = sum((m / total_mass) * h for m, h in zip(masses, hhi_r_vals))

    return hhi_p_material, hhi_r_material


def get_hhi_scores_from_cif(cif_str):
    try:
        formula_str = extract_reduced_formula(cif_str)
        formula_dict = parse_formula(formula_str)
        scores = hhi_scores_from_formula(formula_dict)
    except Exception as e:
        print(f"Error processing CIF: {e}", file=sys.stderr)
        print(f"Problematic CIF:\n{cif_str}\n", file=sys.stderr)
        scores = (None, None)
    return scores


def extract_formula(cif_str):
    try:
        return _shared_extract_reduced_formula(cif_str)
    except Exception:
        return "UnknownFormula"


def parse_novelty_from_tag(tag):
    struct_nov = []
    comp_nov = []
    
    if "Novel-ft-pt" in tag:
        struct_nov = ["ft", "pt"]
    elif "Novel-ft" in tag:
        struct_nov = ["ft"]
    elif "Novel-pt" in tag:
        struct_nov = ["pt"]
    
    if "CompNovel-ft-pt" in tag:
        comp_nov = ["ft", "pt"]
    elif "CompNovel-ft" in tag:
        comp_nov = ["ft"]
    elif "CompNovel-pt" in tag:
        comp_nov = ["pt"]

    struct_str = "-".join(struct_nov) if struct_nov else "None"
    comp_str = "-".join(comp_nov) if comp_nov else "None"

    return struct_str, comp_str


def build_novelty_tag(row):
    tags = []

    if row['is_novel'] and row['is_novel_pt']:
        tags.append("Novel-ft-pt")
    elif row['is_novel']:
        tags.append("Novel-ft")
    elif row['is_novel_pt']:
        tags.append("Novel-pt")

    if row['is_comp_novel'] and row['is_comp_novel_pt']:
        tags.append("CompNovel-ft-pt")
    elif row['is_comp_novel']:
        tags.append("CompNovel-ft")
    elif row['is_comp_novel_pt']:
        tags.append("CompNovel-pt")

    return "__".join(tags) if tags else "NotNovel"


def select_top_materials(df, top_n_slme=15, top_n_sustain=15, slme_threshold=25):
    materials = {}
    summary_records = []

    selection_specs = (
        ('SLME', df.nlargest(top_n_slme, 'predicted_slme'), 'SLME-{slme}'),
        (
            'HHI-SLME',
            df[df['predicted_slme'] > slme_threshold].nsmallest(top_n_sustain, 'HHI_distance_to_0'),
            'Sustain-SLME-{slme}',
        ),
    )

    for metric_name, selected_df, name_template in selection_specs:
        for position, (_, row) in enumerate(selected_df.iterrows(), start=1):
            formula = extract_formula(row['Generated CIF'])
            novelty = build_novelty_tag(row)
            material_name = (
                f"{formula}__{name_template.format(slme=int(row['predicted_slme']))}"
                f"_top_{position}__{novelty}"
            )
            materials[material_name] = row['Generated CIF']

            struct_nov, comp_nov = parse_novelty_from_tag(novelty)
            summary_records.append(
                _material_summary_record(
                    row,
                    formula=formula,
                    position=position,
                    metric_name=metric_name,
                    struct_nov=struct_nov,
                    comp_nov=comp_nov,
                )
            )
    
    summary_df = pd.DataFrame(summary_records)
    return materials, summary_df


def export_materials(materials, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for name, cif_str in materials.items():
        filepath = os.path.join(output_dir, f"{name}.cif")
        with open(filepath, 'w') as f:
            f.write(cif_str)
    
    print(f"Exported {len(materials)} materials to {output_dir}/")


def run_material_selection(input_parquet, output_dir, output_csv, top_n_slme=15, top_n_sustain=15):
    df = pd.read_parquet(input_parquet)
    materials, summary_df = select_top_materials(df, top_n_slme, top_n_sustain)
    
    export_materials(materials, output_dir)

    summary_df.to_csv(output_csv, index=False)
    print(f"Saved summary dataframe to {output_csv}")


def load_count_datasets(datasets: list[tuple]) -> list[tuple]:
    loaded = []
    for name, path, ctx in datasets:
        if path is None:
            print(f"[SKIP] {name}  (count parquet not yet generated)")
            loaded.append((name, None, ctx))
            continue
        df = pd.read_parquet(path)
        pct_over = 100 * (df["token_count"] > ctx).mean()
        print(f"{name},  n={len(df):>7,},  ctx={ctx}  {pct_over:.1f}% beyond context")
        loaded.append((name, df, ctx))
    return loaded


def _extract_atom_counts_worker(cif_text: str) -> tuple:
    try:
        import warnings
        from pymatgen.core import Structure
        from _utils._generating.postprocess import postprocess

        cleaned = postprocess(cif_text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_str(cleaned, fmt="cif")
        return len(struct), len(struct.get_primitive_structure())
    except Exception:
        return None, None


def compute_atom_counts(loaded: list[tuple], num_workers: int = 32) -> list[tuple]:
    import concurrent.futures
    from tqdm import tqdm

    updated = []
    for name, df, ctx in loaded:
        if df is None:
            updated.append((name, None, ctx))
            continue

        if "conv_count" in df.columns and "prim_count" in df.columns:
            print(f"{name}: atom counts already cached ({len(df):,} structures)")
            updated.append((name, df, ctx))
            continue

        cifs = df["CIF"].tolist()
        print(f"{name}: parsing {len(cifs):,} CIFs with {num_workers} workers")
        chunksize = max(1, len(cifs) // (num_workers * 4))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_extract_atom_counts_worker, cifs, chunksize=chunksize),
                total=len(cifs),
                desc="Parsing geometries",
            ))

        df = df.copy()
        df["conv_count"] = pd.to_numeric([r[0] for r in results], errors="coerce")
        df["prim_count"] = pd.to_numeric([r[1] for r in results], errors="coerce")

        n_ok = df["conv_count"].notna().sum()
        print(f"  {n_ok:,} / {len(df):,} parsed successfully")
        print(f"  re-save your parquet to cache atom counts for next run")
        updated.append((name, df, ctx))

    return updated


def plot_dataset_stats(loaded: list[tuple], atom_density_line: int = 20, save_dir: str="plots/") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    C_CONV_WITHIN = "#E69F00"
    C_PRIM_WITHIN = "#009E73"
    C_CONV_ABOVE  = "#CC4125"
    C_PRIM_ABOVE  = "#FF9980"

    AXES_LW, LABEL_FS, TICK_FS, ANNOT_FS = 1.2, 13, 11, 10

    def _style_ax(ax):
        for s in ("top", "right"):   ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_linewidth(AXES_LW)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    os.makedirs(save_dir, exist_ok=True)

    for name, df, ctx in loaded:
        if df is None:
            print(f"[SKIP] {name}")
            continue

        tc = df["token_count"]
        n_over   = int((tc > ctx).sum())
        pct_over = 100.0 * n_over / len(tc)

        fig, (ax_tok, ax_atom) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        fig.suptitle(name, fontsize=LABEL_FS + 1, fontweight="bold")

        tok_bins = np.histogram_bin_edges(tc, bins=80) 
        tok_series = [
            (tc[tc <= ctx], C_PRIM_WITHIN, 0.85),
            (tc[tc >  ctx], C_CONV_ABOVE,  0.85),
        ]
        _draw_hist_series(ax_tok, tok_series, bins=tok_bins)

        ax_tok.axvline(ctx, color="black", linestyle="--", linewidth=1.4)
        xmin, xmax = ax_tok.get_xlim()
        _, ymax = ax_tok.get_ylim()
        ax_tok.text(ctx - (xmax - xmin) * 0.02, ymax * 0.96,
                    f"ctx = {ctx:,}", color="black", fontsize=ANNOT_FS, ha="right", va="top")
        ax_tok.text(ctx + (xmax - xmin) * 0.02, ymax * 0.96,
                    f"{pct_over:.1f}% beyond context  (n={n_over:,})",
                    color=C_CONV_ABOVE, fontsize=ANNOT_FS, ha="left", va="top")
        ax_tok.set_xlabel("Token count", fontsize=LABEL_FS)
        ax_tok.set_ylabel("Structures", fontsize=LABEL_FS)
        _style_ax(ax_tok)

        if "conv_count" not in df.columns or "prim_count" not in df.columns:
            ax_atom.text(0.5, 0.5, "Run compute_atom_counts() first",
                         ha="center", va="center", transform=ax_atom.transAxes,
                         fontsize=ANNOT_FS, color="gray")
        else:
            valid  = df["conv_count"].notna()
            within = df["token_count"] <= ctx
            conv_within = df.loc[valid &  within, "conv_count"].astype(int)
            conv_above  = df.loc[valid & ~within, "conv_count"].astype(int)
            prim_within = df.loc[valid &  within, "prim_count"].astype(int)
            prim_above  = df.loc[valid & ~within, "prim_count"].astype(int)

            all_counts = pd.concat([conv_within, conv_above, prim_within, prim_above])
            if not all_counts.empty:
                max_bin   = min(int(all_counts.max()), 300)
                atom_bins = 80
                atom_series = [
                    (conv_within.clip(upper=max_bin), C_CONV_WITHIN, 0.75, f"Conventional <=ctx  (n={len(conv_within):,})"),
                    (prim_within.clip(upper=max_bin), C_PRIM_WITHIN, 0.80, f"Primitive <=ctx  (n={len(prim_within):,})"),
                    (conv_above.clip(upper=max_bin),  C_CONV_ABOVE,  0.85, f"Conventional >ctx  (n={len(conv_above):,})"),
                    (prim_above.clip(upper=max_bin),  C_PRIM_ABOVE,  0.85, f"Primitive >ctx  (n={len(prim_above):,})"),
                ]
                _draw_hist_series(ax_atom, atom_series, bins=atom_bins)

                ax_atom.axvline(atom_density_line, color="black", linestyle="--",
                                linewidth=1.2, label=f"{atom_density_line} atoms")
                ax_atom.legend(fontsize=ANNOT_FS - 1, frameon=False, loc="upper right")
                cap_label = (f"Atoms per unit cell  (capped at {max_bin})"
                             if int(all_counts.max()) > max_bin else "Atoms per unit cell")
                ax_atom.set_xlabel(cap_label, fontsize=LABEL_FS)
            else:
                ax_atom.text(0.5, 0.5, "No parseable geometries",
                             ha="center", va="center", transform=ax_atom.transAxes,
                             fontsize=ANNOT_FS, color="gray")

        ax_atom.set_ylabel("Structures", fontsize=LABEL_FS)
        ax_atom.set_title(f"Atom count per unit cell  ({len(df):,} structures total)", fontsize=LABEL_FS)
        _style_ax(ax_atom)

        save_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()
        plt.close(fig)
