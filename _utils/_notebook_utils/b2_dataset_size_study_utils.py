"""Helpers for notebooks/B2_Dataset_size_study.ipynb."""

import numpy as np
import pandas as pd

from ._shared_utils import _sem


def get_metrics_dataset_size_study(
    dfs_dict,
    train_df,
    *,
    train_col="Density (g/cm^3)",
    target_col="target_Density (g/cm^3)",
    gen_col="gen_density (g/cm3)",
    targets=(1.2747, 15.30, 22.94),
):
    from scipy.stats import pearsonr, ttest_rel

    def _model(key: str) -> str:
        lowered = key.lower()
        if "pkv" in lowered:
            return "PKV"
        if "slider" in lowered:
            return "Slider"
        return "Prepend"

    def _size(key: str) -> str:
        for size in ("100k", "10k", "1k"):
            if f"-{size}" in key:
                return size
        return "full"

    def _subset(frame: pd.DataFrame, tgt_col: str, tgt: float, pred_col: str):
        mask = np.isclose(frame[tgt_col], tgt, atol=1e-2) & frame["is_valid"] & frame[pred_col].notna()
        return pd.to_numeric(frame.loc[mask, pred_col], errors="coerce").dropna()

    metrics_list = []
    for target in targets:
        for key, frame in dfs_dict.items():
            method = _model(key)
            size = _size(key)
            values = _subset(frame, target_col, target, gen_col)
            if values.empty:
                continue

            metrics_list.append(
                {
                    "method": method,
                    "size": size,
                    "target": target,
                    "count": len(values),
                    "mean": float(np.mean(values.values)),
                    "mae": np.mean(np.abs(values.values - target)),
                    "std": np.std(values.values, ddof=0),
                }
            )

    met_df = pd.DataFrame(metrics_list)
    if "size" in met_df.columns:
        met_df["size"] = met_df["size"].astype(str).str.strip()

    size_map = {"full": len(train_df), "100k": 1e5, "10k": 1e4, "1k": 1e3}
    met_df["size_numeric"] = met_df["size"].map(size_map)

    metrics = {}

    for method in met_df.method.unique():
        subset = met_df[met_df.method == method]
        if len(subset) < 2:
            continue

        corr, p_value = pearsonr(subset["count"], subset["size_numeric"])
        metrics[f"{method}_count_vs_size_correlation"] = corr
        metrics[f"{method}_count_vs_size_pvalue"] = p_value

    for method in met_df.method.unique():
        mae_values = met_df[met_df.method == method]["mae"].values
        if len(mae_values) > 0:
            metrics[f"{method}_avg_mae"] = mae_values.mean()
            metrics[f"{method}_sem_mae"] = _sem(mae_values)

        std_values = met_df[met_df.method == method]["std"].values
        if len(std_values) > 0:
            metrics[f"{method}_avg_std"] = std_values.mean()
            metrics[f"{method}_sem_std"] = _sem(std_values)

    df_1k = met_df[met_df["size"] == "1k"]
    for method in ["Slider", "PKV", "Prepend"]:
        method_counts = df_1k[df_1k["method"] == method]["count"]
        if method_counts.empty:
            continue

        metrics[f"{method}_1k_avg_valid_count"] = method_counts.mean()
        metrics[f"{method}_1k_sem_valid_count"] = _sem(method_counts.values, single_value=0.0)

    for method in met_df.method.unique():
        for size in ("1k", "10k", "100k", "full"):
            subset = met_df[(met_df.method == method) & (met_df.size == size)]
            if subset.empty:
                continue

            values = subset["mae"].values
            metrics[f"{method}_{size}_mae"] = values.mean()
            metrics[f"{method}_{size}_mae_sem"] = _sem(values)

    df_wide = met_df.pivot_table(index=["target", "size"], columns="method", values="mae").dropna()

    if {"Slider", "PKV"}.issubset(df_wide.columns):
        t_stat, p_value = ttest_rel(df_wide["Slider"], df_wide["PKV"])
        metrics["ttest_slider_vs_pkv_t"] = t_stat
        metrics["ttest_slider_vs_pkv_p"] = p_value

    if {"Prepend", "PKV"}.issubset(df_wide.columns):
        t_stat, p_value = ttest_rel(df_wide["Prepend"], df_wide["PKV"])
        metrics["ttest_prepend_vs_pkv_t"] = t_stat
        metrics["ttest_prepend_vs_pkv_p"] = p_value

    for method in met_df.method.unique():
        subset = met_df[met_df.method == method]
        if len(subset) >= 2:
            print(f"{method}: Pearson r (valid count vs size) = {metrics[f'{method}_count_vs_size_correlation']:.3f}")

    for label, mean_suffix, sem_suffix in (("Avg MAE", "avg_mae", "sem_mae"), ("Avg std", "avg_std", "sem_std")):
        for method in met_df.method.unique():
            mean_key = f"{method}_{mean_suffix}"
            sem_key = f"{method}_{sem_suffix}"
            if mean_key in metrics:
                print(f"{method}: {label} = {metrics[mean_key]:.3f} +/- {metrics[sem_key]:.3f}")

    for method in ["Slider", "PKV", "Prepend"]:
        mean_key = f"{method}_1k_avg_valid_count"
        sem_key = f"{method}_1k_sem_valid_count"
        if mean_key in metrics:
            print(f"{method} (1k): Avg valid count = {metrics[mean_key]:.1f} +/- {metrics[sem_key]:.1f}")

    for method in met_df.method.unique():
        for size in ("1k", "10k", "100k", "full"):
            mean_key = f"{method}_{size}_mae"
            sem_key = f"{method}_{size}_mae_sem"
            if mean_key in metrics:
                print(f"{method} ({size}): Avg MAE = {metrics[mean_key]:.3f} +/- {metrics[sem_key]:.3f}")

    if "ttest_slider_vs_pkv_t" in metrics:
        print(
            "t-test Slider vs PKV MAE: "
            f"t={metrics['ttest_slider_vs_pkv_t']:.3f}, p={metrics['ttest_slider_vs_pkv_p']:.3f}"
        )

    if "ttest_prepend_vs_pkv_t" in metrics:
        print(
            "t-test Prepend vs PKV MAE: "
            f"t={metrics['ttest_prepend_vs_pkv_t']:.3f}, p={metrics['ttest_prepend_vs_pkv_p']:.3f}"
        )

    metrics["raw_dataframe"] = met_df
    return metrics


__all__ = ["get_metrics_dataset_size_study"]