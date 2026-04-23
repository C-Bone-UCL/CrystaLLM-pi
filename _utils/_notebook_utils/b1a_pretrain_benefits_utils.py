"""Helpers for notebooks/B1a_Pretrain_benefits.ipynb."""

import numpy as np
import pandas as pd

from ._shared_utils import _fit_line_and_corr, _sem, _top_two_means


def get_metrics_ptnd_vs_scratch(
    df_dict,
    *,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    target_bg_col="target_Bandgap (eV)",
    cond_col="target_Bandgap (eV)",
    hit_tol_eV=0.2,
):
    from scipy.stats import gaussian_kde

    def _method_pretrain(label):
        lowered = label.lower()
        if "slider" in lowered:
            method = "Slider"
        elif "pkv" in lowered:
            method = "PKV"
        elif "prepend" in lowered:
            method = "Prepend"
        else:
            method = "Raw"
        return method, ("scratch" not in lowered)

    conds = set()
    for frame in df_dict.values():
        if cond_col in frame.columns:
            conds.update(frame[cond_col].dropna().unique())
    conds = sorted(map(float, conds))

    rows = []
    for label, frame in df_dict.items():
        method, pretrained = _method_pretrain(label)
        if cond_col not in frame.columns:
            frame = frame.assign(**{cond_col: [np.nan] * len(frame)})

        tau = 0.157
        ehull = np.asarray(frame["ehull_mace_mp"], dtype=float)
        frame = frame.copy()
        frame["is_stable_mace"] = ehull <= tau

        for condition in conds or [np.nan]:
            subset = frame[frame[cond_col] == condition].copy() if not pd.isna(condition) else frame.copy()
            count = len(subset)
            if count == 0:
                rows.append({"m": method, "p": pretrained, "x": condition, "n": 0, "hit": np.nan, "valid": np.nan, "q": np.nan})
                continue

            hit = np.nan
            if {pred_bg_col, target_bg_col}.issubset(subset.columns):
                pred = np.asarray(subset[pred_bg_col], dtype=float)
                true = np.asarray(subset[target_bg_col], dtype=float)
                hit = (np.abs(pred - true) <= hit_tol_eV).mean()

            valid = subset["is_valid"].mean() if "is_valid" in subset.columns else np.nan

            quality = np.nan
            required = {"is_valid", "is_unique", "is_novel", "is_stable_mace"}
            if required.issubset(subset.columns):
                quality_mask = (
                    subset["is_valid"]
                    & subset["is_unique"]
                    & subset["is_novel"]
                    & subset["is_stable_mace"]
                )
                quality = quality_mask.mean()
            rows.append({"m": method, "p": pretrained, "x": condition, "n": count, "hit": hit, "valid": valid, "q": quality})

    met = pd.DataFrame(rows)

    if train_df is not None and train_bg_col in train_df.columns and len(train_df) > 50:
        vals = pd.to_numeric(train_df[train_bg_col], errors="coerce").dropna()
        kde = gaussian_kde(vals, bw_method="scott")
        xs = np.linspace(vals.min(), vals.max(), 400)
        dens = kde(xs)
        dens_norm = dens / dens.max()
        met["density"] = np.interp(met["x"].astype(float), xs, dens_norm)

    met = met[met["n"] > 0].copy()
    met["hit_rate"] = met["hit"]

    metrics = {}

    if "density" in met.columns:
        density = met["density"]
        slope, intercept, corr = _fit_line_and_corr(density, met["hit_rate"])
        metrics["hit_rate_vs_density_slope"] = slope
        metrics["hit_rate_vs_density_intercept"] = intercept
        metrics["hit_rate_density_correlation"] = corr

    diffs_valid = []
    diffs_quality = []
    diffs_hit = []

    for method in met.m.unique():
        group = met[met.m == method]

        valid_pretrained = group[group.p]["valid"].mean()
        valid_scratch = group[~group.p]["valid"].mean()
        if not np.isnan(valid_pretrained) and not np.isnan(valid_scratch):
            diffs_valid.append(valid_pretrained - valid_scratch)

        quality_pretrained = group[group.p]["q"].mean()
        quality_scratch = group[~group.p]["q"].mean()
        if not np.isnan(quality_pretrained) and not np.isnan(quality_scratch):
            diffs_quality.append(quality_pretrained - quality_scratch)

        hit_pretrained = group[group.p]["hit_rate"].mean()
        hit_scratch = group[~group.p]["hit_rate"].mean()
        if not np.isnan(hit_pretrained) and not np.isnan(hit_scratch):
            diffs_hit.append(hit_pretrained - hit_scratch)

    if diffs_valid:
        metrics["avg_delta_validity"] = np.mean(diffs_valid)
        metrics["sem_delta_validity"] = _sem(diffs_valid)

    if diffs_quality:
        metrics["avg_delta_quality"] = np.mean(diffs_quality)
        metrics["sem_delta_quality"] = _sem(diffs_quality)

    if diffs_hit:
        metrics["avg_delta_hit_rate"] = np.mean(diffs_hit)
        metrics["sem_delta_hit_rate"] = _sem(diffs_hit)

    candidates = [condition for condition in conds if 6.0 <= condition < 7.0]
    best_ratio = None
    best_condition = None
    for condition in candidates:
        values = met[met.x == condition]["hit_rate"].dropna()
        if len(values) > 1 and values.min() > 0:
            ratio = values.max() / values.min()
            if best_ratio is None or ratio > best_ratio:
                best_ratio = ratio
                best_condition = condition

    if best_ratio is not None:
        metrics["best_worst_ratio_6_7eV"] = best_ratio
        metrics["best_worst_ratio_bandgap"] = best_condition

    if "density" in met.columns:
        density = met["density"]
        valid_slope, valid_intercept, valid_corr = _fit_line_and_corr(density, met["valid"])
        quality_slope, quality_intercept, quality_corr = _fit_line_and_corr(density, met["q"])

        metrics["valid_vs_density_slope"] = valid_slope
        metrics["valid_vs_density_intercept"] = valid_intercept
        metrics["valid_density_correlation"] = valid_corr
        metrics["quality_vs_density_slope"] = quality_slope
        metrics["quality_vs_density_intercept"] = quality_intercept
        metrics["quality_density_correlation"] = quality_corr

    for metric_name in ("hit_rate", "valid", "q"):
        means = met.groupby(["m", "p"])[metric_name].mean().dropna()
        top_two = _top_two_means(means)
        if top_two is None:
            continue

        ((first_method, first_pretrained), first_value), ((second_method, second_pretrained), second_value) = top_two
        metrics[f"best_{metric_name}_method"] = first_method
        metrics[f"best_{metric_name}_pretrained"] = first_pretrained
        metrics[f"best_{metric_name}_value"] = first_value
        metrics[f"second_best_{metric_name}_method"] = second_method
        metrics[f"second_best_{metric_name}_pretrained"] = second_pretrained
        metrics[f"second_best_{metric_name}_value"] = second_value
        metrics[f"{metric_name}_improvement_percent"] = (first_value - second_value) * 100 if second_value > 0 else np.nan

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

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

    for metric_name in ("hit_rate", "valid", "q"):
        if f"best_{metric_name}_method" not in metrics:
            continue

        first_method = metrics[f"best_{metric_name}_method"]
        first_pretrained = metrics[f"best_{metric_name}_pretrained"]
        first_value = metrics[f"best_{metric_name}_value"]
        second_method = metrics[f"second_best_{metric_name}_method"]
        second_pretrained = metrics[f"second_best_{metric_name}_pretrained"]
        second_value = metrics[f"second_best_{metric_name}_value"]
        improvement = metrics[f"{metric_name}_improvement_percent"]
        first_name = f"{first_method} ({'pretrained' if first_pretrained else 'scratch'})"
        second_name = f"{second_method} ({'pretrained' if second_pretrained else 'scratch'})"
        print(
            f"Best run for mean {metric_name}: {first_name} ({first_value:.3f}), "
            f"{improvement:.1f}% above {second_name} ({second_value:.3f})"
        )

    return metrics


__all__ = ["get_metrics_ptnd_vs_scratch"]