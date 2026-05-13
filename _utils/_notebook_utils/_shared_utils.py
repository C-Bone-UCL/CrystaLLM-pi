"""Notebook-shared helper functions."""

import numpy as np
import pandas as pd


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
    max_matched = matched_counts.max() if len(matched_counts) > 0 else np.nan
    mean_unmatched = unmatched_counts.mean() if len(unmatched_counts) > 0 else np.nan

    return {
        "Atom Count (matched mean)": mean_matched,
        "Atom Count (matched max)": max_matched,
        "Atom Count (unmatched mean)": mean_unmatched,
        "Delta Atom Count (match - unmatch)": (
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


def get_metrics_xrd(df, n_test, only_matched=False, verbose=True):
    mean_rmsd = df["RMS-d"].mean()
    n_matches = df["RMS-d"].notna().sum()
    percent_match = n_matches / n_test * 100

    metrics_df = df

    if only_matched:
        metrics_df = metrics_df[metrics_df["RMS-d"].notna()]
        if verbose:
            print(
                f"Computing metrics only on matched, valid-system structures ({len(metrics_df)} entries)"
            )
    elif verbose:
        matched_in_subset = metrics_df["RMS-d"].notna().sum()
        print(
            f"Computing metrics on all valid-system structures ({len(metrics_df)} entries, {matched_in_subset} matched)"
        )

    property_metrics = _build_xrd_property_metrics(
        metrics_df,
        allow_missing=False,
        volume_label="Volume",
    )

    average_score = metrics_df["Score"].mean() if "Score" in metrics_df.columns else float("nan")

    metrics = {
        "Number of matched structures": n_matches,
        "Total number of structures": n_test,
        "Mean RMS-d": mean_rmsd,
        "Percent Matched (%)": percent_match,
        **property_metrics,
        "Average Score": average_score,
    }

    if verbose:
        print("\n--- Top-Level Stats (Full Dataset) ---")
        print(f"Number of matched structures: {n_matches} / {n_test}")
        print(f"Mean RMS-d: {mean_rmsd:.4f}")
        print(f"Percent Matched (%): {percent_match:.2f}% ({n_matches}/{n_test})")

        print("\n--- Property Metrics (Valid-System Data) ---")
        for key, value in property_metrics.items():
            print(f"{key}: {value:.4f}")
        if "Score" in metrics_df.columns:
            print(f"Average Score: {average_score:.4f}")

    return metrics


def get_stratified_metrics_xrd(
    df_metrics: pd.DataFrame,
    df_test: pd.DataFrame = None,
    only_matched: bool = False,
    verbose=True,
    n_test=None,
) -> pd.DataFrame:
    if df_test is not None:
        df = df_test.copy()
        if "Material ID" in df.columns and "Material ID" in df_metrics.columns:
            cols_to_use = df_metrics.columns.difference(df.columns).tolist() + ["Material ID"]
            df = df.merge(df_metrics[cols_to_use], on="Material ID", how="left")
        else:
            cols_to_use = df_metrics.columns.difference(df.columns)
            df = df.join(df_metrics[cols_to_use], how="left")
    else:
        missing = [column for column in ("is_novel", "is_comp_novel") if column not in df_metrics.columns]
        if missing:
            raise ValueError(
                f"df_test was not provided but df_metrics is missing columns: {missing}. "
                "Pass df_test, or run XRD_metrics.py with a ref_parquet that contains these columns."
            )
        df = df_metrics.copy()

    masks = {
        "Overall": pd.Series(True, index=df.index),
        "Memorized (Seen Comp & Struct)": ~df["is_novel"],
        "Structurally Novel (Seen Comp)": df["is_novel"] & ~df["is_comp_novel"],
        "Compositionally Novel (Unseen Comp)": df["is_comp_novel"],
    }

    results = {}

    for tier_name, mask in masks.items():
        subset = df[mask].copy()
        tier_size = len(subset)

        if tier_size == 0:
            continue

        if isinstance(n_test, dict):
            denom = n_test.get(tier_name, tier_size)
        elif tier_name == "Overall" and n_test is not None:
            denom = n_test
        else:
            denom = tier_size

        n_matches = subset["RMS-d"].notna().sum()
        percent_match = (n_matches / denom) * 100
        mean_rmsd = subset["RMS-d"].mean()
        average_score = subset["Score"].mean() if "Score" in subset.columns else np.nan

        prop_df = subset[subset["RMS-d"].notna()] if only_matched else subset
        atom_stats = _atom_count_stats(subset)
        property_metrics = _build_xrd_property_metrics(
            prop_df,
            allow_missing=True,
            volume_label="Vol",
        )

        results[tier_name] = {
            "Total Structures": denom,
            "Matched": n_matches,
            "Match Rate (%)": percent_match,
            "Mean RMS-d": mean_rmsd,
            **property_metrics,
            "Avg Score": average_score,
            **atom_stats,
        }

    if verbose:
        for tier, metrics in results.items():
            print(f"\n--- Metrics for {tier} ---")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    return pd.DataFrame(results).T

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

        try:
            cifs = df["CIF"].tolist()
        except KeyError:
            cifs = df["Generated CIF"].tolist()
            
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


__all__ = [
    "get_metrics_xrd",
    "get_stratified_metrics_xrd",
    "load_count_datasets",
    "compute_atom_counts",
]