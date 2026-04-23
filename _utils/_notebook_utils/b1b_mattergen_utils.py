"""Helpers for notebooks/B1b_Mattergen.ipynb."""

from ._shared_utils import compute_atom_counts, load_count_datasets

def compute_atom_count_distribution(df):
    count_series = df['prim_count']
    value_counts = count_series.value_counts(normalize=True).sort_index()
    return value_counts.to_dict()

__all__ = [
    "load_count_datasets",
    "compute_atom_counts",
    "compute_atom_count_distribution",
]