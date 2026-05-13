"""Notebook-scoped helper package."""

from _utils._notebook_utils._shared_utils import get_metrics_xrd, get_stratified_metrics_xrd
from _utils._notebook_utils.x_xrd_tio2_utils import process_xrd_to_condition_vector
from _utils._notebook_utils.x_slme_utils import (
    extract_formula,
    build_novelty_tag,
    parse_novelty_from_tag,
    select_top_materials,
    run_material_selection,
)
from _utils._notebook_utils.b1a_pretrain_benefits_utils import get_metrics_ptnd_vs_scratch
from _utils._notebook_utils.b2_dataset_size_study_utils import get_metrics_dataset_size_study
from _utils._notebook_utils.y_dataset_stats_utils import plot_dataset_stats

__all__ = [
    "get_metrics_xrd",
    "get_stratified_metrics_xrd",
    "process_xrd_to_condition_vector",
    "extract_formula",
    "build_novelty_tag",
    "parse_novelty_from_tag",
    "select_top_materials",
    "run_material_selection",
    "get_metrics_ptnd_vs_scratch",
    "get_metrics_dataset_size_study",
    "plot_dataset_stats",
]