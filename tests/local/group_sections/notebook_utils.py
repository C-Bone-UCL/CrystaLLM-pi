"""Local test section: notebook utils."""

import os

import numpy as np
import pandas as pd


class NotebookUtilsTests:
    """Test notebook utility helpers from _utils/_notebook_utils.py."""

    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data

    def _toy_metrics_df(self):
        return pd.DataFrame({
            "RMS-d": [0.1, 0.2, np.nan],
            "True a": [5.0, 6.0, 7.0],
            "Gen a": [5.1, 5.9, 7.2],
            "True b": [5.0, 6.0, 7.0],
            "Gen b": [5.1, 5.9, 7.2],
            "True c": [5.0, 6.0, 7.0],
            "Gen c": [5.1, 5.9, 7.2],
            "True volume": [100.0, 200.0, 300.0],
            "Gen volume": [105.0, 195.0, 310.0],
            "Score": [0.9, 0.8, 0.7],
            "is_novel": [True, False, True],
            "is_comp_novel": [False, False, True],
            "atom_counts": [8, 12, 16],
        })

    def _toy_selection_df(self):
        return pd.DataFrame({
            "Generated CIF": [self.test_data["test_cif"], self.test_data["test_cif"]],
            "predicted_slme": [31.0, 27.0],
            "HHI_p": [100.0, 120.0],
            "HHI_r": [90.0, 110.0],
            "HHI_distance_to_0": [5.0, 8.0],
            "ehull_mace_mp": [0.01, 0.02],
            "is_novel": [True, False],
            "is_novel_pt": [False, False],
            "is_comp_novel": [False, False],
            "is_comp_novel_pt": [False, False],
        })

    def test_get_metrics_xrd_keys_and_counts(self):
        from _utils._notebook_utils import get_metrics_xrd

        metrics = get_metrics_xrd(self._toy_metrics_df(), n_test=3, verbose=False)

        assert metrics["Number of matched structures"] == 2
        assert metrics["Total number of structures"] == 3
        assert "Volume MAE" in metrics
        assert "Average Score" in metrics

    def test_get_stratified_metrics_xrd_tiers(self):
        from _utils._notebook_utils import get_stratified_metrics_xrd

        metrics = get_stratified_metrics_xrd(self._toy_metrics_df(), verbose=False)

        assert "Overall" in metrics.index
        assert "Memorized (Seen Comp & Struct)" in metrics.index
        assert "Structurally Novel (Seen Comp)" in metrics.index
        assert "Compositionally Novel (Unseen Comp)" in metrics.index
        assert "Atom Count (matched mean)" in metrics.columns

    def test_get_stratified_metrics_xrd_only_matched_and_missing_score(self):
        from _utils._notebook_utils import get_stratified_metrics_xrd

        df = self._toy_metrics_df().drop(columns=["Score"])
        metrics = get_stratified_metrics_xrd(df, only_matched=True, verbose=False)

        assert metrics.loc["Overall", "Matched"] == 2
        assert pd.isna(metrics.loc["Overall", "Avg Score"])
        assert "Vol MAE" in metrics.columns

    def test_process_xrd_to_condition_vector_output_length(self):
        from _utils._notebook_utils import process_xrd_to_condition_vector

        raw_pattern = "two_theta intensity\n10 100\n20 50\n"
        vec = process_xrd_to_condition_vector(raw_pattern)
        values = vec.split(",")

        assert len(values) == 40
        assert values[0] == "0.111"
        assert values[20] == "1.0"

    def test_build_and_parse_novelty_round_trip(self):
        from _utils._notebook_utils import build_novelty_tag, parse_novelty_from_tag

        row = pd.Series({
            "is_novel": True,
            "is_novel_pt": False,
            "is_comp_novel": False,
            "is_comp_novel_pt": True,
        })

        tag = build_novelty_tag(row)
        struct_nov, comp_nov = parse_novelty_from_tag(tag)

        assert tag == "Novel-ft__CompNovel-pt"
        assert struct_nov == "ft"
        assert comp_nov == "pt"

    def test_select_top_materials_returns_summary(self):
        from _utils._notebook_utils import select_top_materials

        materials, summary_df = select_top_materials(
            self._toy_selection_df(),
            top_n_slme=1,
            top_n_sustain=1,
            slme_threshold=25,
        )

        assert len(materials) == 2
        assert len(summary_df) == 2
        assert set(summary_df["Metric"]) == {"SLME", "HHI-SLME"}

    def test_export_and_run_material_selection_write_files(self):
        from _utils._notebook_utils import run_material_selection

        input_path = os.path.join(self.temp_dir, "toy_materials.parquet")
        output_dir = os.path.join(self.temp_dir, "selected_cifs")
        output_csv = os.path.join(self.temp_dir, "summary.csv")
        self._toy_selection_df().to_parquet(input_path, index=False)

        run_material_selection(input_path, output_dir, output_csv, top_n_slme=1, top_n_sustain=1)

        assert os.path.exists(output_csv)
        assert any(name.endswith(".cif") for name in os.listdir(output_dir))

    def test_extract_formula_fallback(self):
        from _utils._notebook_utils import extract_formula

        assert extract_formula("not a cif") == "UnknownFormula"

    def test_run_material_selection_preserves_summary_columns(self):
        from _utils._notebook_utils import run_material_selection

        input_path = os.path.join(self.temp_dir, "summary_cols_input.parquet")
        output_dir = os.path.join(self.temp_dir, "summary_cols_cifs")
        output_csv = os.path.join(self.temp_dir, "summary_cols.csv")
        self._toy_selection_df().to_parquet(input_path, index=False)

        run_material_selection(input_path, output_dir, output_csv, top_n_slme=1, top_n_sustain=1)
        summary_df = pd.read_csv(output_csv)

        assert list(summary_df.columns) == [
            "Reduced Formula",
            "Position",
            "Metric",
            "Pred. SLME",
            "HHI_p",
            "HHI_r",
            "HHI_dist",
            "E_hull_mace (eV/atom)",
            "Structure Nov.",
            "Composition Nov.",
        ]

    def test_get_metrics_ptnd_vs_scratch_returns_core_keys(self):
        from _utils._notebook_utils import get_metrics_ptnd_vs_scratch

        train_df = pd.DataFrame({"Bandgap (eV)": np.linspace(0.1, 7.0, 60)})
        df_dict = {
            "slider-pretrained": pd.DataFrame({
                "target_Bandgap (eV)": [6.2, 6.2],
                "ALIGNN_bg (eV)": [6.1, 6.3],
                "ehull_mace_mp": [0.01, 0.02],
                "is_valid": [True, True],
                "is_unique": [True, True],
                "is_novel": [True, True],
            }),
            "slider-scratch": pd.DataFrame({
                "target_Bandgap (eV)": [6.2, 6.2],
                "ALIGNN_bg (eV)": [5.7, 5.8],
                "ehull_mace_mp": [0.01, 0.20],
                "is_valid": [True, False],
                "is_unique": [True, True],
                "is_novel": [True, False],
            }),
        }

        metrics = get_metrics_ptnd_vs_scratch(df_dict, train_df=train_df)

        assert "avg_delta_validity" in metrics
        assert "avg_delta_hit_rate" in metrics
        assert "best_hit_rate_method" in metrics

    def test_get_metrics_dataset_size_study_returns_raw_dataframe(self):
        from _utils._notebook_utils import get_metrics_dataset_size_study

        train_df = pd.DataFrame({"Density (g/cm^3)": [1.0, 2.0, 3.0, 4.0]})
        dfs_dict = {
            "slider-1k": pd.DataFrame({
                "target_Density (g/cm^3)": [1.2747, 1.2747],
                "gen_density (g/cm3)": [1.1, 1.4],
                "is_valid": [True, True],
            }),
            "pkv-1k": pd.DataFrame({
                "target_Density (g/cm^3)": [1.2747, 1.2747],
                "gen_density (g/cm3)": [1.2, 1.3],
                "is_valid": [True, True],
            }),
        }

        metrics = get_metrics_dataset_size_study(dfs_dict, train_df, targets=(1.2747,))

        assert "raw_dataframe" in metrics
        assert "Slider_avg_mae" in metrics
        assert "PKV_avg_mae" in metrics

    def test_plot_dataset_stats_writes_png(self):
        import matplotlib

        matplotlib.use("Agg")

        from _utils._notebook_utils import plot_dataset_stats

        loaded = [(
            "toy_plot",
            pd.DataFrame({
                "token_count": [10, 40],
                "conv_count": [8, 12],
                "prim_count": [4, 6],
            }),
            20,
        )]
        save_dir = os.path.join(self.temp_dir, "plot_output")

        plot_dataset_stats(loaded, save_dir=save_dir)

        assert os.path.exists(os.path.join(save_dir, "toy_plot.png"))