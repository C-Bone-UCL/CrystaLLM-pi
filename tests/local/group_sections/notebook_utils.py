"""Local test section: notebook utils."""

import os

import pandas as pd


class NotebookUtilsTests:
    """Test notebook utility helpers from the _utils/_notebooks package."""

    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data

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

    def test_build_and_parse_novelty_round_trip(self):
        from _utils._notebooks import build_novelty_tag, parse_novelty_from_tag

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
        from _utils._notebooks import select_top_materials

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
        from _utils._notebooks import run_material_selection

        input_path = os.path.join(self.temp_dir, "toy_materials.parquet")
        output_dir = os.path.join(self.temp_dir, "selected_cifs")
        output_csv = os.path.join(self.temp_dir, "summary.csv")
        self._toy_selection_df().to_parquet(input_path, index=False)

        run_material_selection(input_path, output_dir, output_csv, top_n_slme=1, top_n_sustain=1)

        assert os.path.exists(output_csv)
        assert any(name.endswith(".cif") for name in os.listdir(output_dir))

    def test_extract_formula_fallback(self):
        from _utils._notebooks import extract_formula

        assert extract_formula("not a cif") == "UnknownFormula"

    def test_run_material_selection_preserves_summary_columns(self):
        from _utils._notebooks import run_material_selection

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
