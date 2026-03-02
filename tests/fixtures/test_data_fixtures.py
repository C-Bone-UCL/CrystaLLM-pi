"""Shared builders for test DataFrames and fixture paths."""

import os

import pandas as pd


def build_test_dataframe(base_cif: str, include_al2o3_row: bool = False) -> pd.DataFrame:
    """Build canonical test dataframe used by local/API suites."""
    rows = [
        {
            "Material ID": "test-1",
            "Reduced Formula": "SiO2",
            "CIF": base_cif,
            "Database": "test",
            "Bandgap (eV)": 2.5,
            "Density (g/cm^3)": 2.2,
        },
        {
            "Material ID": "test-2",
            "Reduced Formula": "TiO2",
            "CIF": base_cif.replace("SiO2", "TiO2").replace("Si", "Ti"),
            "Database": "test",
            "Bandgap (eV)": 3.1,
            "Density (g/cm^3)": 4.2,
        },
    ]

    if include_al2o3_row:
        rows.append(
            {
                "Material ID": "test-3",
                "Reduced Formula": "Al2O3",
                "CIF": base_cif.replace("SiO2", "Al2O3").replace("Si", "Al"),
                "Database": "test",
                "Bandgap (eV)": 8.8,
                "Density (g/cm^3)": 3.9,
            }
        )

    return pd.DataFrame(rows)


def get_fixture_xrd_csv(project_root: str) -> str:
    """Return absolute path to shared rutile XRD fixture."""
    return os.path.join(project_root, "tests", "fixtures", "test_rutile_processed.csv")
