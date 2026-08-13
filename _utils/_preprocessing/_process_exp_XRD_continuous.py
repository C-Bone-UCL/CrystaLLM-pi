"""
Convert raw experimental powder XRD scans into the continuous (1000, 2) [Q, I]
condition format consumed by the continuous-XRD models (PrefixXRD).

Upgrade to the legacy _process_exp_XRD_inputs.py (top-20 pre-picked peak
lists in 2theta space, for Slider models). This module keeps the full pattern:
it auto-detects where numeric data starts (header lines are skipped, never
interpreted), converts 2theta -> Q with the user-supplied wavelength
(Q-space is wavelength-independent), removes the background with pybaselines
SNIP, resamples onto the canonical Q grid and max-normalizes intensity to [0, 1].
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np

from _models.xrd_utils import NUM_Q_POINTS, QMAX, QMIN, QSTEP

DEFAULT_WAVELENGTH = 1.54056 # CuKa1
MIN_POINTS_ON_GRID = 100      
SNIP_HALF_WINDOW = 30         
_NUMERIC_RUN = 5              
_SPLIT_RE = re.compile(r"[,;\s]+")

def _line_floats(line: str) -> tuple[float, float] | None:
    """First two tokens as finite floats, or None when the line is not data."""
    tokens = [t for t in _SPLIT_RE.split(line.strip()) if t]
    if len(tokens) < 2:
        return None
    try:
        a, b = float(tokens[0]), float(tokens[1])
    except ValueError:
        return None
    return (a, b) if math.isfinite(a) and math.isfinite(b) else None


def _read_lines(path: Path) -> list[str]:
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ValueError(
            f"'{path}' looks binary. Export the scan as text (two-column .xy/.csv)"
        )
    return raw.decode("utf-8", errors="replace").splitlines()


def _excel_to_lines(path: Path) -> list[str]:
    import pandas as pd
    try:
        df = pd.read_excel(path, header=None, dtype=str)
    except ImportError as err:
        raise ValueError(f"Reading '{path.suffix}' needs openpyxl: pip install openpyxl") from err
    return [",".join(str(v) for v in row if v is not None) for row in df.itertuples(index=False)]


def read_xrd_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a raw scan. Returns (two_theta, intensity).

    Data starts at the first line opening a run of >= _NUMERIC_RUN numeric lines;
    everything above is header and is skipped.
    """

    path = Path(path)
    lines = _excel_to_lines(path) if path.suffix.lower() in (".xls", ".xlsx") else _read_lines(path)
    parsed = [_line_floats(line) for line in lines]

    start = next(
        (i for i in range(len(parsed))
         if all(p is not None for p in parsed[i:i + _NUMERIC_RUN]) and len(parsed) - i >= _NUMERIC_RUN),
        None,
    )

    if start is None:
        raise ValueError(
            f"Could not find two numeric columns in '{path}'. Expected lines of "
            "'2theta intensity' separated by whitespace, commas or semicolons."
        )

    # turn into array w/ numpy
    data = np.array([p for p in parsed[start:] if p is not None], dtype=np.float64)
    return data[:, 0], data[:, 1]


def _snip_baseline(grid: np.ndarray, iq: np.ndarray, half_window: int) -> np.ndarray:
    """Background estimate via pybaselines SNIP (the standard for powder diffraction)."""
    from pybaselines import Baseline

    half_window = min(half_window, max(1, (len(iq) - 1) // 2))
    baseline, _ = Baseline(x_data=grid).snip(iq, max_half_window=half_window, decreasing=True)
    # Edge extrapolation can drag the estimate below zero near the scan boundaries.
    # A physical X-ray background is non-negative.
    return np.clip(baseline, 0.0, None)


def _resolve_wavelength(wavelength: float | None, source) -> float:
    if wavelength is None:
        print(f"Warning: no wavelength given for '{source}', assuming CuKa1 {DEFAULT_WAVELENGTH} A. "
              "Specify --xrd_wavelength explicitly for non-CuKa data.")
        return DEFAULT_WAVELENGTH
    return wavelength


def _pipeline_stages(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    background_subtract: bool = True,
    snip_half_window: int = SNIP_HALF_WINDOW,
) -> dict:
    """Run the conversion pipeline, keeping every intermediate for plotting and tests."""

    if not 0.1 < wavelength < 5.0:
        raise ValueError(f"Implausible X-ray wavelength {wavelength} A")
    
    two_theta = np.asarray(two_theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    # Convert to q
    q = 4.0 * np.pi * np.sin(np.radians(two_theta / 2.0)) / wavelength
    order = np.argsort(q)
    q_sorted, iq_sorted = q[order], intensity[order]

    # Sample on grid
    grid = np.arange(QMIN, QMAX, QSTEP)
    covered = int(((grid >= q_sorted.min()) & (grid <= q_sorted.max())).sum())
    if covered < MIN_POINTS_ON_GRID:
        raise ValueError(
            f"Only {covered} grid points fall inside the measured Q range "
            f"[{q_sorted.min():.2f}, {q_sorted.max():.2f}] A^-1, check --xrd_wavelength and the input columns."
        )
    if q_sorted.max() > QMAX:
        print(f"Note: data above Q={QMAX} A^-1 lies outside the model grid and is ignored.")

    iq_interp = np.interp(grid, q_sorted, iq_sorted, left=0.0, right=0.0)

    # Estimate the baseline only inside the measured Q range
    baseline = np.zeros_like(iq_interp)

    if background_subtract:
        measured = (grid >= q_sorted[0]) & (grid <= q_sorted[-1])
        baseline[measured] = _snip_baseline(grid[measured], iq_interp[measured], snip_half_window)

    iq_subtracted = np.clip(iq_interp - baseline, 0.0, None)

    peak = iq_subtracted.max()
    if peak <= 0.0:
        raise ValueError("No positive intensity left on the Q grid after processing.")

    return {
        "two_theta": two_theta, "intensity": intensity, # raw scan
        "q": q_sorted, "iq_q": iq_sorted, # converted to Q
        "grid": grid, "iq_interp": iq_interp, "baseline": baseline,  # resampled + background
        "iq_final": iq_subtracted / peak, # normalized profile
    }


def convert_to_continuous_profile(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    wavelength: float,
    background_subtract: bool = True,
    snip_half_window: int = SNIP_HALF_WINDOW,
) -> list[list[float]]:
    """2theta -> Q -> canonical grid. Returns the nested (1000, 2) [Q, I] list."""
    stages = _pipeline_stages(two_theta, intensity, wavelength, background_subtract, snip_half_window)
    return np.column_stack([stages["grid"], stages["iq_final"]]).tolist()


def save_pipeline_plot(
    input_data: str | Path,
    save_path: str | Path,
    wavelength: float | None = None,
    background_subtract: bool = True,
) -> str:
    """transform overview for the 4 processes above"""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two_theta, intensity = read_xrd_file(input_data)
    stages = _pipeline_stages(two_theta, intensity, _resolve_wavelength(wavelength, input_data), background_subtract)

    fig, axes = plt.subplots(4, 1, figsize=(7, 11), constrained_layout=True)

    axes[0].plot(stages["two_theta"], stages["intensity"], lw=0.6, color="black")
    axes[0].set_xlabel(r"2$\theta$ (deg)")
    axes[0].set_title("1. Raw scan")
    axes[1].plot(stages["q"], stages["iq_q"], lw=0.6, color="black")
    axes[1].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    axes[1].set_title("2. Converted to Q")

    axes[2].plot(stages["grid"], stages["iq_interp"], lw=0.6, color="black", label="resampled (1000 pts)")
    axes[2].plot(stages["grid"], stages["baseline"], lw=1.2, color="crimson", label="SNIP baseline")
    axes[2].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    axes[2].set_title("3. Grid resampling + background estimate")
    axes[2].legend()
    axes[3].plot(stages["grid"], stages["iq_final"], lw=0.6, color="black")
    axes[3].set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
    axes[3].set_title("4. Final (1000, 2) profile, max-normalized")

    for ax in axes:
        ax.set_ylabel("Intensity")

    save_file_path = Path(save_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_file_path, dpi=200)
    plt.close(fig)
    return str(save_file_path)


def process_exp_file_to_continuous(
    input_data: str | Path,
    wavelength: float | None = None,
    background_subtract: bool = True,
) -> list[list[float]]:
    """Raw scan file -> continuous condition. CuKa1 assumed when no wavelength is given."""

    two_theta, intensity = read_xrd_file(input_data)

    # Already-processed profiles (1000 rows on the exact Q grid) pass straight through,
    # so a saved --output_csv can be re-fed via --xrd_files without double conversion.
    if two_theta.shape[0] == NUM_Q_POINTS and np.allclose(two_theta, np.arange(QMIN, QMAX, QSTEP), atol=1e-4):
        iq = np.clip(intensity, 0.0, None)
        if iq.max() <= 0.0:
            raise ValueError(f"Profile '{input_data}' has no positive intensity.")
        return np.column_stack([np.arange(QMIN, QMAX, QSTEP), iq / iq.max()]).tolist()

    return convert_to_continuous_profile(
        two_theta, intensity, _resolve_wavelength(wavelength, input_data), background_subtract
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw experimental XRD scan -> continuous [Q, I] condition profile for PrefixXRD models."
    )
    parser.add_argument("--input_data", required=True, help="Raw scan: text (.txt/.csv/.xy/.dat) or Excel")

    parser.add_argument("--output_csv", default=None,
                        help="Optional destination for the 1000-row 'Q,intensity' profile. Not needed to "
                             "generate, --xrd_files converts a raw scan on the fly")

    parser.add_argument("--xrd_wavelength", type=float, default=None,
                        help="Wavelength in Angstrom (default: CuKa1 1.54056, specify for non-CuKa data)")
    parser.add_argument("--no_background_subtract", action="store_true",
                        help="Skip SNIP background removal")
    parser.add_argument("--save_plot", default=None,
                        help="Optional PNG: 4-panel raw -> Q -> background -> final-profile overview")
    
    args = parser.parse_args()
    if not args.output_csv and not args.save_plot:
        parser.error("nothing to write, pass --output_csv or --save_plot")

    profile = process_exp_file_to_continuous(args.input_data, args.xrd_wavelength, not args.no_background_subtract)

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8") as file:
            file.write("Q,intensity\n")
            file.writelines(f"{point[0]:.2f},{point[1]:.6f}\n" for point in profile)

        print(f"Wrote {len(profile)}-point profile to {args.output_csv}")

    if args.save_plot:
        save_pipeline_plot(args.input_data, args.save_plot, args.xrd_wavelength, not args.no_background_subtract)
        print(f"Pipeline plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()
