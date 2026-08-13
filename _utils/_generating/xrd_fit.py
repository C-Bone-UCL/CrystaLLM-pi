"""Rank generated CIFs by agreement between simulated and input XRD profiles.

Each candidate's powder pattern is simulated on the model's Q grid and compared
to the conditioning profile with Pearson correlation, so a Z search keeps the
candidate that best explains the scan. Pearson was chosen on a rutile recovery
benchmark, where it separated correct from wrong phases most cleanly.
"""

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from tqdm import tqdm

from _models.xrd_utils import QMIN, QMAX, discrete_to_continuous_xrd
from _utils import extract_space_group_symbol, replace_symmetry_operators
from _utils._preprocessing._process_exp_XRD_continuous import DEFAULT_WAVELENGTH

# Higher is better. Kept as a tuple so adding a mode only touches this module.
XRD_FIT_MODES = ("pearson",)

# Same fixed broadening that condition_vector_to_continuous_xrd applies to
# discrete conditioning peaks, so simulated and input profiles are comparable.
SIM_FWHM = 0.05
SIM_ETA = 0.5


def simulate_profile(cif_str: str, wavelength: float = DEFAULT_WAVELENGTH) -> np.ndarray:
    """Simulate a candidate's diffraction profile on the 1000-point Q grid.

    Q positions are wavelength independent, so 2theta runs to 180 deg to cover
    every reflection reachable at this wavelength rather than the usual 90.
    """
    # Scoring runs on raw generated CIFs, which hold only the asymmetric unit plus a
    # placeholder operator list. Expand the symmetry first or the simulated pattern
    # comes from a fraction of the cell contents.
    space_group = extract_space_group_symbol(cif_str)
    if space_group and space_group != "P 1":
        cif_str = replace_symmetry_operators(cif_str, space_group)

    structure = Structure.from_str(cif_str, fmt="cif")
    pattern = XRDCalculator(wavelength=wavelength).get_pattern(structure, two_theta_range=(0, 180))

    q = 4.0 * np.pi * np.sin(np.radians(np.asarray(pattern.x) / 2.0)) / wavelength
    intensity = np.asarray(pattern.y)

    keep = (q > QMIN) & (q < QMAX)
    if not keep.any():
        raise ValueError("No diffraction peaks fall inside the model Q grid")

    broadened = discrete_to_continuous_xrd(
        batch_q=torch.tensor(q[keep], dtype=torch.float32).unsqueeze(0),
        batch_iq=torch.tensor(intensity[keep], dtype=torch.float32).unsqueeze(0),
        fwhm_range=(SIM_FWHM, SIM_FWHM),
        eta_range=(SIM_ETA, SIM_ETA),
        noise_range=None,
        intensity_scale_range=None,
        seed=1,
    )
    return broadened["iq"][0].numpy().astype(np.float64)


def _measured_window(input_iq: np.ndarray) -> slice:
    """Grid slice the scan actually covered.

    The preprocessing pipeline zero-pads the profile outside the measured Q
    range, so comparing there would penalise correctly simulated peaks the
    instrument never saw.
    """
    nonzero = np.flatnonzero(input_iq)
    if nonzero.size == 0:
        raise ValueError("Input XRD profile is all zeros")
    return slice(nonzero[0], nonzero[-1] + 1)


def pearson_score(input_iq: np.ndarray, sim_iq: np.ndarray) -> float:
    """Pearson r between input and simulated profiles over the measured window."""
    window = _measured_window(input_iq)
    obs, calc = input_iq[window], sim_iq[window]

    # A flat simulated profile carries no peaks in the window, treat as no fit.
    if obs.std() == 0.0 or calc.std() == 0.0:
        return 0.0

    return float(np.corrcoef(obs, calc)[0, 1])


def score_generated_rows(df, wavelength: float = None) -> list:
    """One pearson score per dataframe row, np.nan where the CIF cannot be simulated.

    Each row carries its own conditioning profile in condition_vector, the
    nested (1000, 2) [Q, I] list built by the preprocessing pipeline.
    """
    wavelength = wavelength or DEFAULT_WAVELENGTH

    scores = []
    rows = zip(df["Generated CIF"], df["condition_vector"])
    for cif_str, condition in tqdm(rows, total=len(df), desc="XRD fit", dynamic_ncols=True):
        try:
            profile = np.asarray([list(row) for row in condition], dtype=np.float64)
            input_iq = profile[:, 1]
            scores.append(pearson_score(input_iq, simulate_profile(cif_str, wavelength)))
        except Exception:
            # Unparseable CIFs and degenerate profiles drop out of the ranking.
            scores.append(np.nan)

    return scores
