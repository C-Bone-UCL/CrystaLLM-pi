"""Provide ranking strategies for generated CIF candidates.

Log-probability ranking measures model output likelihood and applies to all model families. XRD pearson ranking correlates a simulated powder pattern with the processed conditioning profile so it applies only to continuous-XRD models.
"""

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from tqdm import tqdm

from _models.xrd_utils import QMIN, QMAX, discrete_to_continuous_xrd
from _utils import extract_space_group_symbol, replace_symmetry_operators
from _utils._preprocessing.process_exp_xrd_continuous import DEFAULT_WAVELENGTH


def _score_transition_slice(generated_scores: tuple, original_sequence: torch.Tensor, input_length: int, eos_token_id: int | None=None) -> torch.Tensor:
    """Score one generated sequence from a precomputed transition-score row."""
    scoring_length = len(generated_scores)

    if eos_token_id is not None:
        eos_positions = (original_sequence == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            scoring_length = max(int(eos_positions[0]) - input_length, 0)

    if scoring_length <= 0:
        return float('inf')

    generated_only_scores = generated_scores[:scoring_length]

    if len(generated_only_scores) == 0:
        return float('inf')

    if torch.isnan(generated_only_scores).any() or torch.isinf(generated_only_scores).any():
        valid_scores = generated_only_scores[~(torch.isnan(generated_only_scores) | torch.isinf(generated_only_scores))]
        if len(valid_scores) == 0:
            return float('inf')
        generated_only_scores = valid_scores

    mean_log_prob = torch.mean(generated_only_scores).item()
    return np.exp(-mean_log_prob)


def score_output_logp(model: torch.nn.Module, scores: tuple, full_sequences: torch.Tensor, sequence_idx: int, input_length: int, eos_token_id: int | None=None) -> float:
    """Score one generated output based on transition scores up to EOS if present."""
    batch_scores = score_outputs_logp(
        model=model,
        scores=scores,
        full_sequences=full_sequences,
        input_length=input_length,
        eos_token_id=eos_token_id,
    )
    return batch_scores[sequence_idx]


def score_outputs_logp(model: torch.nn.Module, scores: tuple, full_sequences: torch.Tensor, input_length: int, eos_token_id: int | None=None) -> list[float]:
    """Score every generated output in a batch with one transition-score pass."""
    if scores is None or len(scores) == 0:
        if full_sequences is None:
            return [float('inf')]
        n_sequences = full_sequences.shape[0] if hasattr(full_sequences, "shape") else len(full_sequences)
        return [float('inf')] * n_sequences

    transition_scores = model.compute_transition_scores(
        full_sequences, scores, normalize_logits=True
    )

    if transition_scores.dim() == 1:
        transition_scores = transition_scores.unsqueeze(0)

    scored_outputs = []
    for sequence_idx, original_sequence in enumerate(full_sequences):
        scored_outputs.append(
            _score_transition_slice(
                generated_scores=transition_scores[sequence_idx],
                original_sequence=original_sequence,
                input_length=input_length,
                eos_token_id=eos_token_id,
            )
        )

    return scored_outputs


XRD_FIT_MODES = ("pearson",)

# Same fixed broadening that condition_vector_to_continuous_xrd applies to
# discrete conditioning peaks, so simulated and input profiles are comparable.
SIM_FWHM = 0.05
SIM_ETA = 0.5


def simulate_profile(cif_str: str, wavelength: float = DEFAULT_WAVELENGTH) -> np.ndarray:
    """Simulate a candidate's diffraction profile on the 1000-point Q grid.

    Q positions are wavelength independent, so 2theta runs to 180 deg to cover every reflection reachable at this wavelength rather than the usual 90.
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
    """Return the portion of the scan that was actually measured.

    The preprocessing pipeline zero-pads the profile outside the measured Q range.
    Those padded regions are excluded so simulated peaks outside the measured
    range do not affect the comparison.
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

    Each row carries its own conditioning profile in condition_vector, the nested (1000, 2) [Q, I] list built by the preprocessing pipeline.
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
