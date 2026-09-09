"""Provide ranking strategies for generated CIF candidates.

Log-probability ranking measures model output likelihood and applies to all model families. XRD
pearson ranking correlates a simulated powder pattern with the processed conditioning profile so it
applies only to continuous-XRD models.
"""

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from tqdm import tqdm

from _models.xrd_utils import QMIN, QMAX, discrete_to_continuous_xrd
from _utils import extract_space_group_symbol, replace_symmetry_operators
from _utils._preprocessing.process_exp_xrd_continuous import DEFAULT_WAVELENGTH


def forward_pass_logp(model: torch.nn.Module, full_sequences: torch.Tensor, input_length: int, eos_token_id: int | None=None, condition_tensor: torch.Tensor | None=None) -> list[float]:
    """Score generated sequences by perplexity under the model, in one forward pass.

    Uses a fresh teacher-forced pass over the full vocabulary at temperature 1. Scoring
    `outputs.scores` instead measures the sampling distribution, whose per-step renormaliser
    varies by context and so moves the ranking. Excludes tokens before `input_length` and
    from EOS onward.
    """
    if full_sequences is None or full_sequences.numel() == 0:
        return []

    forward_kwargs = {}
    if condition_tensor is not None:
        if condition_tensor.shape[0] != full_sequences.shape[0]:
            repeat_shape = [full_sequences.shape[0]] + [1] * (condition_tensor.dim() - 1)
            condition_tensor = condition_tensor.repeat(*repeat_shape)
        forward_kwargs["condition_values"] = condition_tensor

    with torch.inference_mode():
        logits = model(input_ids=full_sequences, **forward_kwargs).logits

    shifted_logits = logits[:, :-1, :]
    shifted_labels = full_sequences[:, 1:]
    token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = token_log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

    start_idx = max(int(input_length) - 1, 0)
    positions = torch.arange(shifted_labels.shape[1], device=full_sequences.device).unsqueeze(0)
    valid_mask = positions >= start_idx

    if eos_token_id is not None:
        eos_mask = full_sequences.eq(eos_token_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.where(
            has_eos,
            eos_mask.float().argmax(dim=1),
            torch.full((full_sequences.shape[0],), full_sequences.shape[1], device=full_sequences.device, dtype=torch.long),
        )
        valid_mask = valid_mask & (positions < (first_eos.unsqueeze(1) - 1))

    token_counts = valid_mask.sum(dim=1)
    safe_counts = torch.clamp(token_counts, min=1)
    mean_log_probs = (gathered_log_probs * valid_mask).sum(dim=1) / safe_counts
    perplexities = torch.exp(-mean_log_probs)

    return [
        float("inf") if token_counts[i].item() <= 0 else perplexities[i].item()
        for i in range(full_sequences.shape[0])
    ]


XRD_FIT_MODES = ("pearson",)

# Same fixed broadening that condition_vector_to_continuous_xrd applies to
# discrete conditioning peaks, so simulated and input profiles are comparable.
SIM_FWHM = 0.05
SIM_ETA = 0.5


def simulate_profile(cif_str: str, wavelength: float = DEFAULT_WAVELENGTH) -> np.ndarray:
    """Simulate a candidate's diffraction profile on the 1000-point Q grid.

    Q positions are wavelength independent, so 2theta runs to 180 deg to cover every reflection
    reachable at this wavelength rather than the usual 90.
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

    Each row carries its own conditioning profile in condition_vector, the nested (1000, 2) [Q, I]
    list built by the preprocessing pipeline.
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
