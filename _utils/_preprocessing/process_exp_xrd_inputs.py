r"""Format experimental XRD data for legacy Slider-family conditioning.

Accepts either a list of picked peaks or a continuous powder profile (a measured or digitised scan).
Profiles are detected automatically (at least 50 points, increasing 2theta, roughly uniform step) and
reduced to peaks before formatting: light Gaussian smoothing, subtraction of a rolling-minimum
background, then `scipy.signal.find_peaks` with a relative height threshold and a minimum peak
separation. Peak lists pass through unchanged. Header and comment lines that are not two numbers
are skipped. K-alpha2 components are not stripped; resolved doublets may appear as two peaks.

Usage:
    ```bash
    python _utils/_preprocessing/process_exp_xrd_inputs.py --input_data scan.xy \
        --output_csv conditioned.csv --xrd_wavelength 1.54056
    ```
"""

import os
import argparse
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

# Global constants 
TARGET_WAVELENGTH = 1.54056
MAX_PEAKS = 20

# Profile detection and peak picking defaults (2theta in degrees)
PROFILE_MIN_POINTS = 50
PROFILE_MAX_STEP_CV = 0.5
PICK_MIN_REL_HEIGHT = 0.02
PICK_MIN_SEPARATION = 0.2
PICK_SMOOTHING_FWHM = 0.1
PICK_BACKGROUND_WINDOW = 2.0


def load_two_column_xrd(input_data):
    """Load the first two numeric columns of an XRD file, skipping non-numeric lines.

    Comma, tab and whitespace delimiters are accepted, so header rows such as
    ``2theta,intensity`` or ``Wavelength = 1.54059`` and comment lines are ignored. Row order is
    preserved.
    """
    rows = []
    with open(input_data) as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"Could not parse input file {input_data}. Ensure it is 2 columns of numeric data.")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def load_picked_peaks(input_data):
    """Load XRD two-column for csv/txt/xy/dat. Kept for backward compatibility."""
    if not os.path.exists(input_data):
        raise ValueError(f"Could not parse input file {input_data}: file not found.")
    return load_two_column_xrd(input_data)
    
        
def is_continuous_profile(two_theta):
    """True when the angles look like a scanned profile rather than a list of picked peaks.
        
    A profile has at least ``PROFILE_MIN_POINTS`` points in strictly increasing 2theta with a
    roughly uniform step. Peak lists are short, or listed by intensity, or irregularly spaced.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    if len(two_theta) < PROFILE_MIN_POINTS:
        return False
    steps = np.diff(two_theta)
    if np.any(steps <= 0):
        return False
    return float(np.std(steps) / np.mean(steps)) < PROFILE_MAX_STEP_CV
            

def pick_peaks(two_theta, intensity,
               min_rel_height=PICK_MIN_REL_HEIGHT,
               min_separation=PICK_MIN_SEPARATION,
               smoothing_fwhm=PICK_SMOOTHING_FWHM,
               background_window=PICK_BACKGROUND_WINDOW):
    """Reduce a continuous XRD profile to peak positions and background-corrected heights.

    Args:
        two_theta: Profile angles in degrees, strictly increasing, shape (N,).
        intensity: Profile intensities, shape (N,).
        min_rel_height: Minimum peak prominence as a fraction of the strongest corrected peak.
        min_separation: Minimum distance between picked peaks, in degrees.
        smoothing_fwhm: FWHM of the Gaussian smoothing applied before picking, in degrees.
            Smoothing is skipped when this is under about one step.
        background_window: Width of the rolling-minimum background estimate, in degrees.
            Should exceed the widest peak.

    Returns:
        Peak angles and heights above background, each shape (M,), in increasing 2theta.
    """
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if len(x) < 3:
        return x, y
    step = float(np.median(np.diff(x)))

    sigma = smoothing_fwhm / 2.3548 / step
    if sigma >= 0.5:
        y = gaussian_filter1d(y, sigma, mode="nearest")

    window = max(3, int(round(background_window / step)) | 1)
    background = uniform_filter1d(minimum_filter1d(y, window, mode="nearest"), window, mode="nearest")
    corrected = np.clip(y - background, 0.0, None)
    if corrected.max() <= 0:
        return np.array([]), np.array([])

    idx, _ = find_peaks(corrected,
                        prominence=min_rel_height * corrected.max(),
                        distance=max(1, int(round(min_separation / step))))
    return x[idx], corrected[idx]


def convert_wavelength(two_theta, lambda_in, lambda_out):
    """Maps 2-theta angles using d-spacing as an intermediate."""
    if abs(lambda_in - lambda_out) < 1e-6:
        return two_theta, np.ones_like(two_theta, dtype=bool)

    theta_rad = np.radians(two_theta / 2.0)
    valid_mask = np.abs(np.sin(theta_rad)) > 1e-6
    
    d = np.zeros_like(two_theta)
    d[valid_mask] = lambda_in / (2.0 * np.sin(theta_rad[valid_mask]))
    
    sin_arg = np.zeros_like(two_theta)
    sin_arg[valid_mask] = lambda_out / (2.0 * d[valid_mask])
    
    sin_arg = np.clip(sin_arg, 0, 1)
    two_theta_new = 2.0 * np.degrees(np.arcsin(sin_arg))
    
    final_mask = valid_mask & (~np.isnan(two_theta_new))
    return two_theta_new, final_mask

def process_and_save(angles, intensities):
    """Filter, sort, and normalize peak data."""
    valid_range = (angles >= 0.0) & (angles <= 90.0)
    angles = angles[valid_range]
    intensities = intensities[valid_range]
    
    if len(angles) == 0:
        warnings.warn("No valid peaks found in the 0-90 degree range.")
        return np.array([]), np.array([])

    # Tie-break on angle. The condition vector pairs thetas to intensities by position, so
    # two peaks of equal intensity swapping places changes what the model is conditioned on.
    sort_idx = np.lexsort((angles, -intensities))
    angles = angles[sort_idx][:MAX_PEAKS]
    intensities = intensities[sort_idx][:MAX_PEAKS]
    
    max_int = np.max(intensities)
    if max_int > 0:
        intensities = (intensities / max_int) * 100.0
        
    return angles, intensities

def process_and_convert(input_data, xrd_wavelength=TARGET_WAVELENGTH, peak_pick=None, verbose=True,
                        pick_options=None):
    """Load an XRD file, pick peaks if needed, convert to CuKa1, and keep the top 20.

    Args:
        input_data: Path to a two-column file of 2theta and intensity.
        xrd_wavelength: Wavelength of the data in Angstrom.
        peak_pick: ``None`` picks peaks only when the file looks like a continuous profile,
            ``True`` always picks, ``False`` treats every row as a peak (the pre-picked behaviour).
        verbose: Print a line when a profile is detected and reduced to peaks.
        pick_options: Optional keyword arguments for ``pick_peaks`` (``min_rel_height``,
            ``min_separation``, ``smoothing_fwhm``, ``background_window``).

    Returns:
        List of (2theta, intensity) tuples, strongest first, intensities scaled to 100.
    """
    raw_angles, raw_intensities = load_picked_peaks(input_data)
    
    if peak_pick is None:
        peak_pick = is_continuous_profile(raw_angles)
        if peak_pick and verbose:
            print(f"Detected a continuous XRD profile in {input_data} ({len(raw_angles)} points); "
                  f"picking peaks (disable peak picking to treat every row as a peak).")

    if peak_pick:
        order = np.argsort(raw_angles, kind="stable")
        raw_angles, raw_intensities = pick_peaks(raw_angles[order], raw_intensities[order],
                                                 **(pick_options or {}))
        if verbose:
            print(f"Picked {len(raw_angles)} peaks from {input_data}; keeping up to the strongest {MAX_PEAKS}.")
    elif len(raw_angles) > 250:
        warnings.warn(f"Loaded {len(raw_angles)} peaks, check if this is raw data rather than picked peaks.")
    
    conv_angles, mask = convert_wavelength(raw_angles, xrd_wavelength, TARGET_WAVELENGTH)
    processed_ang, processed_int = process_and_save(conv_angles[mask], raw_intensities[mask])
    
    # Return as list of tuples as expected by test suite logic
    return list(zip(processed_ang, processed_int))

def save_to_crystallm_csv(peaks, output_path):
    """Save processed peaks to the standard CrystaLLM CSV format."""
    with open(output_path, 'w') as f:
        f.write("2theta,intensity\n")
        for ang, inten in peaks:
            f.write(f"{ang:.4f},{inten:.2f}\n")


def peak_pick_mode(value):
    """Map an ``auto``/``on``/``off`` command-line choice to the ``peak_pick`` argument."""
    return {"auto": None, "on": True, "off": False}[value]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XRD peak format converter for CrystaLLM.")
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--xrd_wavelength', type=float, default=TARGET_WAVELENGTH)
    parser.add_argument('--peak_pick', choices=("auto", "on", "off"), default="auto",
                        help="Pick peaks from a continuous profile: auto-detect (default), always, or never.")
    parser.add_argument('--min_rel_height', type=float, default=PICK_MIN_REL_HEIGHT,
                        help="Minimum peak prominence relative to the strongest peak (default 0.02).")
    parser.add_argument('--min_separation', type=float, default=PICK_MIN_SEPARATION,
                        help="Minimum separation between peaks in degrees 2theta (default 0.2).")
    parser.add_argument('--smoothing_fwhm', type=float, default=PICK_SMOOTHING_FWHM,
                        help="Gaussian smoothing FWHM in degrees before picking (default 0.1).")
    parser.add_argument('--background_window', type=float, default=PICK_BACKGROUND_WINDOW,
                        help="Rolling-minimum background window in degrees (default 2.0).")
    
    args = parser.parse_args()

    pick_options = {
        "min_rel_height": args.min_rel_height,
        "min_separation": args.min_separation,
        "smoothing_fwhm": args.smoothing_fwhm,
        "background_window": args.background_window,
    }
    processed_peaks = process_and_convert(args.input_data, args.xrd_wavelength,
                                          peak_pick=peak_pick_mode(args.peak_pick),
                                          pick_options=pick_options)
    
    if not processed_peaks:
        raise ValueError("No valid peaks remained after filtering in the 0-90 degree range.")

    save_to_crystallm_csv(processed_peaks, args.output_csv)
    print(f"Process complete. Data saved to {args.output_csv}")
