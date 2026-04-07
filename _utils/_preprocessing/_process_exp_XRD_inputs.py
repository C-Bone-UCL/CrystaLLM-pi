"""
Format pre-picked XRD peaks for CrystaLLM-pi conditioning.

WARNING - ASSUMPTIONS:
This script assumes the input data contains ALREADY PICKED peaks (not raw diffraction profiles).
It also assumes that redundant secondary radiation peaks (like K-alpha2) have 
already been stripped or accounted for by the user prior to using this tool.
"""

import os
import argparse
import warnings
import numpy as np

# Global constants 
TARGET_WAVELENGTH = 1.54056
MAX_PEAKS = 20

def load_picked_peaks(input_data):
    """Load XRD two-column for csv/txt/xy/dat."""
    ext = os.path.splitext(input_data)[1].lower()
    
    if ext == '.csv':
        try:
            arr = np.loadtxt(input_data, delimiter=',', skiprows=1)
        except ValueError:
            arr = np.loadtxt(input_data, delimiter=',')
        
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, 0], arr[:, 1]
        
    for delim in (None, '\t', ',', ' '):
        try:
            arr = np.loadtxt(input_data, delimiter=delim)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 0], arr[:, 1]
        except Exception:
            continue
            
    raise ValueError(f"Could not parse input file {input_data}. Ensure it is 2 columns of numeric data.")

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

    sort_idx = np.argsort(intensities)[::-1]
    angles = angles[sort_idx][:MAX_PEAKS]
    intensities = intensities[sort_idx][:MAX_PEAKS]
    
    max_int = np.max(intensities)
    if max_int > 0:
        intensities = (intensities / max_int) * 100.0
        
    return angles, intensities

def process_and_convert(input_data, xrd_wavelength=TARGET_WAVELENGTH, peak_pick=False):
    """
    Public API for processing peaks. 
    Matches the signature expected by the test suite.
    """
    # Note: peak_pick is currently unused as the script assumes pre-picked peaks.
    raw_angles, raw_intensities = load_picked_peaks(input_data)
    
    if len(raw_angles) > 250:
        warnings.warn(f"Loaded {len(raw_angles)} peaks; check if this is raw data rather than picked peaks.")
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XRD peak format converter for CrystaLLM.")
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--xrd_wavelength', type=float, default=TARGET_WAVELENGTH)
    
    args = parser.parse_args()

    processed_peaks = process_and_convert(args.input_data, args.xrd_wavelength)
    
    if not processed_peaks:
        raise ValueError("No valid peaks remained after filtering in the 0-90 degree range.")

    save_to_crystallm_csv(processed_peaks, args.output_csv)
    print(f"Process complete. Data saved to {args.output_csv}")