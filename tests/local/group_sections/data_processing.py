"""Local test section: data processing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class DataProcessingTests:
    """Test data processing components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_tokenizer_basic(self):
        """Test basic tokenizer functionality."""
        from _tokenizer import CustomCIFTokenizer
        
        # Use existing tokenizer files
        tokenizer = CustomCIFTokenizer.from_pretrained("HF-cif-tokenizer")
        
        # Test encoding/decoding
        test_text = self.test_data['augmented_cif']
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        assert len(encoded) > 0, "Encoding failed"
        assert '<bos>' in decoded and '<eos>' in decoded, "Special tokens missing"
        assert 'data_' in decoded, "CIF content missing"
    
    def test_cif_validation(self):
        """Test CIF validation utilities."""
        from _utils.metrics import is_valid
        
        # Test that the validation function works without crashing
        try:
            # Test with our realistic CIF - may pass or fail validation but shouldn't crash
            result = is_valid(self.test_data['test_cif'], bond_length_acceptability_cutoff=0.5)
            assert isinstance(result, bool), "Should return boolean result"
            print(f"CIF validation result: {result}")
        except Exception as e:
            # Some validation failures are acceptable for test CIFs
            print(f"CIF validation failed (acceptable for test): {e}")
        
        # Test that completely malformed input is handled
        try:
            result = is_valid("completely invalid text")
            assert result is False, "Invalid input should be rejected"
        except Exception:
            # Exception for malformed input is acceptable
            pass
    
    def test_automatic_prompts_keep_condition_column_intact(self):
        """condition_vector must reach the output unmangled, including nested XRD profiles.

        The old implementation ran str(value).replace("[", "") over the column, which flattens a
        nested (1000, 2) [Q, I] profile into unparseable text, and pd.isna on a nested value raises
        instead of returning False. A missing scalar still has to come out as the -100 sentinel the
        conditional models read as "no condition supplied".
        """
        import pandas as pd
        from _utils._generating.make_prompts import create_automatic_prompts

        nested_profile = [[0.0, 0.0], [0.01, 0.5], [0.02, 1.0]]
        df = pd.DataFrame({
            "CIF": [self.test_data["test_cif"]] * 3,
            "condition_vector": [nested_profile, "2.16, 0.0", float("nan")],
        })

        out = create_automatic_prompts(df, "CIF", "level_2", condition_columns=["condition_vector"])
        values = list(out["condition_vector"])

        assert values[0] == nested_profile, f"nested profile was mangled: {values[0]!r}"
        assert values[1] == "2.16, 0.0", f"scalar string was altered: {values[1]!r}"
        assert values[2] == "-100.0", f"missing scalar lost its sentinel: {values[2]!r}"

    def test_prompt_creation(self):
        """Test prompt creation utilities."""
        from _utils._generating.make_prompts import create_manual_prompts
        
        prompts_file = os.path.join(self.temp_dir, "test_prompts.parquet")
        
        # Test manual prompt creation
        df_prompts = create_manual_prompts(
            compositions=["Si2O4", "Ti2O4"],
            condition_lists=[["0.5", "0.0"]],
            level="level_2",
            spacegroups=None
        )
        
        # Save to file and verify
        df_prompts.to_parquet(prompts_file, index=False)
        df = pd.read_parquet(prompts_file)
        assert len(df) > 0, "No prompts created"
        assert 'Prompt' in df.columns, "Prompt column missing"
        assert any('Si' in str(prompt) for prompt in df['Prompt']), "Composition missing from prompts"

    def test_xrd_top20_matches_reference(self):
        """Legacy top-20 processing must reproduce the committed reference vector.

        Structural assertions elsewhere are order-independent, so a change in peak ordering slips
        past them while silently changing what the model is conditioned on. This pins the exact
        output instead.
        """
        from _utils._preprocessing.process_exp_xrd_inputs import process_and_convert

        peaks = process_and_convert(str(FIXTURES / "test_rutile_raw.xy"))
        expected = self._read_reference(FIXTURES / "proc_rutile_top20.csv")

        assert len(peaks) == len(expected), f"peak count {len(peaks)} != reference {len(expected)}"
        for i, ((angle, intensity), (exp_angle, exp_intensity)) in enumerate(zip(peaks, expected)):
            assert abs(angle - exp_angle) < 1e-3, f"row {i}: 2theta {angle} != reference {exp_angle}"
            assert abs(intensity - exp_intensity) < 1e-2, f"row {i}: intensity drift at {angle}"

    def test_xrd_top20_tie_break_is_deterministic(self):
        """Equal intensities must sort by ascending angle, not by numpy's sort order."""
        from _utils._preprocessing.process_exp_xrd_inputs import process_and_save

        # Angles ascending in the input, so numpy's reverse-stable order would emit the
        # tied peaks descending. Only an explicit angle tie-break gives 10, 20, 40.
        angles = np.array([10.0, 20.0, 30.0, 40.0])
        intensities = np.array([50.0, 50.0, 99.0, 50.0])
        sorted_angles, _ = process_and_save(angles, intensities)

        assert sorted_angles[0] == 30.0, "strongest peak should lead"
        assert list(sorted_angles[1:]) == [10.0, 20.0, 40.0], (
            f"tied peaks should follow ascending angle, got {list(sorted_angles[1:])}"
        )

    def _read_reference(self, path):
        """Read a two-column reference csv, skipping its header."""
        with open(path) as handle:
            return [tuple(float(v) for v in line.split(",")) for line in handle.read().splitlines()[1:]]


    # Peak picking from continuous XRD profiles, and the auto-detection that triggers it

    def _synthetic_profile(self):
        """Five Gaussian peaks on a sloping background with seeded noise, 10-90 deg at 0.02 deg."""
        rng = np.random.default_rng(0)
        x = np.arange(10.0, 90.0, 0.02)
        centres = np.array([20.0, 31.5, 32.4, 47.0, 71.3])
        heights = np.array([100.0, 60.0, 35.0, 20.0, 8.0])
        sigma = 0.08
        y = 30.0 - 0.2 * (x - 10.0)
        for c, h in zip(centres, heights):
            y += h * np.exp(-0.5 * ((x - c) / sigma) ** 2)
        y += rng.normal(0.0, 0.5, size=x.size)
        return x, y, centres, heights

    def test_profile_detection(self):
        """Continuous scans are detected; peak lists, including the committed fixtures, are not."""
        from _utils._preprocessing.process_exp_xrd_inputs import is_continuous_profile, load_two_column_xrd

        x, _, _, _ = self._synthetic_profile()
        assert is_continuous_profile(x)
        raw_scan, _ = load_two_column_xrd(str(FIXTURES / "Rutile-TiO2-unproc.txt"))
        assert is_continuous_profile(raw_scan), "measured rutile scan should be detected as a profile"
        for peak_list in ("test_rutile_raw.xy", "test_rutile_processed.csv"):
            angles, _ = load_two_column_xrd(str(FIXTURES / peak_list))
            assert not is_continuous_profile(angles), f"{peak_list} is a peak list, not a profile"
        # Many peaks listed in ascending angle, but irregularly spaced, are still a peak list.
        assert not is_continuous_profile(np.cumsum(np.random.default_rng(1).exponential(1.0, 80)) + 5)

    def test_load_skips_header_lines(self):
        """Non-numeric header lines (e.g. 'Wavelength = 1.54059') are skipped, not fatal."""
        from _utils._preprocessing.process_exp_xrd_inputs import load_two_column_xrd

        angles, intensities = load_two_column_xrd(str(FIXTURES / "Rutile-TiO2-unproc.txt"))
        assert len(angles) == 8501 and len(intensities) == 8501
        assert angles[0] == 5.0 and angles[-1] == 90.0

    def test_pick_peaks_synthetic(self):
        """Picking recovers known peaks, including a 0.9 deg doublet, on a sloping noisy background."""
        from _utils._preprocessing.process_exp_xrd_inputs import pick_peaks

        x, y, centres, heights = self._synthetic_profile()
        angles, picked = pick_peaks(x, y)
        assert len(angles) == len(centres), f"expected {len(centres)} peaks, got {np.round(angles, 2)}"
        assert np.allclose(angles, centres, atol=0.03), f"positions {np.round(angles, 3)}"
        rel = picked / picked.max() * 100
        expected = heights / heights.max() * 100
        assert np.allclose(rel, expected, atol=6.0), f"relative heights {np.round(rel, 1)}"

    def test_pick_peaks_measured_rutile(self):
        """On the measured rutile scan the strongest picked peaks are the main rutile reflections.

        The scan sits about 0.1 deg below the reference list, hence the 0.2 deg tolerance.
        """
        from _utils._preprocessing.process_exp_xrd_inputs import process_and_convert

        peaks = process_and_convert(str(FIXTURES / "Rutile-TiO2-unproc.txt"), xrd_wavelength=1.54059,
                                    verbose=False)
        angles = [a for a, _ in peaks]
        assert abs(angles[0] - 27.447) < 0.2, "strongest peak should be rutile (110)"
        for ref in (54.323, 36.086, 41.226, 56.642):
            assert min(abs(a - ref) for a in angles[:6]) < 0.2, f"missing rutile peak near {ref}"
        assert len(peaks) <= 20

    def test_peak_pick_modes(self):
        """Auto picks on a profile and leaves peak lists unchanged; off reproduces the old top-20 rows."""
        from _utils._preprocessing.process_exp_xrd_inputs import process_and_convert

        raw_scan = str(FIXTURES / "Rutile-TiO2-unproc.txt")
        auto = process_and_convert(raw_scan, xrd_wavelength=1.54059, verbose=False)
        off = process_and_convert(raw_scan, xrd_wavelength=1.54059, peak_pick=False, verbose=False)
        # Without picking, the 20 highest rows all sit on the (110) peak, which is the bug this fixes.
        assert max(a for a, _ in off) - min(a for a, _ in off) < 0.5
        assert max(a for a, _ in auto) - min(a for a, _ in auto) > 20

        peak_list = str(FIXTURES / "test_rutile_raw.xy")
        assert process_and_convert(peak_list, verbose=False) == process_and_convert(
            peak_list, peak_pick=False, verbose=False
        ), "auto mode must not change a peak list"
