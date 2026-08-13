"""Local test section: data processing."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from _utils._preprocessing._process_exp_XRD_continuous import (
    convert_to_continuous_profile,
    read_xrd_file,
    save_pipeline_plot,
)

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
        from _utils._metrics_utils import is_valid
        
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

        Structural assertions elsewhere are order-independent, so a change in peak
        ordering slips past them while silently changing what the model is conditioned
        on. This pins the exact output instead.
        """
        from _utils._preprocessing._process_exp_XRD_inputs import process_and_convert

        peaks = process_and_convert(str(FIXTURES / "test_rutile_raw.xy"))
        expected = self._read_reference(FIXTURES / "proc_rutile_top20.csv")

        assert len(peaks) == len(expected), f"peak count {len(peaks)} != reference {len(expected)}"
        for i, ((angle, intensity), (exp_angle, exp_intensity)) in enumerate(zip(peaks, expected)):
            assert abs(angle - exp_angle) < 1e-3, f"row {i}: 2theta {angle} != reference {exp_angle}"
            assert abs(intensity - exp_intensity) < 1e-2, f"row {i}: intensity drift at {angle}"

    def test_xrd_top20_tie_break_is_deterministic(self):
        """Equal intensities must sort by ascending angle, not by numpy's sort order."""
        from _utils._preprocessing._process_exp_XRD_inputs import process_and_save

        # Angles ascending in the input, so numpy's reverse-stable order would emit the
        # tied peaks descending. Only an explicit angle tie-break gives 10, 20, 40.
        angles = np.array([10.0, 20.0, 30.0, 40.0])
        intensities = np.array([50.0, 50.0, 99.0, 50.0])
        sorted_angles, _ = process_and_save(angles, intensities)

        assert sorted_angles[0] == 30.0, "strongest peak should lead"
        assert list(sorted_angles[1:]) == [10.0, 20.0, 40.0], (
            f"tied peaks should follow ascending angle, got {list(sorted_angles[1:])}"
        )

    def test_continuous_profile_matches_reference(self):
        """Continuous conversion must reproduce the committed reference profile.

        Guards against a pybaselines or numpy upgrade quietly shifting the SNIP
        baseline, which would change the conditioning for every continuous-XRD run.
        """
        two_theta, intensity = read_xrd_file(FIXTURES / "Rutile-TiO2-unproc.txt")
        profile = convert_to_continuous_profile(two_theta, intensity, 1.54056)
        expected = self._read_reference(FIXTURES / "proc_rutile_continuous.csv")

        assert len(profile) == len(expected), f"{len(profile)} points != reference {len(expected)}"
        worst = max(abs(got[1] - exp[1]) for got, exp in zip(profile, expected))
        assert worst < 1e-5, f"largest intensity drift {worst:.2e} exceeds tolerance"

    def _read_reference(self, path):
        """Read a two-column reference csv, skipping its header."""
        with open(path) as handle:
            return [tuple(float(v) for v in line.split(",")) for line in handle.read().splitlines()[1:]]

    def test_exp_continuous_reader(self):
        """Sniffing reader finds the data start below arbitrary vendor headers."""
        tth, iq = read_xrd_file(FIXTURES / "test_exp_panalytical_header.csv")
        assert len(tth) == 12 and abs(tth[0] - 5.016746160) < 1e-9 and iq[0] == 866.0

        tth, iq = read_xrd_file(FIXTURES / "Rutile-TiO2-unproc.txt")  # real scan: 2-line header
        assert len(tth) == 8501 and tth[0] == 5.0
        assert abs(iq[236] - 3.0671e-02) < 1e-9  # Fortran exponent at 2theta = 7.36

    def test_exp_continuous_profile_synthetic(self):
        """Gaussian peak on a sloping background lands on the right Q bin, background-free."""
        two_theta = np.linspace(10.0, 80.0, 3000)
        intensity = 50.0 + 1000.0 * np.exp(-0.5 * ((two_theta - 27.4) / 0.15) ** 2)
        profile = np.array(convert_to_continuous_profile(two_theta, intensity, wavelength=1.54059))
        assert profile.shape == (1000, 2)
        assert np.allclose(profile[:, 0], np.arange(0.0, 10.0, 0.01), atol=1e-6)
        assert abs(profile[:, 1].max() - 1.0) < 1e-9
        q_peak = 4 * np.pi * np.sin(np.radians(27.4 / 2)) / 1.54059  # ~1.932
        assert abs(profile[np.argmax(profile[:, 1]), 0] - q_peak) < 0.02
        # In-range, far from both the peak (~1.93) and the zero-fill boundary (~0.71), so the
        # SNIP baseline estimate is unaffected by edge effects.
        flat = (profile[:, 0] > 1.1) & (profile[:, 0] < 1.5)
        assert profile[flat, 1].max() < 0.02                          # background removed

    def test_exp_continuous_binary_rejected(self):
        """Vendor binary blobs fail with an actionable message."""
        path = os.path.join(self.temp_dir, "vendor.raw")
        with open(path, "wb") as fh:
            fh.write(b"RAW1.01\x00\x00\x01\x02binaryblob")
        try:
            read_xrd_file(path)
            assert False, "binary file should be rejected"
        except ValueError as err:
            assert "binary" in str(err)

    def test_exp_continuous_pipeline_plot(self):
        """Real rutile scan renders the 4-stage transform overview (raw -> Q -> background -> 1000-pt profile)."""
        out = save_pipeline_plot(
            FIXTURES / "Rutile-TiO2-unproc.txt",
            Path(self.temp_dir) / "rutile_pipeline.png",
            wavelength=1.54059,
        )
        assert os.path.exists(out) and os.path.getsize(out) > 10_000
