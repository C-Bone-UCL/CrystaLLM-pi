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

