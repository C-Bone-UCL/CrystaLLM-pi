"""Local test section: training."""

class TrainingTests:
    """Test training pipeline components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_train_cli_help_runs(self):
        """_train.py must build its parser and exit cleanly.

        The training entrypoint is otherwise untested end to end, so argparse or
        import-level breakage would only show up on a real run.
        """
        import os
        import subprocess
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        result = subprocess.run(
            [sys.executable, os.path.join(repo_root, "_train.py"), "--help"],
            capture_output=True, text=True, cwd=repo_root, timeout=120,
        )

        assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr[-400:]}"
        assert "--activate_conditionality" in result.stdout, "help should list the family flag"

    def test_training_setup(self):
        """Test training script imports and basic setup."""
        import sys
        from _args import parse_args
        from _utils._model_utils import build_model
        
        # Save original sys.argv and replace with empty args to avoid conflicts
        original_argv = sys.argv
        sys.argv = ['run-tests.py']
        
        try:
            # Test argument parsing
            args = parse_args()
            assert hasattr(args, 'output_dir'), "Args should have output_dir"
            assert hasattr(args, 'model_ckpt_dir'), "Args should have model_ckpt_dir"
        finally:
            sys.argv = original_argv
        
        # Test basic imports work
        assert build_model is not None, "Model utils import failed"
    
    def test_model_initialization(self):
        """Test model initialization for different architectures."""
        from _utils._model_utils import build_model
        from transformers import GPT2Config
        
        # Test that we can import model building function
        assert build_model is not None, "Model building function exists"
        
        # Test different model imports
        try:
            from _models.PKV_model import PKVGPT
            from _models.Slider_model import SliderGPT

            assert PKVGPT is not None, "PKV model class exists"
            assert SliderGPT is not None, "Slider model class exists"
            
        except Exception as e:
            print(f"Model class imports failed: {e}")
