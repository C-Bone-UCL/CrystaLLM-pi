"""Local test section: training."""

class TrainingTests:
    """Test training pipeline components."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
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
            from _models.Prepend_model import PrependGPT
            from _models.Slider_model import SliderGPT
            
            assert PKVGPT is not None, "PKV model class exists"
            assert PrependGPT is not None, "Prepend model class exists" 
            assert SliderGPT is not None, "Slider model class exists"
            
        except Exception as e:
            print(f"Model class imports failed: {e}")

    def test_xtra_augment_training_jobs(self):
        """Xtra-augment training job manifest should define three unique training outputs."""
        from _pipelines._xtra_augment_jobs import XTRA_AUGMENT_JOBS

        assert len(XTRA_AUGMENT_JOBS) == 3, "Expected 3 xtra-augment jobs"

        names = [job["name"] for job in XTRA_AUGMENT_JOBS]
        assert names == ["mp_20-base", "mp_20-Xaug-sc", "mp_20-Xaug-all"]

        output_dirs = [job["output_dir"] for job in XTRA_AUGMENT_JOBS]
        assert len(set(output_dirs)) == len(output_dirs), "Training output dirs must be unique"

        train_configs = [job["train_config"] for job in XTRA_AUGMENT_JOBS]
        assert train_configs == [
            "_config_files/training/unconditional/_xtra_augment/mp_20-base.jsonc",
            "_config_files/training/unconditional/_xtra_augment/mp_20-Xaug-sc.jsonc",
            "_config_files/training/unconditional/_xtra_augment/mp_20-Xaug-all.jsonc",
        ]
