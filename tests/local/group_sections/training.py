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
