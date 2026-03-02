"""Local test section: models."""

import torch

class ModelTests:
    """Test model loading and basic operations."""
    
    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data
    
    def test_model_loading(self):
        """Test model class imports and initialization."""
        from _models.PKV_model import PKVGPT
        from _models.Prepend_model import PrependGPT
        from _models.Slider_model import SliderGPT
        from transformers import GPT2LMHeadModel
        
        # Test basic GPT2 config
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2
        )
        
        # Test unconditional model
        model = GPT2LMHeadModel(config)
        assert model is not None, "Failed to create base model"
        
        # Test conditional models can be imported
        assert PKVGPT is not None, "PKV model import failed"
        assert PrependGPT is not None, "Prepend model import failed" 
        assert SliderGPT is not None, "Slider model import failed"
    
    def test_model_forward(self):
        """Test basic model forward pass."""
        from transformers import GPT2LMHeadModel, GPT2Config
        
        config = GPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2
        )
        
        model = GPT2LMHeadModel(config)
        model.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits.shape == (1, 10, 1000), "Unexpected output shape"
    
    def test_pkv_model_forward(self):
        """Test PKVGPT forward pass with condition values."""
        from _models.PKV_model import PKVGPT, PKVGPT2Config
        
        config = PKVGPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2,
            n_input_vector=2,
            n_prefix_tokens=4,
            n_hidden_cond=64,
            dropout=0.1,
            share_layers=False
        )
        
        model = PKVGPT(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        condition_values = torch.rand(batch_size, 2)  # 2 conditions
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition_values=condition_values
            )
        
        # Output shape should match input sequence length
        assert outputs.logits.shape == (batch_size, seq_len, 1000), f"PKV output shape mismatch: {outputs.logits.shape}"
    
    def test_prepend_model_forward(self):
        """Test PrependGPT forward pass with condition values."""
        from _models.Prepend_model import PrependGPT, PrependGPT2Config
        
        config = PrependGPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2,
            n_input_vector=2,
            n_prefix_tokens=4,
            n_hidden_cond=64,
            dropout=0.1
        )
        
        model = PrependGPT(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        condition_values = torch.rand(batch_size, 2)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition_values=condition_values
            )
        
        # PrependGPT slices off prefix tokens from output, so logits should be seq_len
        assert outputs.logits.shape == (batch_size, seq_len, 1000), f"Prepend output shape mismatch: {outputs.logits.shape}"
    
    def test_slider_model_forward(self):
        """Test SliderGPT forward pass with condition values."""
        from _models.Slider_model import SliderGPT, SliderGPT2Config
        
        config = SliderGPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=4,  # Must be divisible by slider_n_heads_sharing_slider
            slider_on=True,
            slider_n_variables=2,
            slider_n_hidden=64,
            slider_n_heads_sharing_slider=2,
            slider_dropout=0.1
        )
        
        model = SliderGPT(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        condition_values = torch.rand(batch_size, 2)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                condition_values=condition_values
            )
        
        assert outputs.logits.shape == (batch_size, seq_len, 1000), f"Slider output shape mismatch: {outputs.logits.shape}"
    
    def test_conditional_model_with_labels(self):
        """Test conditional models compute loss when labels provided."""
        from _models.PKV_model import PKVGPT, PKVGPT2Config
        
        config = PKVGPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=128,
            n_layer=2,
            n_head=2,
            n_input_vector=2,
            n_prefix_tokens=4,
            n_hidden_cond=64
        )
        
        model = PKVGPT(config)
        model.train()
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        condition_values = torch.rand(batch_size, 2)
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            condition_values=condition_values,
            labels=labels
        )
        
        assert outputs.loss is not None, "Model should compute loss when labels provided"
        assert outputs.loss.item() > 0, "Loss should be positive"
