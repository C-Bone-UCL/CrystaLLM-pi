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
        from _models.Slider_model import SliderGPT
        from _models.Prefix_model import PrefixGPT
        from _models.PrefixXRD_model import PrefixXRDGPT
        from _models.Residual_model import ResidualGPT
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
        assert SliderGPT is not None, "Slider model import failed"
        assert PrefixGPT is not None, "Prefix model import failed"
        assert PrefixXRDGPT is not None, "PrefixXRD model import failed"
        assert ResidualGPT is not None, "Residual model import failed"
    
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
    
    def test_prefix_model_forward(self):
        """Test PrefixGPT forward pass with condition values."""
        from _models.Prefix_model import PrefixGPT, PrefixGPT2Config

        config = PrefixGPT2Config(
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

        model = PrefixGPT(config)
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
        assert outputs.logits.shape == (batch_size, seq_len, 1000), f"Prefix output shape mismatch: {outputs.logits.shape}"

    def test_residual_model_forward(self):
        """Test ResidualGPT forward pass with condition values."""
        from _models.Residual_model import ResidualGPT, ResidualGPT2Config

        config = ResidualGPT2Config(
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

        model = ResidualGPT(config)
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

        assert outputs.logits.shape == (batch_size, seq_len, 1000), f"Residual output shape mismatch: {outputs.logits.shape}"

    def _xrd_config(self, n_heads=2, n_prefix=4, hidden=64, n_hidden_cond=64):
        from _models.PrefixXRD_model import PrefixXRDGPT2Config
        # perceiver_heads * perceiver_dim_head must == n_hidden_cond
        return PrefixXRDGPT2Config(
            vocab_size=1000,
            n_positions=256,
            n_embd=hidden,
            n_layer=2,
            n_head=n_heads,
            n_prefix_tokens=n_prefix,
            n_hidden_cond=n_hidden_cond,
            perceiver_depth=1,
            perceiver_heads=2,
            perceiver_dim_head=n_hidden_cond // 2,
            perceiver_ff_mult=2,
            dropout=0.0,
            skip_xrd_convert_model=False,
        )

    def test_prefix_xrd_discrete_forward(self):
        """PrefixXRDGPT: discrete peaks path produces correct logit shape."""
        from _models.PrefixXRD_model import PrefixXRDGPT
        config = self._xrd_config()
        model = PrefixXRDGPT(config)
        model.eval()

        B, S, P = 2, 10, 3
        input_ids = torch.randint(0, 1000, (B, S))
        attn_mask = torch.ones(B, S)
        # discrete [Q, I] peaks — Q must be > 0 for non-padding
        peaks = torch.rand(B, P, 2).clamp(min=0.01)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, condition_values=peaks)

        assert out.logits.shape == (B, S, 1000), f"Unexpected shape: {out.logits.shape}"
        assert not torch.isnan(out.logits).any(), "NaN in XRD logits"

    def test_prefix_xrd_continuous_forward(self):
        """PrefixXRDGPT: skip_xrd_convert_model=True accepts (B, 1000, 2) [Q, I] input."""
        from _models.PrefixXRD_model import PrefixXRDGPT
        from _models.xrd_utils import QMIN, QMAX, QSTEP, NUM_Q_POINTS
        config = self._xrd_config()
        config.skip_xrd_convert_model = True
        model = PrefixXRDGPT(config)
        model.eval()

        B, S = 2, 10
        input_ids = torch.randint(0, 1000, (B, S))
        attn_mask = torch.ones(B, S)
        q_grid = torch.arange(QMIN, QMAX, QSTEP).unsqueeze(0).expand(B, -1)   # (B, 1000)
        iq = torch.rand(B, NUM_Q_POINTS)
        cond = torch.stack([q_grid, iq], dim=-1)   # (B, 1000, 2) as [Q, I]

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, condition_values=cond)

        assert out.logits.shape == (B, S, 1000), f"Unexpected shape: {out.logits.shape}"

    def test_prefix_xrd_continuous_wrong_grid_raises(self):
        """A continuous profile off the canonical Q grid is rejected, not silently encoded."""
        from _models.PrefixXRD_model import PrefixXRDGPT
        from _models.xrd_utils import QMIN, QMAX, QSTEP, NUM_Q_POINTS
        config = self._xrd_config()
        config.skip_xrd_convert_model = True
        model = PrefixXRDGPT(config)
        model.eval()

        # Same 1000 points, wrong spacing: a 0..20 grid instead of the canonical 0..10.
        bad_q = torch.arange(QMIN, QMAX, QSTEP).mul(2.0).unsqueeze(0)
        cond = torch.stack([bad_q, torch.rand(1, NUM_Q_POINTS)], dim=-1)

        try:
            with torch.no_grad():
                model(input_ids=torch.randint(0, 1000, (1, 5)), condition_values=cond)
            assert False, "Expected ValueError for an off-grid continuous profile"
        except ValueError as err:
            assert "Q grid" in str(err), f"Unexpected error: {err}"

    def test_prefix_xrd_no_condition_raises(self):
        """PrefixXRDGPT raises ValueError when condition_values is None."""
        from _models.PrefixXRD_model import PrefixXRDGPT
        config = self._xrd_config()
        model = PrefixXRDGPT(config)
        model.eval()

        try:
            model(input_ids=torch.randint(0, 1000, (1, 5)))
            assert False, "Expected ValueError"
        except ValueError:
            pass

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
