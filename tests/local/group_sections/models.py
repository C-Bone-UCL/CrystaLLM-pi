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
        # discrete [Q, I] peaks, Q must be > 0 for non-padding
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

    def test_positional_embedding_resize_shift_right(self):
        """Test positional embedding resize shifts pretrained rows right for Prefix-style loads."""
        from transformers import GPT2Config, GPT2LMHeadModel
        from _utils.model import resize_positional_embeddings

        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=4,
            n_layer=1,
            n_head=1,
        )
        model = GPT2LMHeadModel(config)
        with torch.no_grad():
            model.transformer.wpe.weight.copy_(
                torch.arange(32, dtype=torch.float32).view(8, 4)
            )

        resized = resize_positional_embeddings(model, 12, shift_right_by=4)

        assert resized.config.n_positions == 12
        assert resized.transformer.wpe.num_embeddings == 12
        assert torch.equal(
            resized.transformer.wpe.weight.data[4:12],
            torch.arange(32, dtype=torch.float32).view(8, 4),
        ), "Old rows should be copied to the right-shifted slice"
        assert resized.transformer.wpe.weight.data[:4].shape == (4, 4)
        assert torch.isfinite(resized.transformer.wpe.weight.data[:4]).all()

    def test_positional_embedding_resize_marks_context_extension(self):
        """True long-context growth records the copied rows to protect."""
        from transformers import GPT2Config, GPT2LMHeadModel
        from _utils.model import resize_positional_embeddings

        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=4,
            n_layer=1,
            n_head=1,
        )
        model = GPT2LMHeadModel(config)

        resized = resize_positional_embeddings(model, 14, shift_right_by=4)
        metadata = resized.context_extension_metadata

        assert metadata["is_context_extension"] is True
        assert metadata["protected_wpe_row_ranges"] == [(4, 12)]
        assert metadata["source_n_positions"] == 8
        assert metadata["target_n_positions"] == 14

    def test_positional_embedding_resize_marks_prefix_conversion_only(self):
        """Prefix slots alone should not activate context-extension warmup."""
        from transformers import GPT2Config, GPT2LMHeadModel
        from _utils.model import resize_positional_embeddings

        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=4,
            n_layer=1,
            n_head=1,
        )
        model = GPT2LMHeadModel(config)

        resized = resize_positional_embeddings(model, 12, shift_right_by=4)
        metadata = resized.context_extension_metadata

        assert metadata["is_context_extension"] is False
        assert metadata["protected_wpe_row_ranges"] == [(4, 12)]

    def _run_prefix_load_shift_probe(self, source_config_dict):
        """Drive load_pretrained_model with dummy classes and return the recorded resize call."""
        from types import SimpleNamespace
        import _utils.model as model_utils

        calls = {}

        class DummyConfig:
            def __init__(self, n_positions=8):
                self.n_positions = n_positions

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(n_positions=8)

            @classmethod
            def get_config_dict(cls, *args, **kwargs):
                return (source_config_dict, None)

        class DummyModel:
            def __init__(self, config):
                self.config = config
                self.transformer = SimpleNamespace(
                    wpe=torch.nn.Embedding(config.n_positions, 4)
                )

            @classmethod
            def from_pretrained(cls, ckpt_dir, config=None, **kwargs):
                return cls(config)

            def resize_token_embeddings(self, vocab_size):
                self.vocab_size = vocab_size
                return self

        def fake_loader(model_class, ckpt_dir, config, **kwargs):
            return model_class.from_pretrained(ckpt_dir, config=config), {"mismatched_keys": []}

        def fake_resize(model, new_n_positions, shift_right_by=0):
            calls["new_n_positions"] = new_n_positions
            calls["shift_right_by"] = shift_right_by
            return model

        # Restore by mutating the entry in place: generate_cifs imports MODEL_REGISTRY by
        # name, so rebinding the module attribute would leak the dummy entry to it.
        original_entry = model_utils.MODEL_REGISTRY["Prefix"]
        original_loader = model_utils._load_with_sdpa_fallback
        original_resize = model_utils.resize_positional_embeddings
        try:
            model_utils.MODEL_REGISTRY["Prefix"] = (DummyConfig, DummyModel)
            model_utils._load_with_sdpa_fallback = fake_loader
            model_utils.resize_positional_embeddings = fake_resize

            args = SimpleNamespace(
                pretrained_model_dir="dummy_ckpt",
                activate_conditionality="Prefix",
                context_length=8,
                n_prefix_tokens=4,
                n_hidden_cond=32,
                condition_columns="['bandgap']",
            )

            class DummyTokenizer:
                bos_token_id = None
                eos_token_id = None
                pad_token_id = None

                def __len__(self):
                    return 16

            model = model_utils.load_pretrained_model(args, DummyTokenizer())
        finally:
            model_utils.MODEL_REGISTRY["Prefix"] = original_entry
            model_utils._load_with_sdpa_fallback = original_loader
            model_utils.resize_positional_embeddings = original_resize

        assert model is not None
        return calls

    def test_prefix_load_resizes_wpe_with_shift(self):
        """Base checkpoint -> Prefix target should request a right shift."""
        calls = self._run_prefix_load_shift_probe(source_config_dict={})
        assert calls["new_n_positions"] == 12
        assert calls["shift_right_by"] == 4

    def test_prefix_load_resizes_wpe_without_shift_for_prefix_source(self):
        """Prefix checkpoint -> Prefix target should not request an extra shift."""
        calls = self._run_prefix_load_shift_probe(
            source_config_dict={"n_prefix_tokens": 4, "architectures": ["PrefixGPT"]}
        )
        assert calls["new_n_positions"] == 12
        assert calls["shift_right_by"] == 0

    def test_pretrained_load_shape_check(self):
        """Loading a checkpoint with mismatched conditioning widths raises instead of silently re-initializing."""
        import os
        from types import SimpleNamespace
        from _models.PrefixXRD_model import PrefixXRDGPT
        from _utils.model import load_pretrained_model

        ckpt_dir = os.path.join(self.temp_dir, "tiny_prefixxrd_ckpt")
        model = PrefixXRDGPT(self._xrd_config())  # n_hidden_cond=64 (2 heads x 32)
        model.save_pretrained(ckpt_dir)

        class DummyTokenizer:
            bos_token_id = None
            eos_token_id = None
            pad_token_id = None

            def __len__(self):
                return 1000

        args = SimpleNamespace(
            pretrained_model_dir=ckpt_dir,
            activate_conditionality="PrefixXRD",
            context_length=252,        # + n_prefix_tokens = checkpoint's 256, so no wpe resize
            n_prefix_tokens=4,
            n_hidden_cond=32,          # checkpoint used 64 -> conditioning shapes mismatch
            perceiver_depth=1,
            perceiver_n_heads=2,
            perceiver_dim_head=16,     # heads * dim_head must equal the mutated n_hidden_cond
            perceiver_ff_mult=2,
            skip_xrd_convert_model=False,
            cond_dropout=0.0,
        )

        try:
            load_pretrained_model(args, DummyTokenizer())
            assert False, "Expected ValueError for mismatched conditioning weights"
        except ValueError as err:
            assert "does not fit" in str(err), f"Unexpected error: {err}"

    def test_context_extension_warmup_freeze(self):
        """Copied wpe rows stay frozen during warmup steps and train afterwards."""
        from types import SimpleNamespace
        from transformers import GPT2Config, GPT2LMHeadModel
        from _utils.model import resize_positional_embeddings
        from _utils.trainer import ContextExtensionWarmupCallback, has_context_extension_wpe

        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=8, n_embd=4, n_layer=1, n_head=1))
        # 8 -> 14 with shift 4: rows 4-12 are checkpoint-copied, rows beyond are true extension
        model = resize_positional_embeddings(model, 14, shift_right_by=4)
        assert has_context_extension_wpe(model)

        callback = ContextExtensionWarmupCallback(context_extension_warmup_steps=1)
        state = SimpleNamespace(global_step=0)
        callback.on_train_begin(None, state, None, model=model)

        wpe = model.transformer.wpe.weight

        def _wpe_grad():
            model.zero_grad()
            input_ids = torch.randint(0, 32, (1, 14))
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            return wpe.grad

        # Step 0 (< warmup): protected rows masked, other rows receive gradient
        grad = _wpe_grad()
        assert torch.all(grad[4:12] == 0), "protected rows should have zero grad during warmup"
        assert torch.any(grad[:4] != 0) or torch.any(grad[12:] != 0), \
            "unprotected rows should receive gradient"

        # Past warmup: gradient flows into the protected rows again
        state.global_step = 1
        grad = _wpe_grad()
        assert torch.any(grad[4:12] != 0), "protected rows should train after warmup"

        callback.on_train_end(None, state, None)

    def test_context_extension_warmup_ignores_prefix_conversion_only(self):
        """A Prefix conversion shifts wpe without extending it, so no rows get protected."""
        from types import SimpleNamespace
        from _utils.trainer import ContextExtensionWarmupCallback

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.wpe = torch.nn.Embedding(6, 3)
                # is_context_extension False is what a shift-only resize records.
                self.context_extension_metadata = {
                    "is_context_extension": False,
                    "protected_wpe_row_ranges": [(2, 6)],
                }

            def forward(self):
                return self.transformer.wpe(torch.arange(6)).sum()

        model = TinyModel()
        callback = ContextExtensionWarmupCallback(context_extension_warmup_steps=2)
        callback.on_train_begin(None, SimpleNamespace(global_step=0), None, model=model)
        model().backward()

        assert torch.all(model.transformer.wpe.weight.grad != 0), \
            "prefix-only resize must not mask any gradients"

    def _muon_args(self):
        """Minimal args namespace for exercising the Muon parameter grouping."""
        from types import SimpleNamespace

        return SimpleNamespace(
            optimizer="muon", max_steps=2, warmup_steps=0, warmup_ratio=0.0,
            lr_scheduler_kwargs={}, lr_scheduler_type="constant",
            muon_lr=1e-3, muon_momentum=0.95, weight_decay=0.0,
            learning_rate=1e-4, adam_beta1=0.9, adam_beta2=0.95,
            cond_lr=None, cond_wd=None, activate_conditionality=None,
        )

    def test_muon_routes_wpe_to_adamw_for_context_extension(self):
        """Muon orthogonalizes whole matrices, so extended wpe must sit in the AdamW group.

        Without this the zeroed gradient rows would still be moved by the Muon update, defeating the
        warmup mask entirely.
        """
        from _utils.trainer import setup_scheduler

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.wpe = torch.nn.Embedding(6, 3)
                self.hidden = torch.nn.Linear(3, 3, bias=False)
                self.context_extension_metadata = {
                    "is_context_extension": True,
                    "protected_wpe_row_ranges": [(0, 4)],
                }

        model = TinyModel()
        optimizer, _ = setup_scheduler(self._muon_args(), model)
        wpe_parameter = model.transformer.wpe.weight

        assert any(p is wpe_parameter for p in optimizer.param_groups[1]["params"]), \
            "extended wpe should be in the AdamW group"
        assert not any(p is wpe_parameter for p in optimizer.param_groups[0]["params"]), \
            "extended wpe should not be in the Muon group"

    def test_muon_keeps_wpe_grouping_without_context_extension(self):
        """Ordinary runs keep wpe on Muon, which is how every released model was trained."""
        from _utils.trainer import setup_scheduler

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = torch.nn.Module()
                self.transformer.wpe = torch.nn.Embedding(6, 3)
                self.hidden = torch.nn.Linear(3, 3, bias=False)

        model = TinyModel()
        optimizer, _ = setup_scheduler(self._muon_args(), model)
        wpe_parameter = model.transformer.wpe.weight

        assert any(p is wpe_parameter for p in optimizer.param_groups[0]["params"]), \
            "wpe should stay on Muon without context extension"
        assert not any(p is wpe_parameter for p in optimizer.param_groups[1]["params"])

    def test_legacy_training_rail(self):
        """build_model refuses the legacy PKV/Slider families with a pointer to the successors."""
        from types import SimpleNamespace
        from _utils.model import build_model

        class DummyTokenizer:
            bos_token_id = None
            eos_token_id = None
            pad_token_id = None

            def __len__(self):
                return 100

        for legacy in ("PKV", "Slider"):
            try:
                build_model(SimpleNamespace(activate_conditionality=legacy), DummyTokenizer())
                assert False, f"Expected ValueError for legacy family {legacy}"
            except ValueError as err:
                assert "legacy" in str(err), f"Unexpected error: {err}"

    def test_train_data_mode_covers_every_registry_family(self):
        """_train.py's data dispatch must stay in step with MODEL_REGISTRY."""
        from _utils.model import (
            LEGACY_FAMILIES,
            MODEL_REGISTRY,
            TRAINABLE_CONDITIONAL_FAMILIES,
            resolve_data_mode,
        )

        # Every trainable family must reach the conditional dataloader. Before this guard,
        # _train.py listed PKV/Slider by hand, so Prefix, PrefixXRD and Residual matched no
        # branch and training died on an unassigned tokenized_dataset.
        for family in TRAINABLE_CONDITIONAL_FAMILIES:
            mode = resolve_data_mode(family)
            assert mode == "conditional", f"{family} resolved to {mode!r}, expected 'conditional'"

        assert set(TRAINABLE_CONDITIONAL_FAMILIES) == {
            name for name in MODEL_REGISTRY if name and name not in LEGACY_FAMILIES
        }, "TRAINABLE_CONDITIONAL_FAMILIES drifted from MODEL_REGISTRY"

        for unconditional in ("None", None):
            mode = resolve_data_mode(unconditional)
            assert mode == "unconditional", f"{unconditional!r} resolved to {mode!r}"

        for legacy in LEGACY_FAMILIES:
            try:
                resolve_data_mode(legacy)
                assert False, f"Expected ValueError for legacy family {legacy}"
            except ValueError as err:
                assert "legacy" in str(err), f"Unexpected error for {legacy}: {err}"

        try:
            resolve_data_mode("NotAFamily")
            assert False, "Expected ValueError for an unknown family"
        except ValueError as err:
            assert "Unknown activate_conditionality" in str(err), f"Unexpected error: {err}"

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
