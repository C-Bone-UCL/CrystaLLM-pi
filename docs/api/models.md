# Models

Three conditional families plus the shared Perceiver blocks and XRD helpers. Each family is a
config, an encoder that turns the conditioning into something the transformer reads, and a
`forward()` whose docstring names every tensor shape.

Legacy `PKV` and `Slider` are load-only and are not listed here; see
[Conventions](conventions.md#legacy-models).

## Prefix

Scalar conditioning prepended as prefix key-values.

::: _models.Prefix_model.PrefixGPT2Config
::: _models.Prefix_model.PrefixEncoder
::: _models.Prefix_model.PrefixGPT
::: _models.Prefix_model.reshape_prefix_kv_to_past_key_values
::: _models.Prefix_model.prepend_prefix_attention_mask

## PrefixXRD

Diffraction conditioning through a Perceiver resampler. Takes either discrete `[Q, I]` peaks or a
continuous 1000-point profile, selected by `skip_xrd_convert_model`.

::: _models.PrefixXRD_model.PrefixXRDGPT2Config
::: _models.PrefixXRD_model.XRDPerceiverEncoder
::: _models.PrefixXRD_model.PrefixXRDGPT

## Residual

Conditioning injected as a residual stream contribution inside each block.

::: _models.Residual_model.ResidualGPT2Config
::: _models.Residual_model.ResidualEncoder
::: _models.Residual_model.ResidualAttention
::: _models.Residual_model.ResidualGPT2Block
::: _models.Residual_model.ResidualGPT2Model
::: _models.Residual_model.ResidualGPT

## Perceiver blocks

Shared by the XRD encoder.

::: _models.perceiver.PerceiverFeedForward
::: _models.perceiver.PerceiverAttention
::: _models.perceiver.PerceiverResampler

## XRD helpers

::: _models.xrd_utils.condition_vector_to_continuous_xrd
::: _models.xrd_utils.discrete_to_continuous_xrd
::: _models.xrd_utils.save_xrd_pipeline_plots
::: _models.xrd_utils.xrd_debug_check

## Loading and building

::: _utils.model.load_pretrained_model
::: _utils.model.build_model
::: _utils.model.resolve_data_mode
::: _utils.model.configure_runtime_model_flags
::: _utils.model.resize_positional_embeddings
