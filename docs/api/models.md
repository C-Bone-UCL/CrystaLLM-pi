# Models

CrystaLLM-pi supports two conditioning mechanisms for crystal structure generation: prefix conditioning and residual conditioning. The tokenizer and datasets used to train these models are documented on the [Data](data.md) page.

Prefix conditioning has two variants that use the same underlying mechanism. `Prefix` accepts scalar properties directly, while `PrefixXRD` uses a Perceiver resampler to encode a full diffraction pattern before passing it to the model.

Each conditional model family includes a configuration, an encoder that maps the conditioning inputs to the representation expected by the transformer, and a GPT-2 model that applies the conditioning.

Tensor shapes and `forward()` behaviour are documented in the source docstrings and are available through the linked source or `help()`.

The earlier `PKV` and `Slider` classes used in the paper are retained for loading released checkpoints. They implement earlier versions of the same conditioning mechanisms and are not part of the current model API, so they are not included below.

## Prefix

Prefix conditioning represents scalar properties as prefix key-values for GPT-2.

::: _models.Prefix_model.PrefixGPT2Config
::: _models.Prefix_model.PrefixEncoder
::: _models.Prefix_model.PrefixGPT

## PrefixXRD

PrefixXRD conditions GPT-2 on XRD data using a Perceiver resampler.

The model accepts either discrete `[Q, I]` peaks or a continuous 1000-point `[Q, I]` profile. The input representation is determined by `skip_xrd_convert_model`.

::: _models.PrefixXRD_model.PrefixXRDGPT2Config
::: _models.PrefixXRD_model.XRDPerceiverEncoder
::: _models.PrefixXRD_model.PrefixXRDGPT

Discrete peaks can be converted to the continuous profile expected by the encoder with:

::: _models.xrd_utils.discrete_to_continuous_xrd

## Residual

Residual conditioning adds the conditioning signal to the attention blocks as a residual contribution.

::: _models.Residual_model.ResidualGPT2Config
::: _models.Residual_model.ResidualEncoder
::: _models.Residual_model.ResidualAttention
::: _models.Residual_model.ResidualGPT2Block
::: _models.Residual_model.ResidualGPT2Model
::: _models.Residual_model.ResidualGPT

## Perceiver resampler

The Perceiver resampler is used by the XRD encoder to compress variable-length inputs into a fixed number of latent vectors.

::: _models.perceiver.PerceiverResampler

## Loading and building

Model selection is registry-based. `activate_conditionality` selects the requested model family from `MODEL_REGISTRY`. An unknown family raises an error rather than falling back to a different model.

::: _utils.model.load_pretrained_model
::: _utils.model.build_model