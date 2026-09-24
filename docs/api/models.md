# Models

CrystaLLM-pi supports two conditioning mechanisms for crystal structure generation: prefix conditioning and residual conditioning. The tokenizer and datasets used to train these models are documented on the [Data](data.md) page.

Each conditional model family includes a configuration, an encoder that maps the conditioning inputs to the representation expected by the transformer, and a GPT-2 model that applies the conditioning.

Tensor shapes and `forward()` behaviour are documented in the source docstrings and are available through the linked source or `help()`.

The earlier `PKV` and `Slider` classes used in the paper are retained for loading released checkpoints. They implement earlier versions of the same conditioning mechanisms and are not part of the current model API, so they are not included below.

## Prefix

Prefix conditioning represents scalar properties as prefix key-values for GPT-2.

::: _models.Prefix_model.PrefixGPT2Config
::: _models.Prefix_model.PrefixEncoder
::: _models.Prefix_model.PrefixGPT

## Residual

Residual conditioning adds the conditioning signal to the attention blocks as a residual contribution.

::: _models.Residual_model.ResidualGPT2Config
::: _models.Residual_model.ResidualEncoder
::: _models.Residual_model.ResidualAttention
::: _models.Residual_model.ResidualGPT2Block
::: _models.Residual_model.ResidualGPT2Model
::: _models.Residual_model.ResidualGPT

## Loading and building

Model selection is registry-based. `activate_conditionality` selects the requested model family from `MODEL_REGISTRY`. An unknown family raises an error rather than falling back to a different model.

::: _utils.model.load_pretrained_model
::: _utils.model.build_model