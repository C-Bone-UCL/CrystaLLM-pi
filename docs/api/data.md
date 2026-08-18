# Data

Everything between a parquet of CIF text and a batch the model can train on.

## Tokenizer

::: _tokenizer.CustomCIFTokenizer

## Dataset and collation

::: _dataloader.load_data
::: _dataloader.CustomCIFDataCollator

## Property normalization

Conditioning values are normalized before they reach the model. The method is recorded in the
checkpoint so generation can invert it.

::: _utils.processing.normalize_property_column
::: _utils.processing.normalize_values_with_method
::: _utils.processing.filter_df_to_context
::: _utils.processing.count_tokens_df

## CIF text helpers

::: _utils.processing.add_variable_brackets_to_cif
::: _utils.processing.remove_atom_props_block
::: _utils.processing.remove_comments
