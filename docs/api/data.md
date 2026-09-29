# Data

This section covers the pipeline from parquet files containing CIF text to the batches consumed during training. It includes the tokenizer that converts CIF strings to token IDs, the dataset loader that tokenizes and filters the data, and the collator that assembles fixed-length training batches.

The preprocessing CLIs used to generate the parquet files are documented on the [CLI entry points](cli.md) page. The models that consume these batches are documented on the [Models](models.md) page.

## Tokenizer

The CIF tokenizer provides the common text interface used by all model families.

::: _tokenizer.CustomCIFTokenizer

## Dataset loading

`load_data` tokenizes a Hugging Face dataset and returns the tokenized dataset together with the collator required by the Hugging Face `Trainer`. For conditional runs, `condition_columns` specifies the dataset columns containing the conditioning values. These values are preserved through tokenization and passed through to the training batch.

Filtering is optional. By default, CIFs longer than the context length are truncated (but i typically prefer to do this in preprocessing), and CIFs containing unknown tokens are retained. Set `remove_CIFs_above_context` or `remove_CIFs_with_unk` to discard these samples instead.

::: _dataloader.load_data

## Custom Dataloader for Conditioning

`CustomCIFDataCollator` constructs the batches consumed during training. Changes to the training pipeline may require changes here.

Every sequence produced by the collator has exactly `context_length` tokens. If a CIF exceeds the context window, it is truncated from the beginning. Shorter CIFs are then packed with tokens from other CIFs, selected round-robin from the remainder of the batch, until the window is filled. This packing keeps the context window occupied by CIF tokens rather than padding, but a single sequence can therefore contain parts of multiple structures.

The collator determines whether a batch is conditional from the features themselves by checking for `condition_values`. The same collator therefore handles both conditional and unconditional training without an additional mode flag. For a packed sequence, the associated condition values come from the CIF that provided the start of the sequence rather than from every CIF contributing tokens to it.

Each batch is returned as a dictionary containing `[B, context_length]` tensors for `input_ids`, `labels`, `fixed_mask`, `attention_mask`, and `special_tokens_mask`. Padding positions in `labels` are set to `-100`. Conditional batches also contain `condition_values`.

Scalar conditions are stacked into tensors of shape `[B, n_conditions]`. Continuous XRD conditions can contain a different number of rows for each sample, so they are front-padded with `-100.0` to match the longest sample in the batch before stacking.

::: _dataloader.CustomCIFDataCollator