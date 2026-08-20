# CLI entry points

The package provides command-line tools as Python modules rather than installed console scripts. Run them from a repository clone.

Each CLI module documents a representative `Usage:` command. Use `argparse --help` for the complete set of available options.

## Load and generate

Load a pretrained model from the Hub or a local checkpoint and generate CIFs from explicit formulas or a parquet file of prompts.

::: _load_and_generate

## Training

Training is configuration-driven. A JSONC file provides the base configuration and command-line flags override individual values.

::: _train

## Dataset preparation

Prepare datasets through the preprocessing pipeline.

::: _utils._preprocessing.cifs_zip_to_parquet
::: _utils._preprocessing.deduplicate
::: _utils._preprocessing.cleaning
::: _utils._preprocessing.calculate_theor_xrd
::: _utils._preprocessing.process_exp_xrd_continuous

Experimental XRD conversion can be checked visually before the resulting profiles are used for conditioning.

::: _utils._preprocessing.process_exp_xrd_continuous.save_pipeline_plot

::: _utils._preprocessing.save_dataset_to_hf
::: _utils._preprocessing.save_model_to_hf
::: _utils._preprocessing.save_tokenizer_to_hf

## Generation and scoring

The generation CLIs are documented on the [Generation](generation.md) page alongside the functions that define each stage of the pipeline.

- [`make_prompts`](generation.md#_utils._generating.make_prompts)
- [`generate_cifs`](generation.md#_utils._generating.generate_cifs)
- [`postprocess`](generation.md#_utils._generating.postprocess)

The evaluation CLIs are documented on the [Metrics and analysis](metrics.md) page alongside the quantities they report.

- [`evaluate_cifs`](metrics.md#_utils._generating.evaluate_cifs)
- [`vun_metrics`](metrics.md#_utils._scoring.vun_metrics)
- [`xrd_metrics`](metrics.md#_utils._scoring.xrd_metrics)
- [`property_metrics`](metrics.md#_utils._scoring.property_metrics)
- [`mace_ehull`](metrics.md#_utils._scoring.mace_ehull)
- [`dft_ehull`](metrics.md#_utils._scoring.dft_ehull)
