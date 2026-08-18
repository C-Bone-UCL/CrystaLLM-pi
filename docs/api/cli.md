# CLI entry points

The package ships no console scripts, so each command-line tool is its module, run from a clone.
Every module docstring below carries a runnable `Usage:` command. `argparse --help` on the same
module is the authority on flags.

## Load and generate

Loads a pretrained model from the Hub or a local checkpoint and generates CIFs against explicit
formulas or a parquet of prompts.

::: _load_and_generate

::: _load_and_generate.main
::: _load_and_generate.run_formula_mode
::: _load_and_generate.run_parquet_mode
::: _load_and_generate.generate_prompts_from_specs
::: _load_and_generate.generate_cifs_with_hf_model
::: _load_and_generate.write_outputs

## Training

::: _train.main
::: _args.parse_args
::: _args.str_to_bool

## Dataset preparation

Pipeline stages, in the order a dataset moves through them.

::: _utils._preprocessing.cifs_zip_to_parquet
::: _utils._preprocessing.deduplicate
::: _utils._preprocessing.cleaning
::: _utils._preprocessing.calculate_theor_xrd
::: _utils._preprocessing.process_exp_xrd_inputs
::: _utils._preprocessing.process_exp_xrd_continuous
::: _utils._preprocessing.save_dataset_to_hf
::: _utils._preprocessing.save_model_to_hf
::: _utils._preprocessing.save_tokenizer_to_hf

## Tokenizer

::: _utils._tokenizer.create_vocab
::: _utils._tokenizer.checks
