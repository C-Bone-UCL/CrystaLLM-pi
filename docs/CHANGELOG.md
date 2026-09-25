# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [v2.0.0] - YYYY-MM-DD

### Models

* **Prefix and Residual replace PKV and Slider for training**: `Prefix` succeeds `PKV` and `Residual` succeeds `Slider`. Training a PKV or Slider model now raises an error naming its successor. Released PKV and Slider checkpoints still generate identically. The T1 and T5 tutorials now fine-tune `Prefix`.
* **New text base model on the Hub**: `CrystaLLM-pi_ft_alex_mp_20-text` is `CrystaLLM-pi_base` fine-tuned on `alex_mp_20` with no conditioning, and is the recommended base for new fine-tuning runs. It trains from `_config_files/training/unconditional/ft-alex-mp-20-text.jsonc`.
* **Training configs on the Hub**: every released model now carries the `training_config.jsonc` that produced it and the resolved `training_args.json`, the same files new training runs write into their checkpoints.
* **Breaking: `LOGP` now measures the model rather than the sampler**: perplexity comes from a separate forward pass over the full vocabulary, not from the generation-time scores, which `top_k`, `top_p` and `temperature` have already truncated and sharpened. Absolute values shift by under 1%, but the top-ranked candidate changes for roughly a third of prompts, so `LOGP` rankings from earlier versions are not directly comparable. `scoring_methods.forward_pass_logp` replaces `score_outputs_logp` and `score_output_logp`. Generation no longer retains per-step logits, which lowers peak memory.

### Generation Screening

* **`--screening_profile`**: sets how hard generated CIFs are screened. `application`, the default here, runs the bond-length check and ranks the whole batch that reaches the target rather than the first candidates to arrive. `benchmark` reproduces the screening behind the published MP-20 and CHILI-100K numbers, and is the default in the paper reproduction repository.
* **Disordered structures pass through the bond-length check**: it now reports "not checked" instead of failing on partial occupancies, so virtualiser output can be screened rather than skipped.
* **Formula consistency handles supercells and partial occupancy**: the declared formula and the atom-site composition are compared up to cell scale, at a tolerance that still rejects a CIF declaring `Fe12C4` whose sites hold `Fe2C`.
* **Unknown config keys are rejected**: a misspelled or stale key in a `.jsonc` config raises instead of doing nothing.

### Post-processing

* **Virtual crystals from more than two elements**: `virtual_pairs` entries may now list any number of elements to merge onto one sublattice, so an ordered ternary or quaternary alloy cell collapses into a single mixed-occupancy solid solution. Two-element entries behave as before.

### Repository Split

* **Paper content moved out**: the paper notebooks and the Prepend/Raw baseline families now live in the standalone reproduction repository [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper), along with all paper configurations. This repository keeps the Prefix, Residual, PKV, Slider and unconditional models, with new tutorial notebooks covering fine-tuning, loading and generation, the API, and SLME.
* **ALIGNN removed**: the separate `alignn_env` environment used for bandgap predictions in some paper studies has been removed, leaving a single repository environment. Bandgap generation remains supported, as does evaluation of the density property when required.
* **Breaking: module renames**: scripts and modules under `_utils/` drop their leading underscore and use lowercase names (`_cleaning.py` is now `cleaning.py`, `generate_CIFs.py` is now `generate_cifs.py`). `_utils/_metrics/` is now `_utils/_scoring/`, `_utils/_api_utils/` is now `_utils/_api/`, `_utils/_tokenizer_utils/` is now `_utils/_tokenizer/`, and the `_*_utils.py` modules lose the suffix (`_processing_utils.py` is now `processing.py`). The tokenizer moved from `HF-cif-tokenizer/` to `_utils/HF-cif-tokenizer/`.

### Documentation

* **Docstrings standardised**: every module and every public function exposed through the documentation follows one format, and registered tests enforce it.
* **Docs site**: the full documentation is now at [c-bone-ucl.github.io/CrystaLLM-pi](https://c-bone-ucl.github.io/CrystaLLM-pi/), built with MkDocs Material, with an API reference generated from the docstrings.
* **AI usage disclosure**: `docs/AI_USAGE_DISCLOSURE.md` records how generative AI was used to build and maintain this codebase.

### Packaging

* **pip install**: the package now installs with `pip install -e ".[all]"`. The `train`, `api`, `metrics`, `notebooks` and `docs` extras install only what each part needs.

### Licensing

* **`LICENSE` now carries a copyright notice**: the MIT permission notice referred to "the above copyright notice" when the file contained none. It also now retains the notice for [CrystaLLM](https://github.com/lantunes/CrystaLLM) (Copyright (c) 2023 Luis M. Antunes), from which parts of this codebase are derived.

## [v1.3.2] - 2026-07-20

## New functionalities
- **Save Training Args**: Now, in the model checkpoint folder, we keep a copy of the training args given to the training script.
- New `_save_model_to_HF.py`: uploads CrystaLLM-$\pi$ models to the Hugging Face Hub. In `_utils/_preprocessing/`
- Edits to `_load_and_generate.py`: Users can now register a custom model using the `--model_registry` argument which points to a `json` registry. The script can then be used as normal. New test in suite added for this and API updated  to integrate it.

## [v1.3.1] - 2026-05-13

### Reproducibility and Branching

- **Maintained Main Branch**: After tagging `v1.3.0` for paper reproduction, `main` restores the maintained generation and validation behavior. Relative to the paper snapshot, `main` re-enables bond-length validity checks during generation-time validation, scores each generated batch before truncating to `target_valid_cifs`, and uses the stricter structure-aware `is_formula_consistent` check in the metrics utilities.

### CI's and Testing
- **Github Actions**: Now we can run test suites via github actions (See `Contributing.md` for details). 
- **CPU API Tests Compatibility**: Now we have a CPU only possible toggle for the full API suite in case environment has no GPU tests will run accordingly on CPU.

## [v1.3.0] - 2026-05-13

### Features and Enhancements

- **Challenge Benchmark Removal**: Removed the Challenge benchmark workflow from the maintained codebase.
- **COD Workflow Retirement**: Removed the deprecated COD workflow, including the utilities for converting disordered structures into ordered representations.
- **MatterGen Comparison Notebook**: Added a MatterGen comparison notebook with P1 space-group handling.
- **New Chili Dataset Support**: Added new Chili dataset training and benchmark workflows.
- **Polymorph Analysis Refresh**: Updated the polymorph analysis workflow with corrected processing.
- **Logit Analysis**: New notebook on how to analyse the model's logits during generation of a crystal for mechanistic understanding `Logits.ipynb`.
- **Loss Landscape Analysis**: New notebook on to show plots of loss landscapes as used in the paper appendices.
- **Dataset Cleaning Improvements**: The cleaning script can now add token counts and support the downstream analyses used in `Dataset_stats.ipynb`.
- **Model Availability Updates**: Updated load-and-generate script support to include `mp-20`, `alex-mp-20`, and `chili100k` models and removed the retired COD one.
- **Notebook Plot Utilities**: Released plotting utilities through the notebook utils package with updated plots for v2 of paper.

### For API/direct generation
- **Load-and-Generate Prompt Mapping**: Updated `_load_and_generate.py` so prompt inputs are provided as aligned per-prompt lists rather than implicitly expanding one reduced formula across multiple Z values. Generation inputs should now be passed with a 1:1 mapping across the prompt-defining arguments, example in the updated `X_XRD_TiO2.ipynb` notebook.

### Repo Structure and Testing

- **Notebook Utils Reorganization**: Split `_utils/_notebook_utils.py` into the `_utils/_notebook_utils/` package, with notebook-specific modules and shared utilities.
- **Code Cleanup**: Removed dead code left behind by retired workflows.
- **Maintained Workflow**: Updated tests, API parity, API documentation, and coverage for the maintained generation, preprocessing, and metrics workflows.
- **Project Metadata**: Added `CODE_OF_CONDUCT.md` and `CONTRIBUTING.md`, refreshed GitHub workflows, and introduced a new `pyproject.toml`.

### Reproducibility and Branching

- **Repository Split**: the `main` branch at `v1.3.0` and `paper-v2` branch preserve the paper-reproduction workflow for the second pre-print iteration. That snapshot intentionally excludes the maintained-branch generation and validation updates later restored on `main`.


## [v1.2.0] - 2026-03-16

### Features and Enhancements

- **Virtual Crystal Generator**: For disordered material (partial occupancy) generation support. Added `_utils/_virtualiser/` subpackage implementing the `crystal_virtualiser` tool (developed by [Dr Ricardo Grau-Crespo](https://github.com/rgraucrespo)). Post-generation utility that converts ordered CIF structures from the model into disordered virtual crystals with promoted symmetry. Element pairs are replaced with fractional occupancies matching the global composition ratio, and the structure is refined to its higher-symmetry parent using spglib via pymatgen. Included passing tests, API endpoints, README update with examples.

### Efficiency improvements & Dependency Changes
- **In Generation Script**: Fixed redundant transition scoring calls in generate with perplexity ranking, improving generation speed without affecting ranked outputs when using the maintained branch behavior.
- **W&B Update**: To v0.25.0


## [v1.1.0] - 2026-03-02

### Features and Enhancements

- **New Conditional Model Integration**: Added support for the Mattergen-XRD model. Updated API test suites for endpoint compatibility.
- **Reduced Formula Search (`_load_and_generate.py`)**: New `--search_zs` flag sweeps Z=1 to 4 automatically. With perplexity ranking it evaluates all Z values and returns the lowest-perplexity outputs. Without ranking it exits early on the first valid CIF. Z can also be set directly if known. Amount of CIFs returned per prompt is controlled by `--target_valid_cifs`.
- **Improved Perplexity Ranking**: Batch processing now scores the full batch before slicing to `target_valid_cifs`, so the best structures are returned rather than the first valid ones.
- **Multi-GPU Generation**: Added multi-GPU support to `_load_and_generate.py`.
- **Automated XRD Preprocessing**: Converts user XRD patterns from any primary radiation wavelength to the model's expected CuKα automatically. Trims to the 20 most intense peaks, filters 2theta to 0-90 and intensities to 0-100, and sorts from most to least intense. Users only need to provide peak-picked data.

### Bug Fixes

- **CIF Structural Validation**: Formula checks now also use composition derived directly from the 3D structure object and compared against CIF tags, correctly handling fractional occupancies and rejecting invalid 0.0 or non-integer site occupancies.

### Repo Structure and Testing

- **Docker Changes**: Consolidated configs into `docker/`, added code-reload development mode, standardised setup commands via Makefile.
- **Apptainer Support**: Added targets for HPC / no-Docker environments. All tests pass in both Apptainer and Docker images.
- **Refactored API Endpoints and Test Suites**: Test suites and API route handlers are now split into logical directories per functionality rather than one large file.

### New Notebook

- New example notebook `notebooks/Z_API_density.ipynb` showing an end-to-end API use case: predicting density (via structure prediction) for given compositions and optionally associated XRDs, example use case on a subset of MP-20 as well as manual prompt creation.

## [v1.0.0] - 2025-11-01

- Initial release to reproduce the paper ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299).
