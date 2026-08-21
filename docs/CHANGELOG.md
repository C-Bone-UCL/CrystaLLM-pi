# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [Unreleased]

### New Model Generation

* **New conditional families**: `Prefix`, `PrefixXRD`, and `Residual` have been imported from CrystaLLM-graph, with minor upgrades. They replace `PKV` and `Slider` respectively. New dependency: `einops`.
* **PKV and Slider are now generation-only**: training either model now raises an error directing users to its successor. All released legacy checkpoints continue to generate identically. T1 and T5 now fine-tune `Prefix`, and `T4_XRD_continuous` is now available for continuous-XRD training. XRD training itself remains in CrystaLLM-graph.
* **Three new models on the Hub**: `Chili100K-cXRD` and `alex_mp_20-cXRD` are continuous-XRD models and KD students of the graph teacher. `ft_alex_mp_20-text` is a text-only `alex_mp_20` model and the recommended base for new fine-tuning runs.
* **Generate from raw XRD scans**: `--xrd_files` now accepts raw diffractometer scans for cXRD models, with no peak picking required. The new `process_exp_xrd_continuous.py` converts scans from 2theta to Q, removes the background, resamples the profile, and can plot each stage for inspection. New dependency: `pybaselines`.
* **XRD-fit ranked Z search**: the new `PEARSON` scoring mode ranks Z-search candidates by fitting each generated candidate's simulated diffraction pattern to the input scan on a 1000-point grid. Initial internal tests suggest that `LOGP` improves RMSD, while `PEARSON` improves match rate.

### Repository Split

* **Paper content moved out**: the paper notebooks and the Prepend/Raw baseline families now live in the standalone reproduction repository [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper). This repository retains the PKV, Slider, and unconditional models, together with new tutorial notebooks covering the fine-tuning pipeline, loading and generation, the API, and SLME. All paper configurations now live in the reproduction repository.
* **ALIGNN removed**: the separate `alignn_env` environment used for bandgap predictions in some paper studies has been removed, leaving a single repository environment. Bandgap generation remains supported, as does evaluation of the density property when required.
* **Breaking: file and folder renames**: standardized names for JOSS.

### Documentation

* **Docstrings standardised**: every module and every public function exposed through the documentation now follows a standardised format, with registered tests enforcing the format.
* **Docs site**: the full documentation is now available at [c-bone-ucl.github.io/CrystaLLM-pi](https://c-bone-ucl.github.io/CrystaLLM-pi/), built with MkDocs Material and an API reference generated from the docstrings.

### Packaging

* **New installation method**: the package can now be installed with `pip install -e ".[all]"`, with optional extras for just the training, the API, metrics, notebooks, and documentation deps. 

### Licensing

* **`LICENSE` now carries a copyright notice**: the MIT permission notice referred to "the above copyright notice" when the file contained none. It also now retains the notice for [CrystaLLM](https://github.com/lantunes/CrystaLLM) (Copyright (c) 2023 Luis M. Antunes), from which parts of this codebase are derived.

### JOSS
* **New documents**: Released necessary documents and files for JOSS submission
* **JOSS paper**: Also added the paper.

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
