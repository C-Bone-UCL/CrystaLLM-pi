# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v1.3.0] - 2026-05-13

### Features and Enhancements

- **Challenge Benchmark Removal**: Removed the Challenge benchmark workflow from the maintained codebase.
- **COD Workflow Retirement**: Removed the deprecated COD workflow, including the utilities for converting disordered structures into ordered representations.
- **MatterGen Comparison Notebook**: Added a MatterGen comparison notebook with P1 space-group handling.
- **New Chili Dataset Support**: Added new Chili dataset training and benchmark workflows.
- **Polymorph Analysis Refresh**: Updated the polymorph analysis workflow with corrected processing.
- **Logit Analysis**: New notebook on how to analyse the model's logits during generation of a crystal for mechanistic understanding `Logits.ipynb`.
- **Loss Landscape Analysis**: New notebook on to show plots of loss ladnscapes as used in the paper appendices.
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

- **Repository Split**: `paper-v2` branch preserves the paper-reproduction workflow and therefore excludes the post-paper generation and validation updates previously introduced on `main`: redundant transition-score removal during perplexity ranking, extra validity checks, full-batch perplexity scoring before slicing to `target_valid_cifs`, and stricter formula-consistency handling. `main` remains the maintained branch and includes those updates.

---

## [v1.2.0] - 2026-03-16

### Features and Enhancements

- **Virtual Crystal Generator**: For disordered material (partial occupancy) generation support. Added `_utils/_virtualiser/` subpackage implementing the `crystal_virtualiser` tool (developed by [Dr Ricardo Grau-Crespo](https://github.com/rgraucrespo)). Post-generation utility that converts ordered CIF structures from the model into disordered virtual crystals with promoted symmetry. Element pairs are replaced with fractional occupancies matching the global composition ratio, and the structure is refined to its higher-symmetry parent using spglib via pymatgen. Included passing tests, API endpoints, README update with examples.

### Efficiency improvements & Dependency Changes
- **In Generation Script**: Fixed redundant transition scoring calls in generate with perplexity ranking, improves generation speed without affecting any outputs or score outputs (backwards compatible)
- **W&B Update**: To v0.25.0

---

## [v1.1.0] - 2026-03-02

### Features and Enhancements

- **New Conditional Model Integration**: Added support for the Mattergen-XRD model. Updated API test suites for endpoint compatibility.
- **Reduced Formula Search (`_load_and_generate.py`)**: New `--search_zs` flag sweeps Z=1–4 automatically. With perplexity ranking it evaluates all Z values and returns the lowest-perplexity outputs.  Without ranking it exits early on the first valid CIF. Z can also be set directly if known. Amount of CIFs returned per prompt is controlled by `--target_valid_cifs`.
- **Improved Perplexity Ranking**: Batch processing now scores the full batch before slicing to `target_valid_cifs`, ensuring the best structures are returned rather than just the first valid ones.
- **Multi-GPU Generation**: Added multi-GPU support to `_load_and_generate.py`.
- **Automated XRD Preprocessing**: Converts user XRD patterns from any primary radiation wavelength to the model's expected CuKα automatically. Trims to the 20 most intense peaks, filters 2theta to 0–90 and intensities to 0–100, and sorts from most to least intense. Users only need to provide peak-picked data.

### Bug Fixes

- **CIF Structural Validation**: Formula hecks now also use composition derived directly from the 3D structure object and compared against CIF tags, correctly handling fractional occupancies and rejecting invalid 0.0 or not integer site occupancies.

### Repo Structure and Testing

- **Docker Changes**: Consolidated configs into `docker/`, added code-reload development mode, standardised setup commands via Makefile.
- **Apptainer Support**: Added targets for HPC / no-Docker environments. All tests pass in both Apptainer and Docker images.
- **Refactored API Endpoints and Test Suites**: Test suites and API route handlers are now split into logical directories per functionality rather than one large file.

### New Notebook

- New example notebook `notebooks/Z_API_density.ipynb` showing an end-to-end API use case: predicting density (via structure prediction) for given compositions and optionally associated XRDs, example use case on a subset of MP-20 as well as manual prompt creation.
---

## [v1.0.0] - 2025-11-01

- Initial release to reproduce the paper ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299).