# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [v1.2.0] - 2026-03-13

### Features and Enhancements

- **Virtual Crystal Generator**: Added `_utils/_virtualiser/` subpackage implementing the `crystal_virtualiser` tool (developed by Dr Ricardo Grau-Crespo). Post-generation utility that converts ordered CIF structures from the model into disordered virtual crystals with promoted symmetry. Element pairs are replaced with fractional occupancies matching the global composition ratio, and the structure is refined to its higher-symmetry parent using spglib via pymatgen.
- **New API Endpoint**: `POST /virtualise` exposes the virtualiser via the REST API, accepting either inline `virtual_pairs` or a YAML config file path.

### Repo Structure and Testing

- New local test section `VirtualiserTests` covering pair fraction calculation, structure virtualisation, symmetry promotion, config loading, and full CIF output.
- New API test section `VirtualiserEndpointTests` with smoke tests for inline pairs, config file mode, and validation errors.

### Documentation

- New **Virtual Crystal Generation** section under Quick Start in the README with CLI usage and an end-to-end example.
- Two new API quickstart examples for the `/virtualise` endpoint (inline pairs and config file modes).

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
