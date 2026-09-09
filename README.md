<div align="center">

<h1> CrystaLLM-<span style="font-size: 1.2em;">&pi; </span> (property injection) </h1>
  <img src="docs/images/Logo.png" alt="CrystaLLM-pi logo" width="150" />
  <p>
    <strong>A Transformer-based model for property-guided crystal structure generation
    </strong>
  </p>
</div>

<p align="center">
<a href="https://huggingface.co/c-bone">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue.svg?style=plastic">
</a>
<a href="https://huggingface.co/spaces/LeMaterial/LeMat-GenBench">
    <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20LeMat%20Bench-Unconditional%20Benchmark-lightblue.svg?style=plastic">
</a>
<a href="https://arxiv.org/pdf/2511.21299">
    <img alt="Preprint" src="https://img.shields.io/badge/Preprint-arXiv-red.svg?style=plastic">
</a>
<a href="https://www.nature.com/articles/s41467-024-54639-7">
    <img alt="Based on CrystaLLM Paper" src="https://img.shields.io/badge/Based%20on-CrystaLLM%20Paper-orange.svg?style=plastic">
</a>
<a href="https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=plastic">
</a>
<a href="https://github.com/C-Bone-UCL/CrystaLLM-pi/actions/workflows/ci.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/C-Bone-UCL/CrystaLLM-pi/ci.yml?branch=main&label=CI&style=plastic">
</a>
<a href="https://c-bone-ucl.github.io/CrystaLLM-pi/">
    <img alt="Documentation" src="https://img.shields.io/badge/Docs-GitHub%20Pages-brightgreen.svg?style=plastic">
</a>

</p>

<br>

## Overview

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> is a Transformer-based system for generating crystalline structures as CIF files. It supports both unconditional generation and several conditional architectures that can generate structures based on target properties like bandgap, density, photovoltaic efficiency and XRD patterns, including raw experimental powder scans.

<div align="center">
<img src="docs/images/Framework_github.png" width="75%" style="background-color:white;"/>
</div>

## Statement of need

CrystaLLM-<span style="font-size: 1.2em;">π</span> is a framework for conditional crystal structure generation. You can fine-tune a pretrained model on numerical properties and generate structures aimed at a target value, or generate directly from one of the open-sourced models.

The framework supports numerical conditioning variants without requiring a separate generation framework for each application. We have demonstrated the approach for materials discovery with target functional properties and for recovering crystal structures from experimental characterisation data.

The repository provides the tools needed to apply the method to new problems, including an installable codebase, tutorials, notebooks, documentation, pretrained models on the Hugging Face Hub, a containerised API for model serving, and a web application for interactive generation.

Modern transformers are memory efficient enough that most training and inference runs on a GPU, or on CPU. Most models fit on a 16GB card for training and need 1-2GB for light generation.

The framework supports experiments with conditional generative models.

## Reproducing the paper
The studies from the ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299) paper live in the standalone repo [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper) (or [v1.3.0 tag](https://github.com/C-Bone-UCL/CrystaLLM-pi/releases/tag/v1.3.0) of this repository). The next graph-conditioned knowledge distillation paper reproduction code can be accessed in [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph). Otherwise, all models so far are accessible to generate with here.

This repository stays the maintained package and is still under active development.

## Key Features

- **Unconditional Generation**: Generate crystal structures from structural/composition priors
- **Property-Guided Generation**: Generate crystal structures conditioned on target properties + structural priors
- **Multiple Architectures**: Two conditioning mechanisms, prefix and residual, plus the unconditional base model. Prefix conditioning also takes full XRD patterns through a Perceiver resampler.
- **Flexible Conditioning**: You can use any set of numerical properties to condition and one of the models handles heterogeneous datasets (some properties are missing in the dataset but not others...)
- **Evaluation of output structures**: Scripts for validity, uniqueness, novelty and stability metrics
- **HuggingFace Integration**: Pre-trained models available on HF Hub

## Documentation

Full documentation: **https://c-bone-ucl.github.io/CrystaLLM-pi/**

- [Installation](https://c-bone-ucl.github.io/CrystaLLM-pi/install/)
- [Quickstart and generation examples](https://c-bone-ucl.github.io/CrystaLLM-pi/quickstart/)
- [Model types and conditioning mechanisms](https://c-bone-ucl.github.io/CrystaLLM-pi/models/)
- [Training, generating and evaluating from scratch](https://c-bone-ucl.github.io/CrystaLLM-pi/training/)
- [Virtual crystal generation](https://c-bone-ucl.github.io/CrystaLLM-pi/virtualiser/)
- [API service and Apptainer builds](https://c-bone-ucl.github.io/CrystaLLM-pi/api-service/)
- [Tutorial notebooks and tokenizer customisation](https://c-bone-ucl.github.io/CrystaLLM-pi/tutorials/)
- [API reference](https://c-bone-ucl.github.io/CrystaLLM-pi/api/cli/)
- [Contributing](docs/CONTRIBUTING.md) and [Code of conduct](docs/CODE_OF_CONDUCT.md)

## Installation

```bash
git clone https://github.com/C-Bone-UCL/CrystaLLM-pi.git
cd CrystaLLM-pi
conda create -n CrystaLLM-pi_env python=3.10
conda activate CrystaLLM-pi_env
pip install -e ".[all]"
```

Lighter installs (generation only, training only, API only), prerequisites and API key configuration are on the [installation page](https://c-bone-ucl.github.io/CrystaLLM-pi/install/).

## Quick Start

Use with pre-trained models from HuggingFace Hub for direct crystal structure generation. The `_load_and_generate.py` script handles downloading models and generating valid CIF structures with desired properties.

> **Note**: Properties (Conditions, Spacegroups, XRD files, Z values) map strictly 1:1 to the canonicalized reduced formulas provided in `--reduced_formula_list`.
> 
> **Outputs**: Outputs can either be saved as a dataframe in a `.parquet` using the `--output_parquet` flag, or as individual CIFs in a directory using the `--output_cif_dir` flag.

**Explicit Z Generation (Unconditional)**

Generate 10 (2 batches of 5) Ti2O4 structures by explicitly setting the reduced formula and Z=2, including a spacegroup constraint.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_ft_alex_mp_20-text" \
    --reduced_formula_list "TiO2" \
    --z_list "2" \
    --spacegroups "P4_2/mnm" \
    --level level_4 \
    --num_return_sequences 5 \
    --max_return_attempts 2 \
    --output_parquet generated_structures.parquet
```

**Raw Experimental Scan Conditioning (Continuous XRD)**

Recover a structure from a raw powder pattern, no peak picking needed. The scan is converted to the model's continuous `[Q, I]` profile automatically.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-cXRD_chili100k" \
    --reduced_formula_list "TiO2" \
    --xrd_files tests/fixtures/Rutile-TiO2-unproc.txt \
    --xrd_wavelength 1.54059 \
    --level level_3 \
    --target_valid_cifs 1 \
    --output_cif_dir outputs/cxrd_demo
```

Mapped condition lists, Z-searches, perplexity and XRD-fit ranking and every configuration option are on the [quickstart page](https://c-bone-ucl.github.io/CrystaLLM-pi/quickstart/).

## Available Pre-trained Models

Each released model exists because a paper study or tutorial produced it. The table says which, so you know where to look for its training setup and evaluation.

| Model | Class | Conditioning | Origin |
|---|---|---|---|
| `c-bone/CrystaLLM-pi_ft_alex_mp_20-text` | GPT-2 | unconditional | **Recommended base model**, LeMat-Bulk pretrain finetuned on Alex-MP-20 CIFs, from [CrystaLLM-cXRD](https://github.com/C-Bone-UCL/CrystaLLM-cXRD) |
| `c-bone/CrystaLLM-cXRD_chili100k` | PrefixXRD | continuous XRD profiles | CHILI-100K model from [CrystaLLM-cXRD](https://github.com/C-Bone-UCL/CrystaLLM-cXRD), experimental-structure priors, the choice for measured scans |
| `c-bone/CrystaLLM-cXRD_alex-mp-20` | PrefixXRD | continuous XRD profiles | Alex-MP-20 bridge model from [CrystaLLM-cXRD](https://github.com/C-Bone-UCL/CrystaLLM-cXRD), broader coverage than the CHILI model |
| `c-bone/CrystaLLM-cXRD_mp20` | PrefixXRD | continuous XRD profiles | MP-20 benchmark model from [CrystaLLM-cXRD](https://github.com/C-Bone-UCL/CrystaLLM-cXRD), trained from scratch |
| `c-bone/CrystaLLM-pi_base` | GPT-2 | unconditional | LeMaterial base model from the first paper |
| `c-bone/CrystaLLM-pi_mp_20_base` | GPT-2 | unconditional | mp-20 pretraining base from the paper's pretraining studies |
| `c-bone/CrystaLLM-pi_alex_mp_20_base` | GPT-2 | unconditional | alex-mp-20 pretraining base from the paper's dataset-size study |
| `c-bone/CrystaLLM-pi_SLME` | Prefix (legacy `PKV`) | solar efficiency (SLME), 0-33% | SLME discovery study, maintained here in [`T5_SLME`](notebooks/T5_SLME.ipynb) |
| `c-bone/CrystaLLM-pi_bandgap` | Prefix (legacy `PKV`) | bandgap + stability, 0-18 eV / 0-5 eV/atom | Pretraining-benefits study ([B1a notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B1a_Pretrain_benefits.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_density` | Prefix (legacy `PKV`) | density + stability, 0-25 g/cm3 / 0-0.1 eV/atom | Dataset-size study ([B2 notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B2_Dataset_size_study.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_Mattergen-XRD` | Residual (legacy `Slider`, top-20 peaks) | XRD peaks (theoretical patterns, fully ordered bias) | XRD recovery studies ([X_XRD notebooks](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/tree/main/notebooks) in the paper repo) |

Model metadata (class, conditions, normalization) lives in [`_utils/model_registry.json`](_utils/model_registry.json). To generate with a model that is not in the table, pass a JSON file with the same schema via `--model_registry`. [`notebooks/T1_finetune_density_example.ipynb`](notebooks/T1_finetune_density_example.ipynb) walks through that full loop (finetune a density model, upload it, register it, generate with it).

The conditioning mechanism behind each class is described on the [models page](https://c-bone-ucl.github.io/CrystaLLM-pi/models/).

## Tutorial Notebooks

Five notebooks in [`notebooks/`](notebooks/) cover the maintained workflows end to end:

* [`T1_finetune_density_example.ipynb`](notebooks/T1_finetune_density_example.ipynb): finetune a base model on your own property dataset, push it to the Hub, register it, and generate with it
* [`T2_load_and_generate.ipynb`](notebooks/T2_load_and_generate.ipynb): generate structures with the released Hub models (courtesy of [Joley Lin](https://github.com/yhjollin/))
* [`T3_API_density_example.ipynb`](notebooks/T3_API_density_example.ipynb): predict density for a composition through the containerised API
* [`T4_XRD_continuous.ipynb`](notebooks/T4_XRD_continuous.ipynb): recover a structure from a raw experimental XRD scan with the continuous-XRD model
* [`T5_SLME.ipynb`](notebooks/T5_SLME.ipynb): discover a material with a target photovoltaic efficiency

The paper studies are not here, they live in [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper) (see [Reproducing the paper](#reproducing-the-paper)).

Customising the tokenizer is covered on the [tutorials page](https://c-bone-ucl.github.io/CrystaLLM-pi/tutorials/).

## Citation

Please cite the following when using this work: ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299)

```
@misc{bone2026discoveryrecoverycrystallinematerials,
      title={Discovery and recovery of crystalline materials with property-conditioned transformers}, 
      author={Cyprien Bone and Matthew Walker and Bradley A. A. Martin and Kuangdai Leng and Luis M. Antunes and Ricardo Grau-Crespo and Amil Aligayev and Javier Dominguez and Keith T. Butler},
      year={2026},
      eprint={2511.21299},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2511.21299}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact cyprien.bone.24@ucl.ac.uk or raise an issue on the GitHub page.

## Acknowledgments
This work has been supported by UKRI funding (EP/Y000552/1 and EP/Y014405/1)
