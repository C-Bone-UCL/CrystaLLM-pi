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

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> generates crystal structures as CIF files, with optional targets such as bandgap, density, photovoltaic efficiency or XRD peaks. You can use a released model or fine-tune one on your own data.

<div align="center">
<img src="docs/images/Framework_github.png" width="75%" style="background-color:white;"/>
</div>

The same training and generation workflow supports different numerical properties. To work with a new property, prepare a dataset with the corresponding values and fine-tune a model.

This repo includes the package, tutorials, documentation and a containerised API, with pretrained models available on Hugging Face. The API also powers the [CrystaLLM-π web application](https://crystallm-pi.psdi.ac.uk/). Most models fit on a 16 GB GPU for training and need 1–2 GB for generation. Generation also runs on CPU.

## Reproducing the paper

The notebooks and configs for ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299) are in [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper). The [v1.3.0 tag](https://github.com/C-Bone-UCL/CrystaLLM-pi/releases/tag/v1.3.0) preserves the version used for those studies.

Use this repo for current development and generation with any released model.

## Key Features

- **Structure generation**: generate CIFs from scratch or from a formula, with optional Z and space-group inputs.
- **Property targets**: guide generation with one or more numerical properties.
- **Fine-tuning**: train Prefix or Residual models on your own data. Residual models also support missing property values.
- **Evaluation**: check validity, uniqueness, novelty and stability.
- **Released models**: download pretrained models from Hugging Face.

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

Mapped condition lists, Z-searches, perplexity ranking and every configuration option are on the [quickstart page](https://c-bone-ucl.github.io/CrystaLLM-pi/quickstart/).

## Available Pre-trained Models

Each released model exists because a paper study or tutorial produced it. The table says which, so you know where to look for its training setup and evaluation.

| Model | Class | Conditioning | Origin |
|---|---|---|---|
| `c-bone/CrystaLLM-pi_ft_alex_mp_20-text` | GPT-2 | unconditional | **Recommended base model**, LeMat-Bulk pretrain finetuned on Alex-MP-20 CIFs |
| `c-bone/CrystaLLM-pi_base` | GPT-2 | unconditional | LeMaterial base model from the first paper |
| `c-bone/CrystaLLM-pi_mp_20_base` | GPT-2 | unconditional | mp-20 pretraining base from the paper's pretraining studies |
| `c-bone/CrystaLLM-pi_alex_mp_20_base` | GPT-2 | unconditional | alex-mp-20 pretraining base from the paper's dataset-size study |
| `c-bone/CrystaLLM-pi_SLME` | Prefix (legacy `PKV`) | solar efficiency (SLME), 0-33% | SLME discovery study, maintained here in [`T5_SLME`](notebooks/T5_SLME.ipynb) |
| `c-bone/CrystaLLM-pi_bandgap` | Prefix (legacy `PKV`) | bandgap + stability, 0-18 eV / 0-5 eV/atom | Pretraining-benefits study ([B1a notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B1a_Pretrain_benefits.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_density` | Prefix (legacy `PKV`) | density + stability, 0-25 g/cm3 / 0-0.1 eV/atom | Dataset-size study ([B2 notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B2_Dataset_size_study.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_Mattergen-XRD` | Residual (legacy `Slider`, top-20 peaks) | XRD peaks (theoretical patterns, fully ordered bias) | XRD recovery studies ([X_XRD notebooks](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/tree/main/notebooks) in the paper repo) |
| `c-bone/CrystaLLM-pi_Chili100K-XRD` | Residual (legacy `Slider`, top-20 peaks) | XRD peaks (experimental patterns) | Model used in the paper's [CHILI-100K recovery study](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/X_XRD_chili100k.ipynb) in the paper repo |

Model metadata (class, conditions, normalization) lives in [`_utils/model_registry.json`](_utils/model_registry.json). To generate with a model that is not in the table, pass a JSON file with the same schema via `--model_registry`. [`notebooks/T1_finetune_density_example.ipynb`](notebooks/T1_finetune_density_example.ipynb) walks through that full loop (finetune a density model, upload it, register it, generate with it).

The conditioning mechanism behind each class is described on the [models page](https://c-bone-ucl.github.io/CrystaLLM-pi/models/).

## Tutorial Notebooks

Four notebooks in [`notebooks/`](notebooks/) cover the maintained workflows end to end:

* [`T1_finetune_density_example.ipynb`](notebooks/T1_finetune_density_example.ipynb): finetune a base model on your own property dataset, push it to the Hub, register it, and generate with it
* [`T2_load_and_generate.ipynb`](notebooks/T2_load_and_generate.ipynb): generate structures with the released Hub models (courtesy of [Joley Lin](https://github.com/yhjollin/))
* [`T3_API_density_example.ipynb`](notebooks/T3_API_density_example.ipynb): predict density for a composition through the containerised API
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
