# CrystaLLM-<span style="font-size: 1.2em;">&pi;</span>

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> generates crystal structures as CIF files, with optional targets such as bandgap, density, photovoltaic efficiency or XRD peaks. You can use a released model or fine-tune one on your own data.

<div align="center" markdown>
![CrystaLLM-pi framework overview](images/Framework_github.png){ width="75%" style="background-color:white" }
</div>

The same training and generation workflow supports different numerical properties. To work with a new property, prepare a dataset with the corresponding values and fine-tune a model.

This repo includes the package, tutorials, documentation and a containerised API, with pretrained models available on Hugging Face. The API also powers the [CrystaLLM-π web application](https://crystallm-pi.psdi.ac.uk/). Most models fit on a 16 GB GPU for training and need 1–2 GB for generation. Generation also runs on CPU.

## Key Features

- **Structure generation**: generate CIFs from scratch or from a formula, with optional Z and space-group inputs.
- **Property targets**: guide generation with one or more numerical properties.
- **Fine-tuning**: train Prefix or Residual models on your own data. Residual models also support missing property values.
- **Evaluation**: check validity, uniqueness, novelty and stability.
- **Released models**: download pretrained models from Hugging Face.

## Where to start

| If you want to | Page | Notebook |
|---|---|---|
| Install the package | [Installation](install.md) | |
| Generate with a released model | [Quickstart](quickstart.md) | [T2](tutorials.md) |
| Understand the conditional families | [Models](models.md) | |
| Finetune on your own property | [Training from scratch](training.md) | [T1](tutorials.md) |
| Recover a structure from XRD peaks | [Quickstart](quickstart.md) | |
| Run it as an HTTP service | [API service](api-service.md) | [T3](tutorials.md) |
| Screen for a target property | [Training from scratch](training.md) | [T5](tutorials.md) |
| Look up a function | [API reference](api/cli.md) | |

## Reproducing the paper

The notebooks and configs for ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299) are in [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper). Use this repo for current development and generation with any released model.
