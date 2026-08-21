# CrystaLLM-<span style="font-size: 1.2em;">&pi;</span>

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> is a Transformer-based system for generating crystalline structures as CIF files. It supports both unconditional generation and several conditional architectures that can generate structures based on target properties like bandgap, density, photovoltaic efficiency and XRD patterns, including raw experimental powder scans.

<div align="center" markdown>
![CrystaLLM-pi framework overview](images/Framework_github.png){ width="75%" style="background-color:white" }
</div>

## Statement of need

CrystaLLM-<span style="font-size: 1.2em;">π</span> is a lightweight framework for conditional crystal structure generation. You can fine-tune a pretrained model on numerical properties and generate structures aimed at a target value, or generate directly from one of the open-sourced models.

The framework supports a large range of numerical conditioning variants without requiring a separate generation framework for each application. We have demonstrated the approach for materials discovery with target functional properties and for recovering crystal structures from experimental characterisation data.

The repository provides the tools needed to apply the method to new problems, including an installable codebase, tutorials, notebooks, documentation, pretrained models on the Hugging Face Hub, a containerised API for model serving, and a web application for interactive generation.

Modern transformers are memory efficient enough that most training and inference runs on a GPU, or on CPU. Most models fit on a 16GB card for training and need 1-2GB for light generation.

Together this keeps the framework within reach of any researcher who wants to work with conditional generative models.

## Key Features

- **Unconditional Generation**: Generate crystal structures from structural/composition priors
- **Property-Guided Generation**: Generate crystal structures conditioned on target properties + structural priors
- **Multiple Architectures**: Two conditioning mechanisms, prefix and residual, plus the unconditional base model. Prefix conditioning also takes full XRD patterns through a Perceiver resampler. 
- **Flexible Conditioning**: You can use any set of numerical properties to condition, and the Residual conditioning natively handles heterogeneous datasets (some properties are missing in the dataset but not others...)
- **Evaluation of output structures**: Scripts for validity, uniqueness, novelty and stability metrics
- **HuggingFace Integration**: Pre-trained models available on HF Hub

## Where to start

| If you want to | Page | Notebook |
|---|---|---|
| Install the package | [Installation](install.md) | |
| Generate with a released model | [Quickstart](quickstart.md) | [T2](tutorials.md) |
| Understand the conditional families | [Models](models.md) | |
| Finetune on your own property | [Training from scratch](training.md) | [T1](tutorials.md) |
| Recover a structure from an XRD scan | [Quickstart](quickstart.md) | [T4](tutorials.md) |
| Run it as an HTTP service | [API service](api-service.md) | [T3](tutorials.md) |
| Screen for a target property | [Training from scratch](training.md) | [T5](tutorials.md) |
| Look up a function | [API reference](api/cli.md) | |

## Reproducing the paper

The studies from the ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299) paper live in the standalone repo [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper). The next graph-conditioned knowledge distillation paper reproduction code can be accessed in [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph). Otherwise, all models so far are accessible to generate with here.

This repository stays the maintained package and is still under active development.
