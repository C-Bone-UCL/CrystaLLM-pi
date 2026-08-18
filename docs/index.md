# CrystaLLM-<span style="font-size: 1.2em;">&pi;</span>

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> is a Transformer-based system for generating crystalline structures as CIF files. It supports both unconditional generation and several conditional architectures that can generate structures based on target properties like bandgap, density, photovoltaic efficiency and XRD patterns, including raw experimental powder scans.

<div align="center">
<img src="images/Framework_github.png" width="75%" style="background-color:white;"/>
</div>

## Statement of need

Most generative models for inorganic crystals sample the space of stable structures. To get a material with a particular bandgap, or the structure behind a measured powder pattern, you generate broadly and filter afterwards, and most of the sampling budget goes on structures you throw away.

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> conditions the generation itself. It builds on [CrystaLLM](https://www.nature.com/articles/s41467-024-54639-7), a GPT trained on CIF text, and injects property vectors or diffraction patterns into the model. The target is fixed before sampling starts. The output stays CIF text, which keeps symmetry, occupancies and composition explicit, and nothing decodes between the model and a structure you can open.

It is built for materials researchers working on their own problem rather than a benchmark. Pretrained models sit on the Hugging Face Hub, finetuning on a new property is a config file, the containerised HTTP service runs generation on a shared GPU box, and the evaluation scripts cover validity, uniqueness, novelty, energy above hull and XRD match. The XRD models read a raw diffractometer scan, so recovering a structure from an unindexed pattern is one command.

## Key Features

- **Unconditional Generation**: Generate crystal structures from structural/composition priors
- **Property-Guided Generation**: Generate crystal structures conditioned on target properties + structural priors
- **Multiple Architectures**: Three conditional families (Prefix, PrefixXRD, Residual) plus the unconditional base model. Legacy PKV and Slider checkpoints still generate.
- **Flexible Conditioning**: You can use any set of numerical properties to condition, and one of the models handles heterogeneous datasets (some properties are missing in the dataset but not others...)
- **Evaluation of output structures**: Scripts for validity, uniqueness, novelty and stability metrics
- **HuggingFace Integration**: Pre-trained models available on HF Hub

## Where to start

| If you want to | Page | Notebook |
|---|---|---|
| Install the package | [Installation](install.md) | |
| Generate with a released model | [Quickstart](quickstart.md) | [T2](tutorials.md) |
| Understand the conditional families | [Models](models.md) | |
| Finetune on your own property | [Training from scratch](training.md) | [T1](tutorials.md) |
| Recover a structure from an XRD scan | [Quickstart](quickstart.md) | [T5](tutorials.md) |
| Run it as an HTTP service | [API service](api-service.md) | [T3](tutorials.md) |
| Screen for a target property | [Training from scratch](training.md) | [T6](tutorials.md) |
| Look up a function | [API reference](api/conventions.md) | |

## Reproducing the paper

The studies from the ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299) paper live in the standalone repo [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper) (or [v1.3.0 tag](https://github.com/C-Bone-UCL/CrystaLLM-pi/releases/tag/v1.3.0) of this repository). The next graph-conditioned knowledge distillation paper reproduction code can be accessed in [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph). Otherwise, all models so far are accessible to generate with here.

This repository stays the maintained package: it is an ongoing project and improvements are continuously being implemented!
