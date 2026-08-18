# Model Types

!!! tip "Run it in a notebook"
    [`T5_XRD_continuous.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T5_XRD_continuous.ipynb) shows what the continuous-XRD conditioning signal looks like at every stage, from raw scan to the `(1000, 2)` `[Q, I]` profile the model reads.

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> supports one unconditional and several conditional model architectures, allowing for both standard and property-driven generation. The desired model can be selected during training using the `--activate_conditionality` flag.

### 1. Unconditional CrystaLLM

Do not set the `--activate_conditionality` flag.

Standard CrystaLLM/GPT-2 architecture for generative tasks. Learns underlying patterns and grammar of CIF files without explicit property guidance.

### 2. Conditional Models

#### a. Prefix-GPT (Prefix Attention)

`--activate_conditionality="Prefix"`

Injects property information directly into the attention mechanism's past key-values. This allows the model to steer generation based on desired properties by concatenating conditional embeddings at each transformer layer. Provides strong conditioning while maintaining straightforward implementation. Based on ghost tokens from the [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190). Successor of the paper-era PKV family (GELU activation and a revised KV layout), model code from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph).

<div align="center">
<img src="images/Prefix_github.png" width="75%" style="background-color:white;"/>
</div>

#### b. PrefixXRD-GPT (Perceiver Prefix Attention)

`--activate_conditionality="PrefixXRD"`

Prefix conditioning driven by a Perceiver Resampler that encodes full powder XRD patterns. 
During training, discrete `[Q, I]` peak lists are synthetically broadened and noised on the fly. During inference, we can use continuous unprocessed XRD profiles to condition generation on the full xrd signal (no need to pick peaks, no restrictions in incident angles etc.)

#### c. Residual-GPT (Residual Attention)

`--activate_conditionality="Residual"`

Conditioning information is dynamically injected into each attention block via a 'slider' mechanism: two separate attention mechanisms at every token generation, one for main text and one for conditions, combined via weighted sum. Handles missing or unspecified conditions with softer conditioning. Successor of the paper-era Slider family. 

<div align="center">
<img src="images/Residual_github.png" width="75%" style="background-color:white;"/>
</div>

### Legacy families: PKV and Slider (generation only)

`PKV` and `Slider` are the paper-era architectures (called `Prefix attention` and `Residual attention` in the paper). The released hub checkpoints keep generating exactly as before, but **training them is blocked**. `--activate_conditionality="PKV"`/`"Slider"` raises with a pointer to the successors.

> The paper additionally benchmarks two comparative baselines (Prepend-GPT and Raw-GPT). These live in the reproduction repo [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper).

<br>
<br>

## Available Pre-trained Models

Each released model exists because a paper study or tutorial produced it. The table says which, so you know where to look for its training setup and evaluation.

| Model | Class | Conditioning | Origin |
|---|---|---|---|
| `c-bone/CrystaLLM-pi_ft_alex_mp_20-text` | GPT-2 | unconditional | **Recommended base model**, LeMat-Bulk pretrain finetuned on Alex-MP-20 CIFs, taken from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph)|
| `c-bone/CrystaLLM-pi_Chili100K-cXRD` | PrefixXRD | continuous XRD profiles | CHILI-100K KD student from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph), different priors, in theory better for more experimental structs |
| `c-bone/CrystaLLM-pi_alex_mp_20-cXRD` | PrefixXRD | continuous XRD profiles | Alex-MP-20 KD student from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph), in theory more coverage than the chili model |
| `c-bone/CrystaLLM-pi_base` | GPT-2 | unconditional | LeMaterial base model from the first paper |
| `c-bone/CrystaLLM-pi_mp_20_base` | GPT-2 | unconditional | mp-20 pretraining base from the paper's pretraining studies |
| `c-bone/CrystaLLM-pi_alex_mp_20_base` | GPT-2 | unconditional | alex-mp-20 pretraining base from the paper's dataset-size study |
| `c-bone/CrystaLLM-pi_SLME` | PKV (legacy) | solar efficiency (SLME), 0-33% | SLME discovery study, maintained here in [`T6_SLME`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T6_SLME.ipynb) |
| `c-bone/CrystaLLM-pi_bandgap` | PKV (legacy) | bandgap + stability, 0-18 eV / 0-5 eV/atom | Pretraining-benefits study ([B1a notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B1a_Pretrain_benefits.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_density` | PKV (legacy) | density + stability, 0-25 g/cm3 / 0-0.1 eV/atom | Dataset-size study ([B2 notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B2_Dataset_size_study.ipynb) in the paper repo) |
| `c-bone/CrystaLLM-pi_Mattergen-XRD` | Slider (legacy top-20 pipeline) | XRD peaks (theoretical patterns, fully ordered bias) | XRD recovery studies ([X_XRD notebooks](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/tree/main/notebooks) in the paper repo) |
| `c-bone/CrystaLLM-pi_Chili100K-XRD` | Slider (legacy top-20 pipeline, superseded by the cXRD model + [`T5_XRD_continuous`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T5_XRD_continuous.ipynb)) | XRD peaks (experimental patterns) | CHILI-100K recovery study from the first paper |

Model metadata (class, conditions, normalization) lives in [`_utils/model_registry.json`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/model_registry.json). To generate with a model that is not in the table, pass a JSON file with the same schema via `--model_registry`. [`notebooks/T1_finetune_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T1_finetune_density_example.ipynb) walks through that full loop (finetune a density model, upload it, register it, generate with it).

<br>

> **Continuous-XRD model (`Chili100K-cXRD`, recommended):** pass the **raw diffractometer scan** directly via `--xrd_files` (`.csv`, `.xy`, `.txt`, `.dat`; arbitrary header lines are skipped automatically). The pipeline converts 2theta to Q using your `--xrd_wavelength` (CuKa1 assumed with a warning when omitted), removes the background with SNIP, resamples onto the model's 1000-point Q grid and max-normalizes intensity. Inspect the transform with [`process_exp_xrd_continuous.py`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_preprocessing/process_exp_xrd_continuous.py) `--save_plot` before generating. `--xrd_files` is required for this model, it has no missing-conditioning fallback.
>
> **Legacy Slider models (top-20 pipeline):** provide **pre-picked peak data** (not raw profiles) via `--xrd_files`. Many open-source programs do this (e.g., [fityk](https://fityk.nieto.pl/) for academic use). The preprocessing engine converts picked peaks to the expected CuKa wavelength (via `--xrd_wavelength`), filters valid ranges, normalizes intensities, and selects the top peaks. Redundant peaks from additional radiation sources need removing first (e.g. K-alpha2 peaks when irradiated with K-alpha1 and K-alpha2). If `--xrd_files` is omitted for a Slider model, generation still runs with missing conditioning values.
