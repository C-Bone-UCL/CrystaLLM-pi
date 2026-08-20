# Model Types

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> supports unconditional and conditional model architectures for standard and property-conditioned generation. Select the desired architecture during training with the `--activate_conditionality` flag.

### 1. Unconditional CrystaLLM

Do not set the `--activate_conditionality` flag.

This is the standard CrystaLLM/GPT-2 architecture. It learns the patterns and grammar of CIF files without explicit property conditioning.

### 2. Conditional Models

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> provides two mechanisms for incorporating property information. **Prefix** conditioning adds the conditioning information to the attention mechanism as cached key-values. **Residual** conditioning incorporates it into each attention block via a residual term which modulates the base attention. Both are selected using `--activate_conditionality`.

#### a. Prefix-GPT (Prefix Attention)

`--activate_conditionality="Prefix"`

Property information is injected into the attention mechanism through its past key-values. Conditional embeddings are added at each transformer layer, allowing generation to be guided towards specified properties. The approach is based on ghost tokens from the [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190). The model implementation is based on [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph).

<div align="center" markdown>
![Prefix attention conditioning](images/Prefix_github.png){ width="75%" style="background-color:white" }
</div>

#### b. PrefixXRD-GPT (Perceiver Prefix Attention)

`--activate_conditionality="PrefixXRD"`

This model uses the same prefix conditioning mechanism, preceded by a Perceiver Resampler module. Scalar properties contain only a small number of values, whereas a diffraction pattern may contain thousands of measurements. The resampler maps the large incoming information dense tensor to a fixed number of latent vectors, which are then used as the prefix key-values. This allows compression to filter out noise and keep only relevant condtiioning information, and also allows us to feed varying length inputs into the prefix model which normally cant handle heterogeneous data.

During training, discrete `[Q, I]` peak lists are synthetically broadened and noised on the fly. During inference, continuous, unprocessed XRD profiles can be used directly as conditioning input. This avoids peak selection and trimming, and does not impose restrictions on the incident angle range compared to the old XRD conditioning model we had.

#### c. Residual-GPT (Residual Attention)

`--activate_conditionality="Residual"`

Conditioning information is injected into each attention block through a slider mechanism. At each token generation step, separate attention mechanisms process the main text and conditioning information (using the same query), and their outputs are combined with a weighted sum. This also allows conditions to be omitted or left unspecified, resulting in softer conditioning.

<div align="center" markdown>
![Residual attention conditioning](images/Residual_github.png){ width="75%" style="background-color:white" }
</div>

??? note "Paper-era checkpoints"
    Models released with the paper were trained with earlier implementations of the same two conditioning mechanisms. These are named `PKV` (prefix) and `Slider` (residual) in the code, and `Prefix attention` and `Residual attention` in the paper. They are loaded and used automatically based on the model name - you can use them in the `_load_and_generate.py` script. The two generations remain separate classes because their weights are not interchangeable - we made some minor improvements to the model internals. Training these legacy classes is disabled, and `--activate_conditionality="PKV"` or `"Slider"` raises an error pointing to the current implementations.

    > The paper also benchmarks two comparative baselines, Prepend-GPT and Raw-GPT. These are available in the reproduction repository [CrystaLLM-pi-paper](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper).

<br>

## Available Pre-trained Models

Each released model corresponds to a paper study or tutorial. The table identifies its origin and conditioning setup, so the corresponding training and evaluation details can be found in the relevant material.

| Model | Class | Conditioning | Origin |
|---|---|---|---|
| `c-bone/CrystaLLM-pi_ft_alex_mp_20-text` | GPT-2 | unconditional | **Recommended base model**, a LeMat-Bulk pretrained model fine-tuned on Alex-MP-20 CIFs, from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph) |
| `c-bone/CrystaLLM-pi_Chili100K-cXRD` | PrefixXRD | continuous XRD profiles | CHILI-100K KD student from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph), has larger unit cell and experimental structure oriented priors (better for difficult larger structs) |
| `c-bone/CrystaLLM-pi_alex_mp_20-cXRD` | PrefixXRD | continuous XRD profiles | Alex-MP-20 KD student from [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph), with broader coverage than the CHILI model (better for recovering known structs) |
| `c-bone/CrystaLLM-pi_base` | GPT-2 | unconditional | LeMat-Bulk base model from the first paper |
| `c-bone/CrystaLLM-pi_mp_20_base` | GPT-2 | unconditional | MP-20 pretraining base from the paper's pretraining studies |
| `c-bone/CrystaLLM-pi_alex_mp_20_base` | GPT-2 | unconditional | Alex-MP-20 pretraining base from the paper's dataset-size study |
| `c-bone/CrystaLLM-pi_SLME` | Prefix (legacy `PKV`) | solar efficiency (SLME), 0-33% | SLME discovery study, maintained in [`T5_SLME`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T5_SLME.ipynb) |
| `c-bone/CrystaLLM-pi_bandgap` | Prefix (legacy `PKV`) | bandgap + stability, 0-18 eV / 0-5 eV/atom | Pretraining-benefits study ([B1a notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B1a_Pretrain_benefits.ipynb) in the paper repository) |
| `c-bone/CrystaLLM-pi_density` | Prefix (legacy `PKV`) | density + stability, 0-25 g/cm3 / 0-0.1 eV/atom | Dataset-size study ([B2 notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/B2_Dataset_size_study.ipynb) in the paper repository) |
| `c-bone/CrystaLLM-pi_Mattergen-XRD` | Residual (legacy `Slider`, top-20 peaks) | XRD peaks (theoretical patterns, fully ordered bias) | XRD recovery studies ([X_XRD notebooks](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/tree/main/notebooks) in the paper repository) |
| `c-bone/CrystaLLM-pi_Chili100K-XRD` | Residual (legacy `Slider`, top-20 peaks) | XRD peaks (experimental patterns) | CHILI-100K recovery study from the first paper [CHILI-100K notebook](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/notebooks/X_XRD_chili100k.ipynb) |

Model metadata, including the model class, conditioning variables, and normalisation, is stored in [`_utils/model_registry.json`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/model_registry.json). To generate with a model that is not listed above with the `_load_and_generate.py` script, provide a JSON file with the same schema through `--model_registry`. See [`notebooks/T1_finetune_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T1_finetune_density_example.ipynb) for complete workflow, from fine-tuning a density model to uploading, registering, and generating with it.

<br>