# CrystaLLM-2.0

## Description

**CrystaLLM-2.0** is a Transformer-based model for generating crystalline structures as Crystallographic Information Files (CIFs). This project provides a complete codebase for both conditional (property-driven) and non-conditional GPT-style training, generation, and evaluation.

The models are based on the [CrystaLLM Paper](https://www.nature.com/articles/s41467-024-54639-7)

It leverages a few tools:
- **Hugging Face**: For dataset hosting, model sharing, and tokenizer access.
- **Weights & Biases**: For experiment tracking and visualization.
- **CodeCarbon**: For tracking the carbon footprint of training runs.
- **DeepSpeed**: For efficient, distributed training on multi-GPU systems.

## Table of Contents
- [CrystaLLM-2.0](#crystallm-20)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Configuration](#configuration)
    - [API Keys](#api-keys)
  - [Model Types:](#model-types)
    - [1. Unconditional CrystaLLM](#1-unconditional-crystallm)
    - [2. Conditional Models](#2-conditional-models)
      - [a. PKV-GPT](#a-pkv-gpt)
      - [b. Prepend-GPT](#b-prepend-gpt)
      - [c. Slider-GPT](#c-slider-gpt)
      - [d. Raw-GPT](#d-raw-gpt)
  - [Usage](#usage)
    - [Data Processing Pipeline](#data-processing-pipeline)
    - [Training the Model](#training-the-model)
      - [Base Model Pretraining](#base-model-pretraining)
      - [Conditional Fine-tuning](#conditional-fine-tuning)
    - [Generating Crystal Structures](#generating-crystal-structures)
      - [Step 1: Create Prompts](#step-1-create-prompts)
      - [Step 2: Generate CIFs](#step-2-generate-cifs)
      - [Optional: Evaluate CIFs (pass non-processed CIFs)](#optional-evaluate-cifs-pass-non-processed-cifs)
      - [Step 3: Post-process](#step-3-post-process)
  - [Evaluation](#evaluation)
    - [Core Evaluation Metrics](#core-evaluation-metrics)
      - [1. VUN Metrics (Validity, Uniqueness, Novelty)](#1-vun-metrics-validity-uniqueness-novelty)
      - [2. Energy Above Hull (Stability)](#2-energy-above-hull-stability)
      - [3. Other metrics](#3-other-metrics)
  - [Studies](#studies)
  - [License](#license)
  - [Contact](#contact)

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- Conda for environment management
- (Optional) NVIDIA GPU for accelerated training.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/C-Bone-UCL/CrystaLLM-2.0.git
    cd CrystaLLM-2.0
    ```

2.  **Create a virtual environment for the main package:**
    ```bash
    conda create -n CrystaLLM-2.0_env python=3.10
    conda activate CrystaLLM-2.0_env
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the repo in editable:**
    ```bash
    pip install -e .
    ```

5.  **(Optional) ALIGNN Environment:** For property prediction(of bandgap) set up a separate environment for ALIGNN to avoid dependency conflicts.
    ```bash
    conda create -n alignn_env python=3.10
    conda activate alignn_env
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
    pip install -r requirements-alignn.txt
    ```

## Configuration

### API Keys
For integration with Hugging Face and Weights & Biases, create a file named `API_keys.jsonc` in the root directory. This file uses the `.jsonc` format, which allows comments.

```jsonc
// filepath: API_keys.jsonc
{
  "HF_key": "hf_xxxYYYzzz", // Your Hugging Face token
  "wandb_key": "your_wandb_api_key_here" // Your Weights & Biases key
}
```

## Model Types:

CrystaLLM-2.0 supports one unconditional and four conditional model architectures, allowing for both standard and property-driven generation. The desired model can be selected during training using the `--activate_conditionality` flag.

### 1. Unconditional CrystaLLM
Do not set the `--activate_conditionality` flag.

This is the standard CrystaLLM/GPT-2 architecture for generative tasks. It learns the underlying patterns and grammar of CIF files without any explicit property guidance.

### 2. Conditional Models

#### a. PKV-GPT 
(`--activate_conditionality="PKV"`)

PKV (Property-Key-Value) conditioning injects property information directly into the attention mechanism's past key-values. This allows the model to steer generation based on the desired properties by concatenating the conditional embeddings at each layer of the transformer in the attention mechanism. It is a 'strong' conditioning method and balances expressivity of conditioning and straightforward implementation.

#### b. Prepend-GPT 
(`--activate_conditionality="Prepend"`)

This model prepends a sequence of learned embeddings (a “soft prompt”) to the input sequence. These prefix tokens are trained to represent the desired conditional properties, guiding the model's output. They are not passed in as regular tokens though. This method is also strongly conditioning and is traightforward, but is less flexible and performs less well than attention based conditioning.

#### c. Slider-GPT 
(`--activate_conditionality="Slider"`)

A novel architecture where conditioning information is dynamically injected into each attention block via a 'slider' mechanism. The result is two separate attention mechanisms calculated at every token generation, one for the main text and one for the condition. The attention scores are added via a weighted sum. A main benefit is the ability to handle missing or unspecified conditions seamlessly. It is a 'softer' conditioning as the weight of the conditioned attention is initialised at 0 during finetuning.

#### d. Raw-GPT 
(`--activate_conditionality="Raw"`)

A baseline approach where numerical condition values are simply converted to text and appended to the input prompt. This method requires no architectural changes but increases the sequence length. Implemented for comparison but is less performant.


## Usage

### Data Processing Pipeline
Before training, crystallographic data needs preprocessing to ensure quality and consistency. The pipeline consists of deduplication, cleaning, optional XRD conditioning, and formatting for Hugging Face.

**Step 1: Data Preparation**

Your input data should be a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) saved as a [Parquet file](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format). This format efficiently handles large databases and integrates seamlessly with Hugging Face datasets.

**Required columns:**
- `Database`: Source database name
- `Reduced Formula`: Standard reduced chemical formula (used in evaluation metrics) 
- `CIF`: The crystallographic structure in CIF format

**For structure recovery tasks:**
- `Material ID`: Database identifier (like mp-390 for materials project etc.)

**Optional columns:**
- `<Property Columns>`: Target properties for conditional generation (e.g., "Bandgap (eV)", "Density (g/cm^3)")
- `condition_vector`: If we have an already standardised and compatible condition vector (like in the XRD studies), we can store a column which contains the vector string, removing the need for any additional processing

**Step 2: Deduplication and Filtering**

Remove duplicate structures and filter invalid entries using `_deduplicate.py`. This script identifies unique structures based on chemical formula and space group, keeping the one with lowest volume per formula unit.

```bash
python _utils/_preprocessing/_deduplicate.py \
  --input_file /path/to/raw_data.parquet \
  --output_parquet /path/to/deduplicated_data.parquet \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --filter_na_columns "['Bandgap (eV)']" \
  --filter_zero_columns "['Density (g/cm^3)']" \
  --filter_negative_columns "['Bandgap (eV)']"
```

**Available filters:**
- `--filter_na_columns`: Remove entries with N/A or NaN values
- `--filter_zero_columns`: Remove entries with zero values  
- `--filter_negative_columns`: Remove entries with negative values

**Step 3: CIF Cleaning and Property Normalization**

Standardize CIF format and normalize properties using `_cleaning.py`. This performs several critical transformations:

- Standardizes CIF syntax and formatting
- Adds atomic property blocks (electronegativity, radius, ionic radius)
- Rounds numerical values to consistent precision
- Normalizes property values to [0,1] range for stable training

```bash
python _utils/_preprocessing/_cleaning.py \
  --input_parquet /path/to/deduplicated_data.parquet \
  --output_parquet /path/to/cleaned_data.parquet \
  --num_workers 8 \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --property1_normaliser "power_log" \
  --property2_normaliser "linear"
```

**Normalization methods:**
- `linear`: Linear scaling to [0,1]
- `power_log`: Power transformation followed by log scaling
- `signed_log`: Signed logarithmic transformation
- `log10`: Base-10 logarithmic scaling
- `None`: No normalization

**(Optional): XRD Pattern Generation**

For XRD-based conditional generation, compute powder diffraction patterns using `_calculate_XRD.py` which leverages pymatgens XRD calculator. This makes condition vectors from the top-20 diffraction peaks (from their CIFs).

**Step 5: Dataset Upload to Hugging Face**

Convert to Hugging Face format with train/validation/test splits:

```bash
python _utils/_preprocessing/_save_dataset_to_HF.py \
  --input_parquet /path/to/processed_data.parquet \
  --output_parquet "your-dataset-name" \
  --test_size 0.1 \
  --valid_size 0.1 \
  --HF_username "your-username" \
  --save_hub \
  --save_local
```

**Split options:**
- `--duplicates`: Prevents data leakage by splitting on Material ID
- `--test_size 0.0 --valid_size 0.0`: All data goes to training set
- Uses existing "Split" column if present in DataFrame

<!-- > See the [Data Preprocessing Notebook](notebooks/A_Preprocess_data.ipynb). -->

### Training the Model

CrystaLLM-2.0 supports both single-GPU and multi-GPU training. All training parameters are configured via `.jsonc` config files. See example configs in [`_config_files/`](_config_files/).

#### Base Model Pretraining

You can train a model from scratch on CIFs using the `_config_files/og_train/lematerial-small.jsonc` config. Or you ca simply download the pretrained model using:

!!!!!!!todo!!!!!!!!!!

#### Conditional Fine-tuning

Fine-tune a pretrained base model to condition on specific properties:

**Single GPU:**
```bash
python _train.py --config _config_files/cg_train/ft-slme/slme_ft-PKV.jsonc
```

**Multi-GPU:**
```bash
torchrun --nproc_per_node=2 _train.py --config _config_files/cg_train/ft-slme/slme_ft-PKV.jsonc
```

To get an idea of what the arguments do and now to use them, go to the [Args Script](_args.py_)

<!-- > See the [Training Notebook](notebooks/B_Train_model.ipynb) for examples. -->

### Generating Crystal Structures

Generating CIFs with a standardised format is a 3-step process here: create prompts, generate CIFs, and post-process. All outputs are `.parquet` files for easy handling.

#### Step 1: Create Prompts

Use `_utils/_generating/make_prompts.py` to create input prompts for generation.

**Manual Prompts (specify compositions and conditions):**
```bash
# For conditional generation with multiple property values
python _utils/_generating/make_prompts.py \
  --manual \
  --compositions "Na1Cl1,K2S1" \
  --condition_lists "0.2,0.5,1.0" "0.0" \
  --output_parquet "test_prompts.parquet"
```
> Here the first list is for example property 1 (bandgap) and second is property 2 (ehull stability)
> Here we can imagine that for each composition, we will make a prompts for each of the condition list pairs. So well have 2 comps * 3 bandgaps * 1 ehull = 6 prompts for each composition-property pair

**Automatic Prompts (extract from existing datasets):**
```bash
# Extract prompts from HF dataset with different detail levels
python _utils/_generating/make_prompts.py \
  --automatic \
  --HF_dataset "c-bone/mp_20_pxrd" \
  --split "test" \
  --level "level_2" \
  --condition_columns "Condition Vector" \
  --output_parquet "dataset_prompts.parquet"
```

**Prompt detail levels:**
- `level_1`: Minimal (unconditional generation)
- `level_2`: Composition only  
- `level_3`: Composition + atomic properties
- `level_4`: Up to space group information


> In some cases like with XRD studies, we've already constructed the condition vector rather than just storing them in associated columns (because its quite large so would be inefficient to have distinct columsn for each peak and associated intensity - but that would be doable)

#### Step 2: Generate CIFs

Use `_utils/_generating/generate_CIFs.py` with a config file containing your model and generation settings:

```bash
python _utils/_generating/generate_CIFs.py \
  --config _config_files/cg_eval/your_eval_config.jsonc
```

**Required config settings:**
```jsonc
{
  "model_ckpt_dir": "path/to/your/model/checkpoint",
  "input_parquet": "test_prompts.parquet", 
  "output_parquet": "generated_cifs.parquet",
  "activate_conditionality": "PKV",  // or "Prepend", "Slider", "Raw", "None"
  "gen_max_length": 1024, // whatever your contect length here or below
  "do_sample": true,
  "num_return_sequences": 1, //How much 1 GPU can handle at once
  "num_repeats": 2 // num_return_sequences * num_repeats = total gens per prompt
}
```

#### Optional: Evaluate CIFs (pass non-processed CIFs)
**Evaluate** structural validity:
```bash
python _utils/_generating/evaluate_CIFs.py \
  --input_parquet "generated_cifs.parquet" \
  --num_workers 8 \
  --save_valid_parquet
```
> `--save_valid_parquet` can be set if you want to save just the chemically cvalid outputs
> you dont need to evaluate your output cifs, but its quick check to see if your model isnt outputting gibberish

#### Step 3: Post-process

**Post-process** generated CIFs to standard format:
```bash
python _utils/_generating/postprocess.py \
  --input_parquet "generated_cifs.parquet" \
  --output_parquet "processed_cifs.parquet" \
  --num_workers 4
```

<!-- > See complete examples in the [Generation Notebook](notebooks/C_Generate_CIFs.ipynb). -->

## Evaluation

CrystaLLM-2.0 provides comprehensive metrics to evaluate generated crystal structures across structural validity, property accuracy, and material stability.

### Core Evaluation Metrics

#### 1. VUN Metrics (Validity, Uniqueness, Novelty)
Essential metrics for assessing generation quality using structural analysis.

```bash
python _utils/_metrics/VUN_metrics.py \
  --gen_data generated_structures.parquet \
  --huggingface_dataset "c-bone/mp_20" \
  --output_csv vun_results.csv \
  --num_workers 8
```

**Metrics computed:**
- **Validity**: Structures with correct spacegroup, reasonable bond lengths, and consistent atom multiplicities
- **Uniqueness**: Distinct structures within the generated set (using BAWL hashing)
- **Novelty**: Structures not present in the training dataset (from Structure Matcher if same Reduced Formula)

#### 2. Energy Above Hull (Stability)
Calculate thermodynamic stability using MACE energy predictions and Materials Project corrections.

```bash
python _utils/_metrics/mace_ehull.py \
  --post_parquet postprocessed_structures.parquet \
  --output_parquet stability_results.parquet \
  --num_workers 4
```

Lower E-hull values indicate higher thermodynamic stability. In theory between anything below 0.1 ev/atom is metastable, here if we account for MAE of the surrogate model (See MACE-MP-0 on Matbench Discovery), we can loosen theshold to 0.157 if desired.

#### 3. Other metrics
XRD, Bandgap, Density, or challenge set benchmark metrics can also be calculated for further property analysis. If wanted, find the associated scripts in [the _metrics folder](_utils/_metrics/)

> **Environment Note**: ALIGNN-based scripts require the separate `alignn_env` environment due to dependency conflicts. Use `conda run -n alignn_env` for these tools.

## Studies

To have a look at how the experiments from the paper were carried out, the notebooks used to generate the results are available in the [Notebooks that start with X_](notebooks/)

Full information given for each, but most detail and example of a full end to end pipeline for Discovery can be found in the mp-20 notebook and recovery SLME notebook

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please contact cyprien.bone.24@ucl.ac.uk or raise an issue on the gighub page