# CrystaLLM-2.0

## Description

**CrystaLLM-2.0** is a Transformer-based model for generating crystalline structures as Crystallographic Information Files (CIFs). This project provides a codebase for conditional (property-driven) and non-conditional training inference and validations.

The models are based on the [CrystaLLM Paper](https://www.nature.com/articles/s41467-024-54639-7)

A few tools are used which you should setup for full functionality:
- **Hugging Face**: For dataset hosting, model sharing, and tokenizer access.
- **Weights & Biases**: For experiment tracking and visualisation.
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
      - [Direct HF Generation](#direct-hf-generation)
      - [Advanced: 3-Step Generation Pipeline](#advanced-3-step-generation-pipeline)
      - [Step 1: Create Prompts](#step-1-create-prompts)
      - [Step 2: Generate CIFs](#step-2-generate-cifs)
      - [Optional: Evaluate CIFs](#optional-evaluate-cifs)
      - [Step 3: Post-process](#step-3-post-process)
  - [Evaluation](#evaluation)
    - [Core Evaluation Metrics](#core-evaluation-metrics)
      - [1. VUN Metrics (Validity, Uniqueness, Novelty)](#1-vun-metrics-validity-uniqueness-novelty)
      - [2. Energy Above Hull (Stability)](#2-energy-above-hull-stability)
      - [3. Other metrics](#3-other-metrics)
  - [API](#api)
    - [Build container](#1-build-the-container)
    - [Run container](#2-run-the-container)
    - [Post to API](#3-api-usage)
  - [Studies](#studies)
  - [License](#license)
  - [Contact](#contact)

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- Conda for environment management
- (Optional) NVIDIA GPU for accelerated training. (it should also work with CPU)

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

This model prepends a sequence of learned embeddings (a "soft prompt") to the input sequence. These prefix tokens are trained to represent the desired conditional properties, guiding the model's output. They are not passed in as regular tokens though. This method is also strongly conditioning and is traightforward, but is less flexible and performs less well than attention based conditioning.

#### c. Slider-GPT 
(`--activate_conditionality="Slider"`)

A novel architecture where conditioning information is dynamically injected into each attention block via a 'slider' mechanism. The result is two separate attention mechanisms calculated at every token generation, one for the main text and one for the condition. The attention scores are added via a weighted sum. A main benefit is the ability to handle missing or unspecified conditions seamlessly. It is a 'softer' conditioning as the weight of the conditioned attention is initialised at 0 during finetuning.

#### d. Raw-GPT 
(`--activate_conditionality="Raw"`)

A baseline approach where numerical condition values are simply converted to text and appended to the input prompt. This method requires no architectural changes but increases the sequence length. Implemented for comparison but is less performant.


## Usage

### Data Processing Pipeline
Before training, crystallographic data needs preprocessing to compatibility with model. The pipeline is deduplication, cleaning, optional XRD condition vector generation. Then you can save the datasets/tokenizers to huggingface.

**Step 1: Data Preparation**

Your input data should be a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) saved as a [Parquet file](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format). This format handles large databases and integrates seamlessly with Hugging Face datasets.

**Required columns:**
- `Database`: Source database name (traceability)
- `Reduced Formula`: Standard reduced chemical formula (used in evaluation metrics for efficiency) 
- `CIF`: The crystallographic structure in CIF format (Pymatgen style is good)

**For structure recovery tasks:**
- `Material ID`: Database identifier (like mp-390 for materials project etc., well need theis if you want to test models generations against a 'true' struct)

**Optional columns:**
- `<Property Columns>`: Target properties for conditional generation (e.g., "Bandgap (eV)", "Density (g/cm^3)")
- `condition_vector`: If we have an already standardised and compatible condition vector (like in the XRD studies), we can store a column which contains the vector string, removing the need for any additional processing

**Step 2: Deduplication and Filtering**

Remove duplicate structures and filter invalid entries using `_deduplicate.py`. This script takes unique structures based on chemical formula and space group, keeping the one with lowest volume per formula unit.

```bash
python _utils/_preprocessing/_deduplicate.py \
  --input_file /path/to/raw_data.parquet \
  --output_parquet /path/to/deduplicated_data.parquet \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --filter_na_columns "['Bandgap (eV)']" \
  --filter_zero_columns "['Density (g/cm^3)']" \
  --filter_negative_columns "['Bandgap (eV)']"
```

**Available filters (for the property columns):**
- `--filter_na_columns`: Remove entries with N/A or NaN values
- `--filter_zero_columns`: Remove entries with zero values  
- `--filter_negative_columns`: Remove entries with negative values

**Step 3: CIF Cleaning and Property Normalisation**

Standardise CIF format and normalise properties using `_cleaning.py`. It:
- Adds atomic property blocks (electronegativity, radius, ionic radius)
- Rounds numerical values to consistent precision
- Normalises property values to [0,1] range for stable training
- Adds the 'variable brackets' mentioned in the paper

```bash
python _utils/_preprocessing/_cleaning.py \
  --input_parquet /path/to/deduplicated_data.parquet \
  --output_parquet /path/to/cleaned_data.parquet \
  --num_workers 8 \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --property1_normaliser "power_log" \
  --property2_normaliser "linear"
```

**Normalisation methods:**
- `linear`: Linear scaling to [0,1]
- `power_log`: Power transformation followed by log scaling
- `signed_log`: Signed logarithmic transformation
- `log10`: Base-10 logarithmic scaling
- `None`: No normalisation

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

### Training the Model

CrystaLLM-2.0 supports both single-GPU and multi-GPU training. All training parameters are configured via `.jsonc` config files. See example configs in [`_config_files/`](_config_files/).

#### Base Model Pretraining

You can train a model from scratch on CIFs using the `_config_files/og_train/lematerial-small.jsonc` config. Or you can simply download one of the pretrained models from the HF Hub (see generation section).

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

To get an idea of what the arguments do and how to use them, go to the [Args Script](_args.py)

### Generating Crystal Structures

Two approaches for structure generation: 
a quick direct method using pre-trained HF models, or do it yourself if its your own model or want full reproducibility.

#### Direct HF Generation

Use `_load_and_generate.py` for direct generation with pre-trained models from Hugging Face Hub:

**Available Models:**
- `c-bone/CrystaLLM-2.0_base`: Unconditional generation
- `c-bone/CrystaLLM-2.0_SLME`: Solar efficiency conditioning  
- `c-bone/CrystaLLM-2.0_bandgap`: Bandgap + stability conditioning
- `c-bone/CrystaLLM-2.0_density`: Density + stability conditioning
- `c-bone/CrystaLLM-2.0_COD-XRD`: XRD pattern conditioning

> **Note**: Some models may require requesting access. All condition values must be normalised [0-1] and properly formatted. I'll change this in the future so its easier to use.

**Examples:**

```bash
# Unconditional generation
python _load_and_generate.py \
  --hf_model_path "c-bone/CrystaLLM-2.0_base" \
  --manual --compositions "LiFePO4,TiO2" \
  --output_parquet structures.parquet

# Solar efficiency conditioning (SLME model)
python _load_and_generate.py \
  --hf_model_path "c-bone/CrystaLLM-2.0_SLME" \
  --manual --compositions "CsPbI3" \
  --condition_lists "0.8" \
  --output_parquet structures.parquet

# Bandgap conditioning (bandgap + stability)
python _load_and_generate.py \
  --hf_model_path "c-bone/CrystaLLM-2.0_bandgap" \
  --manual --compositions "Si" \
  --condition_lists "0.3" "0.0" \
  --output_parquet structures.parquet

# From existing prompts file
python _load_and_generate.py \
  --hf_model_path "c-bone/CrystaLLM-2.0_bandgap" \
  --input_parquet prompts.parquet \
  --output_parquet structures.parquet
```

**Prompt levels for manual generation:**
- `level_1`: Minimal (unconditional generation)
- `level_2`: Composition only (default)
- `level_3`: Composition + atomic properties  
- `level_4`: Up to space group information

#### Advanced: 3-Step Generation Pipeline

For custom models or advanced control.

#### Step 1: Create Prompts

Use `_utils/_generating/make_prompts.py` to create input prompts for generation.

**Manual Prompts (specify compositions and conditions):**
```bash
# Multi-property conditioning with different levels
python _utils/_generating/make_prompts.py \
  --manual \
  --compositions "Na1Cl1,K2S1" \
  --condition_lists "0.2,0.5,1.0" "0.0" \
  --level "level_3" \
  --output_parquet "test_prompts.parquet"

# Level 4 with spacegroups
python _utils/_generating/make_prompts.py \
  --manual \
  --compositions "TiO2" \
  --condition_lists "0.5" \
  --level "level_4" \
  --spacegroups "P42/mnm" \
  --output_parquet "prompts.parquet"
```

**Automatic Prompts (extract from existing datasets or dataframe):**
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

> For XRD studies and other complex conditioning, we use pre-constructed condition vectors rather than individual property columns (more efficient for large vectors).

#### Step 2: Generate CIFs

Use `_utils/_generating/generate_CIFs.py` with a config file:

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
  "gen_max_length": 1024,
  "do_sample": true,
  "num_return_sequences": 1,
  "num_repeats": 2 // total = num_return_sequences * num_repeats per prompt
}
```

#### Optional: Evaluate CIFs
**Evaluate** structural validity:
```bash
python _utils/_generating/evaluate_CIFs.py \
  --input_parquet "generated_cifs.parquet" \
  --num_workers 8 \
  --save_valid_parquet
```
> Quick check to verify models aren't outputting gibberish

#### Step 3: Post-process

**Post-process** generated CIFs to standard format:
```bash
python _utils/_generating/postprocess.py \
  --input_parquet "generated_cifs.parquet" \
  --output_parquet "processed_cifs.parquet" \
  --num_workers 4
```

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

Lower E-hull values indicate higher thermodynamic stability. In theory anything below 0.1 eV/atom is metastable, here if we account for MAE of the surrogate model (See MACE-MP-0 on Matbench Discovery), we can loosen threshold to 0.157 if desired.

#### 3. Other metrics
XRD, Bandgap, Density, or challenge set benchmark metrics can also be calculated for further property analysis. If wanted, find the associated scripts in [the \_metrics folder](_utils/_metrics/)

> **Environment Note**: ALIGNN-based scripts require the separate `alignn_env` environment due to dependency conflicts. Use `conda run -n alignn_env` for these tools.

## API

The CLI tools are available via a Containerised API.

### 1. Build the Container

```bash
docker build -t crystallm-api .
```

### 2. Run the Container

```bash
mkdir -p data outputs

docker run \
  -u $(id -u):$(id -g) \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -e HF_KEY="your_hf_token_here" \ 
  -e WANDB_KEY="your_wandb_key_here"
  --name crystallm-api \
  crystallm-api
```

### 3. API usage

```bash
# ==================== DATA PREPROCESSING ====================

# Deduplicate and filter data
curl -X POST "http://localhost:8000/preprocessing/deduplicate" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file": "/app/data/raw_data.parquet",
    "output_parquet": "/app/data/deduplicated_data.parquet",
    "property_columns": "[\"Bandgap (eV)\", \"Density (g/cm^3)\"]",
    "filter_na_columns": "[\"Bandgap (eV)\"]",
    "filter_zero_columns": "[\"Density (g/cm^3)\"]",
    "filter_negative_columns": "[\"Bandgap (eV)\"]"
  }'

# Clean and normalize CIF data
curl -X POST "http://localhost:8000/preprocessing/clean" \
  -H "Content-Type: application/json" \
  -d '{
    "input_parquet": "/app/data/deduplicated_data.parquet",
    "output_parquet": "/app/data/cleaned_data.parquet",
    "num_workers": 8,
    "property_columns": "[\"Bandgap (eV)\", \"Density (g/cm^3)\"]",
    "property1_normaliser": "power_log",
    "property2_normaliser": "linear"
  }'

# Save dataset to HuggingFace
curl -X POST "http://localhost:8000/preprocessing/save-dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "input_parquet": "/app/data/cleaned_data.parquet",
    "output_parquet": "your-dataset-name",
    "test_size": 0.1,
    "valid_size": 0.1,
    "HF_username": "your-username",
    "save_hub": true,
    "save_local": false
  }'

# ==================== TRAINING ====================

# Train model (single GPU)
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "config_file": "_config_files/cg_train/ft-slme/slme_ft-PKV.jsonc",
    "multi_gpu": false
  }'

# Train model (multi-GPU)
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "config_file": "_config_files/cg_train/ft-slme/slme_ft-PKV.jsonc",
    "multi_gpu": true,
    "nproc_per_node": 2
  }'

# ==================== GENERATION ====================

# Unconditional generation
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-2.0_base",
    "output_parquet": "/app/outputs/structures.parquet",
    "manual": true,
    "compositions": "LiFePO4,TiO2"
  }'

# Solar efficiency conditioning (SLME model)
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-2.0_SLME",
    "output_parquet": "/app/outputs/structures.parquet",
    "manual": true,
    "compositions": "CsPbI3",
    "condition_lists": ["0.8"]
  }'

# Bandgap conditioning (bandgap + stability)
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-2.0_bandgap",
    "output_parquet": "/app/outputs/structures.parquet",
    "manual": true,
    "compositions": "Si",
    "condition_lists": ["0.3", "0.0"]
  }'

# Generate from existing prompts file
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-2.0_bandgap",
    "output_parquet": "/app/outputs/structures.parquet",
    "manual": false,
    "input_parquet": "/app/data/prompts.parquet"
  }'

# ==================== ADVANCED GENERATION PIPELINE ====================

# Step 1: Create manual prompts (multi-property conditioning)
curl -X POST "http://localhost:8000/generate/make-prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "output_parquet": "/app/outputs/test_prompts.parquet",
    "manual": true,
    "compositions": "Na1Cl1,K2S1",
    "condition_lists": ["0.2,0.5,1.0", "0.0"],
    "level": "level_3"
  }'

# Create prompts with level 4 (including spacegroups)
curl -X POST "http://localhost:8000/generate/make-prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "output_parquet": "/app/outputs/prompts.parquet",
    "manual": true,
    "compositions": "TiO2",
    "condition_lists": ["0.5"],
    "level": "level_4",
    "spacegroups": "P42/mnm"
  }'

# Create automatic prompts from HF dataset
curl -X POST "http://localhost:8000/generate/make-prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "output_parquet": "/app/outputs/dataset_prompts.parquet",
    "manual": false,
    "automatic": true,
    "HF_dataset": "c-bone/mp_20_pxrd",
    "split": "test",
    "level": "level_2",
    "condition_columns": "Condition Vector"
  }'

# Step 2: Generate CIFs from config
curl -X POST "http://localhost:8000/generate/cifs" \
  -H "Content-Type: application/json" \
  -d '{
    "config_file": "_config_files/cg_eval/your_eval_config.jsonc"
  }'

# Optional: Evaluate CIF validity
curl -X POST "http://localhost:8000/generate/evaluate-cifs" \
  -H "Content-Type: application/json" \
  -d '{
    "input_parquet": "/app/outputs/generated_cifs.parquet",
    "num_workers": 8,
    "save_valid_parquet": true
  }'

# Step 3: Post-process CIFs
curl -X POST "http://localhost:8000/generate/postprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "input_parquet": "/app/outputs/generated_cifs.parquet",
    "output_parquet": "/app/outputs/processed_cifs.parquet",
    "num_workers": 4
  }'

# ==================== EVALUATION METRICS ====================

# Calculate VUN metrics (Validity, Uniqueness, Novelty)
curl -X POST "http://localhost:8000/metrics/vun" \
  -H "Content-Type: application/json" \
  -d '{
    "gen_data": "/app/outputs/generated_structures.parquet",
    "huggingface_dataset": "c-bone/mp_20",
    "output_csv": "/app/outputs/vun_results.csv",
    "num_workers": 8
  }'

# Calculate Energy Above Hull (stability)
curl -X POST "http://localhost:8000/metrics/ehull" \
  -H "Content-Type: application/json" \
  -d '{
    "post_parquet": "/app/outputs/postprocessed_structures.parquet",
    "output_parquet": "/app/outputs/stability_results.parquet",
    "num_workers": 4
  }'

# ==================== JOB MANAGEMENT ====================

# Check job status (replace JOB_ID with actual ID from response)
curl -X GET "http://localhost:8000/jobs/JOB_ID"

# List all jobs
curl -X GET "http://localhost:8000/jobs"

# API root info
curl -X GET "http://localhost:8000/"
```

## Studies

To have a look at how the experiments from the paper were carried out, the notebooks used to generate the results are available in the [Notebooks that start with X_](notebooks/)

Full information given for each, but most detail and example of a full end to end pipeline for Discovery can be found in the mp-20 notebook and recovery SLME notebook

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please contact cyprien.bone.24@ucl.ac.uk or raise an issue on the github page
