# CrystaLLMv2

## Description

**CrystaLLMv2** is a Transformer-based model for generating crystalline structures as Crystallographic Information Files (CIFs). This project provides a complete codebase for both conditional (property-driven) and non-conditional GPT-style training, generation, and evaluation.

The models are based on the [CrystaLLM Paper](https://www.nature.com/articles/s41467-024-54639-7)

It leverages a few tools:
- **Hugging Face**: For dataset hosting, model sharing, and tokenizer access.
- **Weights & Biases**: For experiment tracking and visualization.
- **CodeCarbon**: For tracking the carbon footprint of training runs.
- **DeepSpeed**: For efficient, distributed training on multi-GPU systems.

## Table of Contents
- [CrystaLLMv2](#crystallmv2)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Configuration](#configuration)
    - [API Keys](#api-keys)
    - [Training and evaluation Parameters](#training-and-evaluation-parameters)
  - [Model Types:](#model-types)
    - [1. Unconditional CrystaLLM](#1-unconditional-crystallm)
    - [2. Conditional Models](#2-conditional-models)
      - [a. PKV-GPT](#a-pkv-gpt)
      - [b. Prepend-GPT](#b-prepend-gpt)
      - [c. Slider-GPT](#c-slider-gpt)
      - [d. Raw-GPT](#d-raw-gpt)
  - [Usage](#usage)
    - [Data Processing](#data-processing)
    - [Training the Model](#training-the-model)
    - [Crafting Prompts](#crafting-prompts)
      - [Unconditional Prompts](#unconditional-prompts)
      - [Conditional Prompts](#conditional-prompts)
    - [Generating Structures](#generating-structures)
  - [Evaluation](#evaluation)
    - [Structural Metrics](#structural-metrics)
    - [Stability (MACE E-hull)](#stability-mace-e-hull)
    - [Automated Pipelines](#automated-pipelines)
  - [Example uses](#example-uses)
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
    git clone https://github.com/C-Bone-UCL/CrystaLLMv2_CG.git
    cd CrystaLLMv2_CG
    ```

2.  **Create a virtual environment for the main package:**
    ```bash
    conda create -n crystallmv2_env python=3.10
    conda activate crystallmv2_env
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


### Training and evaluation Parameters
All training arguments are defined in `_args.py` and can be set in two ways:

1.  **Config File**: Create a `.jsonc` file and pass it with the `--config` flag. This is the recommended method.  Can be used with `_train.py` and `generate_CIFs.py` files.
    ```bash
    python _train.py --config _config_files/cg_train/test.jsonc
    ```
2.  **Command-Line Arguments**: Override any config setting by passing arguments directly.
    ```bash
    python _train.py --n_layer 8 --n_embd 512 --learning_rate 5e-4
    ```

## Model Types:

CrystaLLMv2 supports one unconditional and four conditional model architectures, allowing for both standard and property-driven generation. The desired model can be selected during training using the `--activate_conditionality` flag.

### 1. Unconditional CrystaLLM
Do not set the `--activate_conditionality` flag.

This is the standard CrystaLLM/GPT-2 architecture for generative tasks. It learns the underlying patterns and grammar of CIF files without any explicit property guidance.

### 2. Conditional Models

#### a. PKV-GPT 
(`--activate_conditionality="PKV"`)

PKV (Past Key-Value) conditioning injects property information directly into the attention mechanism's past key-values. This allows the model to steer generation based on the desired properties by concatenating the conditional embeddings at each layer of the transformer in the attention mechanism. It is a 'strong' conditioning method and balances expressivity of conditioning and straightforward implementation.

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

### Data Processing
Before training, raw crystallographic data must be processed to ensure quality and consistency. The preprocessing pipeline involves deduplication, cleaning, normalization, and formatting for use with Hugging Face.


**1. Gathering Data**
To gather data in the format readable by the package, you must creata a [pandas dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) and save it to a [.parquet format](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format). This format is good for efficiently handling large databases and is how huggingface datasets are stored.

The dataframe **must** have the following columns:
- `Database`: Name of database that CIF file comes from
- `Reduced Formula`: Standard reduced formula of the material (is used in evaluation metrics)
- `CIF`: The CIF of the material that you want the model to train on

*Optionally*, we can include:
- `Material ID`: If the material has an ID in the database, it is good to track the Material ID for each structure to ensure traceability
- `<Property Columns>`: If there is a conditional generation task, the property associated to each of the CIFs should be stored in the column of that row. (Change name to whatever the property is)

**2. Deduplication and Filtering**

The first step is to remove duplicate entries and filter the dataset. The `_deduplicate.py` script identifies unique structures based on their chemical formula and space group, keeping the one with the lowest volume per formula unit. It can also filter out entries with missing or invalid property values.

```bash
python _utils/_preprocessing/_deduplicate.py \
  path/to/your/raw_data.parquet \
  --out path/to/your/deduplicated_data.parquet \
  ### Optionally if doing conditional generation ###
  --property_columns "['Bandgap (eV)', 'energy_above_hull']" \
  --filter_na_columns "['Bandgap (eV)']"
```
> Note: For the file paths, make sure you include a full path rather than just a file name. 

**3. Cleaning and Normalization**

Next, the CIF strings are cleaned and standardized using `_cleaning.py`. This script performs several augmentations:
- Standardizes the CIF format for the model.
- Adds a block with atomic properties (e.g., electronegativity).
- Rounds numerical values to a consistent number of decimal places.
- For conditional training, it normalizes the target property values to a [0, 1] range using methods like linear or power-log scaling.

```bash
python _utils/_preprocessing/_cleaning.py \
  path/to/your/deduplicated_data.parquet \
  --out path/to/your/cleaned_data.parquet \
  --workers 8
  ### Optionally if doing conditional generation ###
  --property_columns "['Bandgap (eV)', 'energy_above_hull']" \
  --property1_normaliser "power_log" \
  --property2_normaliser "linear" \
```

**4. Saving to Hugging Face**

- `_save_dataset_to_HF.py`: Converts the final DataFrame into a `DatasetDict` (with train, validation, and test splits) and pushes it to the Hub.
- `_save_tokenizer_to_HF.py`: Saves and uploads the custom CIF tokenizer.

```bash
# Save the dataset to the Hub
python _utils/_preprocessing/_save_dataset_to_HF.py \
  path/to/your/cleaned_data.parquet \
  --out "local-dataset-dir/your-dataset-name" \
  --hf_key_path "API_keys.jsonc" \
  --HF_username "your-hf-username" \
  --save_hub \
  ### You can optionally save locally ###
  --save_local


#####################
# Optional
#####################
# Save the tokenizer to the Hub
# The tokenizer is already present in the repo and works as is
# But if you make changes to it this is how you can push it to hub
python _utils/_preprocessing/_save_tokenizer_to_HF.py \
  --path "HF-cif-tokenizer" \
  --hub_path "your-hf-username/cif-tokenizer" \
  --hf_key_path "API_keys.jsonc" \
  --push_to_hub
```

> See the [Data Preprocessing Notebook](notebooks/A_Preprocess_data.ipynb).


### Training the Model

**If training on CPU or 1 GPU, use python _train.py**

**If training on multiple GPUs, use torchrun --nproc_per_node=$N$ _train.py --config ''**
**And make sure the deepspeed path argument is set to the deepspeed config default path**

**Standard (Non-conditional) Training:**
```bash
python _train.py --config _config_files/cg_train/test.jsonc
```

**Conditional Training:**
To train a model that conditions on properties, activate conditionality and specify the relevant columns from your dataset.
```bash
python _train.py \
  --config _config_files/cg_train/your_config.jsonc \
  --activate_conditionality="PKV" \
  --condition_columns="['Bandgap (eV)']" \
  --n_prefix_tokens=2
```

> See the [Training Notebook](notebooks/B_Train_model.ipynb).

### Crafting Prompts
You can generate prompts for both unconditional and conditional generation using the provided scripts. The output is a `.parquet` file that can be used as input for the generation scripts.


#### Unconditional Prompts

Use the `_utils/_evaluation_og/make_prompts.py` script to create prompts from a chemical formula or a Hugging Face dataset.

**1. From a Chemical Formula:**
```bash
python _utils/_evaluation_og/make_prompts.py \
  --material_composition "Na1Cl1,K2S1" \
  --out "prompts/unconditional_prompts.parquet"
```

**2. From a Dataset (for Benchmarking):**
Extract prompts from a test set on Hugging Face to evaluate performance on known structures.
```bash
python _utils/_evaluation_og/make_prompts.py \
  --HF_dataset "your-hf-username/your-dataset" \
  --split "test" \
  --out "prompts/benchmark_prompts.parquet"
```
> See an example in the [Unconditional Evaluation Notebook](notebooks/C1_Evaluation_og.ipynb).


#### Conditional Prompts
Use the `_utils/_evaluation_conditional/make_prompts.py` script to create prompts with specific property conditions. The script will generate prompts for every combination of the provided compositions and conditions.
```bash
python _utils/_evaluation_conditional/make_prompts.py \
  --material_composition "Al2O3" \
  --bg_den_conditions "5.5, 6.0" \
  --ehull_conditions "0.1, 0.05" \
  --out "prompts/conditional_prompts.parquet"
```
> See an example in the [Conditional Evaluation Notebook](notebooks/C2_Evaluation_CG.ipynb).


### Generating Structures

**Unconditional Generation:**
```bash
python _utils/_evaluation_og/generate_CIFs.py \
  --config _config_files/og_eval/your_config.jsonc \
  --prompt="<bos>\ndata_[Na1Cl1]\n" \
  --num_return_sequences=5
```

> See an example in the [Unconditional Evaluation Notebook](notebooks/C1_Evaluation_og.ipynb).


**Conditional Generation:**
Provide a `condition_vector` to guide the generation towards a specific property value.
```bash
python _utils/_evaluation_conditional/generate_CIFs.py \
  --model_ckpt_dir="model_ckpts/my_conditional_model" \
  --prompt="<bos>\ndata_[Al2O3]\n" \
  --condition_vector="5.5" \
  --num_return_sequences=5
```

> See an example in the [Conditional Evaluation Notebook](notebooks/C2_Evaluation_CG.ipynb).


## Evaluation

### Structural Metrics
After generation, CIFs must be evaluated for physical and chemical correctness. The `evaluate_CIFs.py` script assesses structural validity.

**A Structure is Valid if:**
- The declared spacegroup is consistent with the generated structure
- The generated bond lengths are reasonable, and
- The declared atom site multiplicity is consistent with the cell composition.

**Run Evaluation:**
```bash
python _utils/_evaluation_og/evaluate_CIFs.py \
  --gen_cifs path/to/generated.parquet \
  --metrics_out path/to/metrics.parquet \
  --workers 8
```
> See an example in the [Conditional Evaluation Notebook](notebooks/C2_Evaluation_CG.ipynb).

Any other postprocessing we need to postprocess files (so they are back to standard CIF files format)

### Stability (MACE E-hull)
A critical metric for material stability is the energy above the convex hull (E-hull). Lower values (closer to zero) indicate higher stability. We use the MACE model for fast and accurate energy predictions.

The `mace_ehull.py` script calculates the E-hull for a set of CIFs. It requires a parquet file of valid, post-processed structures.

**Run Stability Calculation:**
```bash
# Ensure you are in the crystallmv2_env conda environment
python _utils/_evaluation_conditional/mace_ehull.py \
  --post_parquet path/to/postprocessed.parquet \
  --output_parquet path/to/stability_results.parquet \
  --n_jobs 4
```
> See an example in the [Conditional Evaluation Notebook](notebooks/C2_Evaluation_CG.ipynb).


### Automated Pipelines
For streamlined workflows, use the provided pipeline scripts:

- **Conditional Pipeline (`CG_pipeline.py`):** Automates conditional generation, property evaluation, and stability analysis.
  ```bash
  python _utils/_evaluation_conditional/CG_pipeline.py
  ```
  > See the [Conditional Metrics Notebook](notebooks/D2_Metrics_CG.ipynb).
- **Unconditional Benchmark Pipeline (`bench_pipeline.py`):** Automates generation, post-processing, and benchmarking against standard datasets.
  ```bash
  python _utils/_evaluation_og/bench_pipeline.py
  ```
  > See the [Unconditional Evaluation Notebook](notebooks/C1_Evaluation_og.ipynb).


> See the respective scripts for more details on the pipeline.

## Example uses

To have a look at how the experiments from the paper were carried out, the notebooks used to generate the results are available in the [Notebooks that start with X_](notebooks/)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact cyprien.bone.24@ucl.ac.uk or raise an issue on the