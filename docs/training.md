# Training, Generating & Evaluating from Scratch

!!! tip "Run it in a notebook"
    [`T1_finetune_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T1_finetune_density_example.ipynb) runs this pipeline end to end on a density dataset: prepare the data, finetune, push to the Hub, register, generate. [`T6_SLME.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T6_SLME.ipynb) does the same for a photovoltaic efficiency target and screens the output.

Complete pipeline for training your own models from data preprocessing to evaluation. All training and generation parameters and options are defined in [`_args.py`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_args.py). Training & generating should be done via configuration files (`.jsonc` format) which specify all necessary parameters.

> Maintained notebook workflow: [`notebooks/T5_XRD_continuous.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T5_XRD_continuous.ipynb) covers raw-scan to continuous-profile conversion and conditioned generation with the cXRD model. XRD-model training (including the CHILI-100K KD pipeline) lives in [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph).

## Data Processing Pipeline

### Step 1: Data Preparation (**Required**)

Input data should be a pandas DataFrame saved as Parquet file. To train a model you should save a dataframe to a parquet file which contains:

**Required columns:**

* `Database`: Source database name
* `Reduced Formula`: Standard reduced chemical formula
* `CIF`: Crystallographic structure in CIF format

**For structure recovery benchmarks**

* `Material ID`: Database identifier required for structure recovery benchmarks

**Optional columns:**

* `<Property Columns>`: Target properties (e.g., "Bandgap (eV)", "Density (g/cm^3)")
* `condition_vector`: Pre-computed condition vectors (for XRD studies)

### Step 2: Deduplication and Filtering (Optional)

**Script:** `_utils/_preprocessing/deduplicate.py` - Removes duplicate structures and filters invalid entries based on chemical formula and space group, keeping the structure with lowest volume per formula unit.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/deduplicate.py \
  --input_parquet /path/to/raw_data.parquet \
  --output_parquet /path/to/deduplicated_data.parquet \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --filter_na_columns "['Bandgap (eV)']" \
  --filter_zero_columns "['Density (g/cm^3)']" \
  --filter_negative_columns "['Bandgap (eV)']"
```

**Key arguments:**

* `--filter_na_columns`: Remove entries with N/A or NaN values
* `--filter_zero_columns`: Remove entries with zero values
* `--filter_negative_columns`: Remove entries with negative values

</details>

### Step 3: CIF Cleaning and Normalization (**Required**)

**Script:** `_utils/_preprocessing/cleaning.py` - Standardizes CIF format and normalizes properties for stable training. Adds atomic property blocks, rounds numerical values, and applies variable brackets.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/cleaning.py \
  --input_parquet /path/to/deduplicated_data.parquet \
  --output_parquet /path/to/cleaned_data.parquet \
  --num_workers 8 \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --property1_normaliser "power_log" \
  --property2_normaliser "linear"
```

> Tip: Keep a note somewhere of the lowest and highest property values for each property, so that later when you have a particular property target you can easily normalize it to the format the model expects.

**Key arguments:**

* `--property1_normaliser` / `--property2_normaliser`: Normalization methods (`linear`, `power_log`, `signed_log`, `log10`, `None`)
* `--make_disordered_ordered`: Convert disordered structures to ordered ones
* `--num_workers`: Number of parallel workers for processing

**Normalization methods:**

* `linear`: Simple min-max scaling to [0,1] range
* `power_log`: Power transformation ($\beta$=0.8) followed by logarithmic scaling for skewed distributions
* `signed_log`: Signed logarithmic transformation for handling negative values
* `log10`: Base-10 logarithmic scaling for properties spanning multiple orders of magnitude
* `None`: No normalization applied

</details>

### Step 4: Dataset Upload to HuggingFace (**Required**)

**Script:** `_utils/_preprocessing/save_dataset_to_hf.py` - Converts to HuggingFace format with train/validation/test splits and uploads to HF Hub.

> Important: You need to make sure that the data trained on has been passed through CIF cleaning, a quick way to make sure is check whether the CIFs in your dataframe contain brackets. If they do then text should be ready for training.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/save_dataset_to_hf.py \
  --input_parquet /path/to/processed_data.parquet \
  --output_parquet "your-dataset-name" \
  --test_size 0.1 \
  --valid_size 0.1 \
  --HF_username "your-username" \
  --save_hub \
  --save_local
```

**Key arguments:**

* `--duplicates`: Prevents data leakage by splitting on Material ID (optional)
* `--test_size` / `--valid_size`: Split ratios (set both to 0.0 for training-only)
* `--save_hub` / `--save_local`: Upload to HF Hub and/or save locally (specify at least one)

</details>




## Training

All training should be done via configuration files (`.jsonc` format). These files specify model architecture, hyperparameters, data paths, and training settings. See example configs in `_config_files/training/` and review [`_args.py`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_args.py). for all available parameters.

> The `Muon` optimiser is now available for training, see [this blog post](https://kellerjordan.github.io/posts/muon/) for details. Importantly, you cannot use deepspeed when using muon. Simply do not feed a deepspeed configuration file and it will work fine (multi-GPU training still supported). Muon speeds up and stabilises training without any performance trade-offs (did some internal checks).

<details>
<summary>Base model training CLI example</summary>

### Base Model Pretraining

Train the unconditional base models from scratch:

```bash
python _train.py --config _config_files/training/unconditional/lematerial-small.jsonc
```

**Multi-GPU Training:**

```bash
torchrun --nproc_per_node=2 _train.py --config your_config.jsonc
```

</details>

<br>

<details>
<summary>Conditional finetuning CLI example</summary>

### Conditional Fine-tuning

Fine-tune pretrained base models for property-guided generation:

**Single GPU:**

```bash
python _train.py --config _config_files/training/conditional/ft-slme/slme_ft-prefix-opt.jsonc
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=2 _train.py --config _config_files/training/conditional/ft-slme/slme_ft-prefix-opt.jsonc
```

Loads pretrained weights as starting point (or trains from scratch), adds conditional architecture layers, and uses split optimizer with different learning rates for conditioning vs base layers.

</details>




## Advanced Generation Pipeline

### Step 1: Create Prompts

**Script:** `_utils/_generating/make_prompts.py` - Generate input prompts for conditional generation with different levels of structural information.

<details>
<summary>Examples of Prompt Construction and Args</summary>

**Manual Prompts:**

```bash
python _utils/_generating/make_prompts.py \
  --manual \
  --compositions "Na1Cl1,K2S1" \
  --condition_lists "0.2,0.0" "0.5,0.0" \
  --level "level_3" \
  --output_parquet "test_prompts.parquet"
```

**Automatic Prompts from Dataset:**

```bash
python _utils/_generating/make_prompts.py \
  --automatic \
  --HF_dataset "c-bone/mp_20_pxrd" \
  --split "test" \
  --level "level_2" \
  --condition_columns "Condition Vector" \
  --output_parquet "dataset_prompts.parquet"
```

**Prompt levels `--level`:**

* `level_1`: Minimal (unconditional generation)
* `level_2`: Composition only (default)
* `level_3`: Composition + atomic properties
* `level_4`: Up to space group information

**Composition-Condition Pairing modes `--mode`:**

Each quoted string is a **complete condition vector** (comma-separated property values).

* `cartesian` (default): All conditions applied to all compositions
* `paired`: 1:1 mapping - must have same count of conditions and compositions
* `broadcast`: Single condition applied to all compositions

</details>

### Step 2: Generate CIFs

**Script:** `_utils/_generating/generate_cifs.py` - Generate crystal structures from prompts using trained models.

<details>
<summary>Examples of CIF generation and Args</summary>

```bash
python _utils/_generating/generate_cifs.py \
  --config _config_files/generation/conditional/slme/slme-PKV-opt_eval.jsonc
```

> You can generate with arguments from the CLI, but it's easier to use the config file. You can find a lot of examples in [`_config_files/generation`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_config_files/generation)

**Key generation settings:**

* **Temperature:** Controls randomness (default ~1.0, higher is more exploratory but higher chance of gibberish)
* **Top-p/Top-k:** Sampling parameters (typical: 0.95, 50)
* **scoring_mode:** if set to `None` and `target_valid_cifs = 0`, then we generate `max_return_attempts * num_return_sequences` CIFs per Prompt/Condition pair without validation. If set to `None` and `target_valid_cifs > 0`, then we validate generated CIFs and stop once that many valid CIFs are found, without ranking. If set to `LOGP`, we validate and rank using a perplexity based scoring method. `PEARSON` (continuous-XRD models only) validates and ranks by agreement between each candidate's simulated diffraction pattern and the input scan; a continuous-XRD `--search_zs` run defaults to `PEARSON` when no mode is given.
* **num_return_sequences:** Batch size for generation (adjust for GPU mem.)
* **max_return_attempts:** In raw mode, total generation for each Prompt/Condition pair = `max_return_attempts * num_return_sequences`. In validation-targeted modes, generation stops when `target_valid_cifs` valid CIFs are found or `max_return_attempts` is reached.

</details>

### Step 3: Post-process

**Script:** `_utils/_generating/postprocess.py` - Clean and validate generated CIF structures.

<details>
<summary>Examples of postprocessing and Args</summary>

```bash
python _utils/_generating/postprocess.py \
  --input_parquet "generated_cifs.parquet" \
  --output_parquet "processed_cifs.parquet" \
  --num_workers 4
```

Convcerts LLM outputs to standard Pymatgen style CIF format.

</details>




## Evaluation

### VUN Metrics (Validity, Uniqueness, Novelty)

**Script:** `_utils/_scoring/vun_metrics.py` - Essential metrics for assessing generation quality using structural analysis.

**Required:** Structures must be post-processed with Reduced Formulas column included

**Metrics computed:**

* **Validity**: Structures with correct spacegroup, reasonable bond lengths, and consistent atom multiplicities
* **Uniqueness**: Distinct structures within the generated set (using BAWL hashing)
* **Novelty**: Structures not present in the reference dataset
* **Compositional Novelty**: Reduced Formula not present in reference dataset

<details>
<summary>Example Usage</summary>

```bash
python _utils/_scoring/vun_metrics.py \
  --input_parquet generated_structures_processed.parquet \
  --huggingface_dataset "c-bone/mp_20" \
  --output_parquet vun_results.parquet \
  --num_workers 8
```

We can optionally set the `--check_comp_novelty` flag, which adds an `is_comp_novel` boolean column to the metrics dataframe.

</details>

### Energy Above Hull (Stability)

**Script:** `_utils/_scoring/mace_ehull.py` - Calculate thermodynamic stability using MACE energy predictions. See the [MACE paper](https://arxiv.org/abs/2206.07697) for details on the surrogate model.

> To calculate E_hull First, total energies are computed using the MACE-MP default calculator, predicted energies are then processed using the *MaterialsProject2020Compatibility* scheme to ensure consistency between GGA and GGA+U calculations. The surrogate energy predictions are compared to formation energies of known materials from the MP dataset and used to construct a convex hull. The energy above the convex hull (E_hull) quantifies thermodynamic stability by comparing a material's formation energy to competing phases.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_scoring/mace_ehull.py \
  --post_parquet postprocessed_structures.parquet \
  --output_parquet stability_results.parquet \
  --num_workers 4
```

Lower E_hull values indicate higher thermodynamic stability. Structures with E_hull < 0.1 eV/atom are typically considered experimentally synthesizable. (We can extend to 0.157 eV/atom if we want to account for MAE in energy predictions of this MACE model)

</details>

### Additional Metrics

XRD or density property metrics, VUN, and stability metrics are available in `_utils/_scoring/`.
