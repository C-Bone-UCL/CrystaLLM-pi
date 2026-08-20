# Quick Start

!!! tip "Run it in a notebook"
    [`T2_load_and_generate.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T2_load_and_generate.ipynb) walks through everything on this page with a released Hub model. For the XRD side, [`T4_XRD_continuous.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T4_XRD_continuous.ipynb) recovers a structure from a raw experimental scan.

Use with pre-trained models from HuggingFace Hub for direct crystal structure generation. The `_load_and_generate.py` script handles downloading models and generating valid CIF structures with desired properties.

## How It Works

The script automatically:

1. **Downloads models** from HuggingFace Hub (cached locally after first use).
2. **Normalizes property values** - provide standard unit values (e.g., bandgap in eV, density in g/cm3).
3. **Creates prompts** at different detail levels using explicitly mapped Z-values or automated Z-searches (Z being stoichiometry number).
4. **Generates structures** using the appropriate conditional model architecture.
5. **Validates & Ranks** outputs based on structural integrity and optional perplexity (`LOGP`) scoring, or Pearson correlation of generated to input XRD profiles (`PEARSON`), which is available for the continuous-XRD models only.

Each model can be used by providing a list of reduced formulas (`--reduced_formula_list`) paired with either explicit stoichiometric scaling factors (`--z_list`) or an automated discovery sweep (`--search_zs`). XRD-conditioned models take raw scan files via `--xrd_files`: the continuous-XRD models (`Chili100K-cXRD` or `alex_mp_20-cXRD`) convert full diffractometer scans automatically, while the older `Mattergen-XRD` and `Chili100K-XRD` models use a top-20 pre-picked-peak pipeline and can also run without `--xrd_files` using missing conditioning values. cXRD-model *training* happens in [CrystaLLM-graph](https://github.com/C-Bone-UCL/CrystaLLM-graph), this is because they also incorproate knowledge distillation steps.

## Generation Examples

Expand below for a list of how you can generate with the models using the script

<details markdown>
<summary>Examples</summary>

<br>

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

**Raw Experimental Scan Conditioning (Continuous XRD)**

Recover a structure from a raw powder pattern, no peak picking needed. The scan is converted to the model's continuous `[Q, I]` profile automatically.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_Chili100K-cXRD" \
    --reduced_formula_list "TiO2" \
    --xrd_files tests/fixtures/Rutile-TiO2-unproc.txt \
    --xrd_wavelength 1.54059 \
    --level level_3 \
    --target_valid_cifs 1 \
    --output_cif_dir outputs/cxrd_demo
```

**XRD-Fit Ranked Z-Search (Continuous XRD, recommended)**

Sweep Z values and rank every valid candidate by agreement between its simulated pattern and the input scan (Pearson correlation on the model's Q grid). This is the default when `--search_zs` is used with a continuous-XRD model and no `--scoring_mode` is given.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_alex_mp_20-cXRD" \
    --reduced_formula_list "TiO2" \
    --search_zs \
    --scoring_mode "PEARSON" \
    --xrd_files tests/fixtures/Rutile-TiO2-unproc.txt \
    --xrd_wavelength 1.54059 \
    --level level_3 \
    --num_return_sequences 10 \
    --output_parquet xrd_fit_ranked.parquet
```

**Mapped Lists (Bandgap Conditioning)**

Provide parallel lists to generate multiple specific structures at once. Each condition vector (bandgap, E_hull) directly corresponds to the respective formula.

```bash
# Maps: (TiO2, Z=2, bg=1.8, E_hull=0.0) and (SiO2, Z=4, bg=5.0, E_hull=0.0)
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_bandgap" \
    --reduced_formula_list "TiO2,SiO2" \
    --z_list "2,4" \
    --condition_lists "1.8,0.0" "5.0,0.0" \
    --level level_3 \
    --num_return_sequences 5 \
    --output_parquet semiconductors.parquet
```

**Early-Stopping Z-Search (Density Conditioning)**

Automatically search over Z=1, 2, 3, 4 and 6 to find valid structures. Because `scoring_mode` is None, the worker stops the search and return a structure once it satisfies the `--target_valid_cifs`.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_density" \
    --reduced_formula_list "SiO2" \
    --search_zs \
    --condition_lists "2.143,0.0" \
    --level level_2 \
    --num_return_sequences 5 \
    --target_valid_cifs 1 \
    --output_parquet fast_discovery.parquet
```

**Ranked Z-Search (LOGP)**

Search across all Z values (1, 2, 3, 4 and 6), generate batches for all of them, and then rank the valid outputs using LOGP perplexity to find the most theoretically stable structures.

```bash
python _load_and_generate.py \
  --hf_model_path "c-bone/CrystaLLM-pi_base" \
  --reduced_formula_list "SiO2,TiO2" \
  --search_zs \
  --scoring_mode "LOGP" \
  --target_valid_cifs 3 \
  --num_return_sequences 10 \
  --output_parquet reduced_formula_best.parquet
```

**Solar Efficiency (Level 1)**

Unconditionally generate with a high photovoltaic efficiency.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_SLME" \
    --condition_lists "25.0" \
    --level level_1 \
    --num_return_sequences 5 \
    --target_valid_cifs 0 \
    --output_parquet solar_screening.parquet
```

</details>

## Configuration Options

**Prompt levels `--level`:**

* `level_1`: Minimal (unconditional/property only generation)
* `level_2`: Composition only (default)
* `level_3`: Composition + atomic properties
* `level_4`: Composition + spacegroup

**Stoichiometry Control:**

* `--z_list "X,Y"`: Provide a comma-separated list of exact stoichiometric multipliers mapping 1:1 to your reduced formulas.
* `--search_zs`: Trigger an automated sweep over Z=1, 2, 3, 4 and 6 for each formula.

!!! tip
    Combine `--search_zs` with `--target_valid_cifs X` and it will loop through Z until it finds a valid CIF. If on top of that you add the `LOGP` or `PEARSON` scoring, itll generate for each Z. For all the Zs with a valid CIFs, it will return the models single most confident prediction for the reduced formula.

**Perplexity Scoring (`--scoring_mode "LOGP"`)**

* For each generation which passes basic chemical validity checks, we compute transition scores for the token sequence to the perplexity score. Lower perplexity values indicate higher model confidence in the generated sequence according to its learned probability distribution. [See Blog Post for more info](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-21-intrinsic-evaluation-metrics/interpreting-perplexity-scores)

**XRD-Fit Scoring (`--scoring_mode "PEARSON"`)**

* `PEARSON` is supported only by the continuous-XRD models `c-bone/CrystaLLM-pi_Chili100K-cXRD` and `c-bone/CrystaLLM-pi_alex_mp_20-cXRD`. Ranking requires the per-row `(1000, 2)` `[Q, I]` conditioning profile, so passing `PEARSON` to any other model raises an error.

* For each generated structure that passes the validity checks, its powder diffraction pattern is simulated with pymatgen, broadened onto the model's 1000-point Q grid using the same peak shape as the conditioning profile, and compared with the input scan using the Pearson correlation coefficient. Higher Pearson correlation is better and same for with `LOGP`, `PEARSON` requires `--target_valid_cifs` to be greater than 0. Its also the default for the cXRD models.

Generally, the perplexity scoring has given me better RMSD to input target, and the pearson is new but seems to match to target XRDs better since we have moved to the cXRD models.

<br>
