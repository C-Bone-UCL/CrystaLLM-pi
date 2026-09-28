# Quick Start

!!! tip "Run it in a notebook"
    [`T2_load_and_generate.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T2_load_and_generate.ipynb) walks through everything on this page with a released Hub model.

Use with pre-trained models from HuggingFace Hub for direct crystal structure generation. The `_load_and_generate.py` script handles downloading models and generating valid CIF structures with desired properties.

## How It Works

The script automatically:

1. **Downloads models** from HuggingFace Hub (cached locally after first use).
2. **Normalizes property values**: provide standard unit values (e.g., bandgap in eV, density in g/cm3).
3. **Creates prompts** at different detail levels using explicitly mapped Z-values or automated Z-searches (Z being stoichiometry number).
4. **Generates structures** using the appropriate conditional model architecture.
5. **Validates & Ranks** outputs based on structural integrity and optional perplexity (`LOGP`) scoring.

Each model can be used by providing a list of reduced formulas (`--reduced_formula_list`) paired with either explicit stoichiometric scaling factors (`--z_list`) or an automated discovery sweep (`--search_zs`). XRD-conditioned models take scan files via `--xrd_files`: the `Mattergen-XRD` and `Chili100K-XRD` models use a top-20 pre-picked-peak pipeline and can also run without `--xrd_files` using missing conditioning values.

## Generation Examples

Expand below for a list of how you can generate with the models using the script

<details markdown>
<summary>Examples</summary>

<br>

> **Note**: Properties (Conditions, Spacegroups, XRD files, Z values) map strictly 1:1 to the reduced formulas provided in `--reduced_formula_list`.
> 
> **Outputs**: Outputs can either be saved as a dataframe in a `.parquet` using the `--output_parquet` flag, or as individual CIFs in a directory using the `--output_cif_dir` flag.

**Recovery: known composition and space group, no property**

Generate Ti2O4 (TiO2 with Z=2) in space group P4_2/mnm. The model samples up to 2 batches of 5 and keeps 5 valid structures.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_ft_alex_mp_20-text" \
    --reduced_formula_list "TiO2" \
    --z_list "2" \
    --spacegroups "P4_2/mnm" \
    --level level_4 \
    --num_return_sequences 5 \
    --max_return_attempts 2 \
    --target_valid_cifs 5 \
    --output_cif_dir recovery_cifs
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

Automatically search over Z=1, 2, 3, 4 and 6 to find valid structures. Because `scoring_mode` is None, the worker stops the search and returns a structure once it satisfies `--target_valid_cifs`.

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

**Discovery: property target only**

Generate structures with no composition given, asking the SLME model for a photovoltaic efficiency of 25%. The model chooses the elements, stoichiometry and space group, sampling up to 2 batches of 5 and keeping 5 valid structures.

```bash
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_SLME" \
    --condition_lists "25.0" \
    --level level_1 \
    --num_return_sequences 5 \
    --max_return_attempts 2 \
    --target_valid_cifs 5 \
    --output_cif_dir discovery_cifs
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
    Combine `--search_zs` with `--target_valid_cifs X` and it will loop through Z until it finds a valid CIF. If on top of that you add the `LOGP` scoring, it will generate for each Z. Across all the Zs that gave a valid CIF, it returns the model's single most confident prediction for the reduced formula.

**Perplexity Scoring (`--scoring_mode "LOGP"`)**

* For each generation which passes basic chemical validity checks, we compute transition scores for the token sequence to the perplexity score. Lower perplexity values indicate higher model confidence in the generated sequence according to its learned probability distribution. [See Blog Post for more info](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-21-intrinsic-evaluation-metrics/interpreting-perplexity-scores)

<br>
