# Virtual Crystal Generation (Post-processing)

After generating ordered CIF structures, you can convert them to disordered virtual crystals using the [`virtualiser`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_virtualiser/virtualiser.py) utility. This replaces specified element pairs with fractional occupancies at shared sites and promotes the structure to its higher-symmetry parent with spglib. Useful for comparing against experimental diffraction data or estimating a disordered structure candidate.

<details markdown>
<summary>Example Usage and Config</summary>

**Config file (YAML):**

```yaml
symprec: 0.003
angle_tolerance: 0.5
virtual_pairs:
  - [Mg, Zn]
```

**Example:**

```bash
# Generate an ordered structure
python _load_and_generate.py \
    --hf_model_path "c-bone/CrystaLLM-pi_base" \
    --reduced_formula_list "Mg3ZnO4" \
    --z_list "1" \
    --num_return_sequences 10 \
    --scoring_mode "LOGP" \
    --target_valid_cifs 1 \
    --output_cif_dir outputs/

# Virtualise the result
python _utils/_virtualiser/virtualiser.py \
    --in outputs/Mg3ZnO4.cif \
    --config config.yaml \
    --out outputs/Mg3ZnO4_virtual.cif
```

</details>

<br>
