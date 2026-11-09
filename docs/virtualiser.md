# Virtual Crystal Generation (Post-processing)

After generating ordered CIF structures, you can convert them to disordered crystals using the [`virtualiser`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/_utils/_virtualiser/virtualiser.py) script. This replaces specified element groups with fractional occupancies at shared sites and promotes the structure to its higher-symmetry parent with spglib. Each `virtual_pairs` entry lists two or more elements to merge onto one shared sublattice.

## Example Usage and Config

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

## Scanning symmetry tolerances

Generated structures are often slightly distorted, so a single tight tolerance can miss the parent space group. `symprec` (in Angstrom) and `angle_tolerance` (in degrees) can each be a list. The virtualiser tries every combination and keeps the highest-symmetry result, at the smallest tolerances that reach it.

```yaml
symprec: [0.003, 0.01, 0.03, 0.1, 0.2, 0.3]
angle_tolerance: [0.5, 5.0]
virtual_pairs:
  - [Mg, Zn]
```

XRD-conditioned Mg3ZnO4 structures from `CrystaLLM-pi_Mattergen-XRD`, for example, include L1<sub>2</sub>-ordered Pm-3m cells at Z=1, which reach the Fm-3m rock-salt parent at `symprec` 0.003 Angstrom, and slightly distorted I4/mmm cells at Z=2, which need 0.1 Angstrom. Check which tolerance was kept, because very coarse values can merge sites that are genuinely distinct.

## Batch mode

Pass a directory to `--in` to virtualise every `*.cif` in it. `--out` is then an output directory, and the virtualiser writes a `virtualise_summary.csv` there, or to `--summary`, with the space group, kept tolerances and lattice parameters of each structure.

```bash
python _utils/_virtualiser/virtualiser.py \
    --in outputs/ \
    --config config.yaml \
    --out outputs_virtual/
```

From Python, `promote_symmetry_scan(struct, symprecs, angle_tols)` returns the refined structure, the kept record and one record per tolerance pair tried.
