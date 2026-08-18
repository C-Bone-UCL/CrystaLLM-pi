# Metrics and analysis

Validity predicates, the VUN pipeline, Materials Project reference data and the virtualiser.

The scoring CLIs that wrap these live in `_utils/_scoring/` and are documented by their module
docstrings: `vun_metrics`, `xrd_metrics`, `property_metrics`, `mace_ehull`, `dft_ehull`.

## Validity predicates

Each returns a bool and is safe to call on model output that may not parse.

::: _utils.validity.is_valid
::: _utils.validity.is_sensible
::: _utils.validity.is_formula_consistent
::: _utils.validity.is_space_group_consistent
::: _utils.validity.is_atom_site_multiplicity_consistent
::: _utils.validity.bond_length_reasonableness_score
::: _utils.validity.get_density

## Validity, uniqueness and novelty

::: _utils.metrics.get_valid
::: _utils.metrics.get_unique
::: _utils.metrics.get_novelty
::: _utils.metrics.get_comp_novelty
::: _utils.metrics.load_and_process_generated_data
::: _utils.metrics.build_generated_structures
::: _utils.metrics.extract_generated_formulas
::: _utils.metrics.load_and_filter_training_data
::: _utils.metrics.build_reference_compositions
::: _utils.metrics.predict_properties

## Materials Project reference data

Downloaded and cached once; only the stability metrics need it.

::: _utils.mp_data.MPDataProvider
::: _utils.mp_data.download_mp_data

## Virtual crystals

Converts an ordered structure into a disordered virtual crystal and promotes it to its
higher-symmetry parent.

::: _utils._virtualiser.virtualiser.load_config
::: _utils._virtualiser.virtualiser.compute_pair_fractions
::: _utils._virtualiser.virtualiser.virtualise_structure
::: _utils._virtualiser.virtualiser.promote_symmetry
