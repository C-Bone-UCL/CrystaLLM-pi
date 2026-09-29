# Metrics and analysis

The evaluation layer defines structure validity, uniqueness, novelty, XRD agreement, property targets, and stability metrics for generated CIFs.

The validity predicates define the structural checks used by the evaluation workflows. The scoring CLIs operate on post-processed generation results stored in parquet files.

## Validity predicates

These predicates return booleans and can be applied to model output that may fail to parse.

::: _utils.validity.is_valid
::: _utils.validity.is_sensible
::: _utils.validity.is_formula_consistent
::: _utils.validity.is_space_group_consistent
::: _utils.validity.is_atom_site_multiplicity_consistent
::: _utils.validity.bond_length_reasonableness_score

Density is calculated from the parsed structure.

::: _utils.validity.get_density

## Evaluation CLIs

### Validity evaluation

::: _utils._generating.evaluate_cifs

### Validity, uniqueness, and novelty

::: _utils._scoring.vun_metrics
::: _utils._scoring.vun_metrics.compute_vun_metrics

### XRD structure match

::: _utils._scoring.xrd_metrics
::: _utils._scoring.xrd_metrics.get_match_rate_and_rms
::: _utils._scoring.xrd_metrics.is_valid_bench

### Property targets

::: _utils._scoring.property_metrics

### Stability

::: _utils._scoring.mace_ehull
::: _utils._scoring.dft_ehull

## Materials Project reference data

Materials Project entries are downloaded and cached for the stability metrics.

::: _utils.mp_data.MPDataProvider
    options:
      members:
        - get_phase_diagram
        - compute_ehull_and_eform

## Virtual crystals

Convert ordered structures to disordered virtual crystals and promote them to higher-symmetry parent structures.

::: _utils._virtualiser.virtualiser
::: _utils._virtualiser.virtualiser.virtualise_structure
::: _utils._virtualiser.virtualiser.promote_symmetry