# Generation

Prompt construction, the multi-GPU generation pool, candidate ranking and post-processing.

## Prompts

::: _utils._generating.make_prompts.create_automatic_prompts
::: _utils._generating.make_prompts.create_manual_prompts
::: _utils._generating.make_prompts.augment_cif_for_prompt
::: _utils._generating.make_prompts.extract_composition_from_cif
::: _utils._generating.make_prompts.load_hf_dataset
::: _utils._generating.make_prompts.is_already_bracketed

## Generation pool

`run_generation_pool` is the programmatic entry point. The worker-side functions run inside pool
processes and share module state through globals, which is why they sit in their own module.

::: _utils._generating.generate_cifs.run_generation_pool
::: _utils._generating.generate_cifs.resolve_generation_plan
::: _utils._generating.generate_cifs.build_generation_kwargs
::: _utils._generating.generate_cifs.build_output_df
::: _utils._generating.generate_cifs.parse_condition_vector
::: _utils._generating.generate_cifs.get_model_class
::: _utils._generating.generate_cifs.get_model_max_length
::: _utils._generating.generate_cifs.init_tokenizer
::: _utils._generating.generate_cifs.check_cif
::: _utils._generating.generate_cifs.get_material_id

## Workers

::: _utils._generating.workers.setup_device
::: _utils._generating.workers.init_worker
::: _utils._generating.workers.generate_on_gpu
::: _utils._generating.workers.progress_listener

## Ranking candidates

Two strategies for picking a winner out of a Z search. `LOGP` ranks by model perplexity.
`PEARSON` simulates each candidate's powder pattern and correlates it with the input scan.

::: _utils._generating.scoring_methods.score_output_logp
::: _utils._generating.scoring_methods.score_outputs_logp
::: _utils._generating.scoring_methods.simulate_profile
::: _utils._generating.scoring_methods.pearson_score
::: _utils._generating.scoring_methods.score_generated_rows

## Post-processing

::: _utils._generating.postprocess.postprocess
::: _utils._generating.postprocess.process_dataframe
::: _utils._generating.postprocess.validate_cif_numerics

## Direct generation helpers

Used by `_load_and_generate.py` to turn command-line arguments into generation specs.

::: _utils.direct_gen.get_condition_format
::: _utils.direct_gen.is_xrd_model
::: _utils.direct_gen.validate_model_conditions
::: _utils.direct_gen.parse_xrd_file_to_condition_vector
::: _utils.direct_gen.parse_condition_list_args
::: _utils.direct_gen.parse_reduced_formula_list_arg
::: _utils.direct_gen.canonicalize_reduced_formulas
::: _utils.direct_gen.reduced_formula_to_explicit_formula
::: _utils.direct_gen.build_reduced_formula_specs
::: _utils.direct_gen.build_formula_condition_map
::: _utils.direct_gen.attach_prompt_metadata
::: _utils.direct_gen.reduce_rows_for_reduced_formula_search
::: _utils.direct_gen.get_hf_model_max_length
::: _utils.direct_gen.get_visible_gpu_count
::: _utils.direct_gen.resolve_multi_gpu_workers
