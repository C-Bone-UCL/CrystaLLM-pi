# Generation

The generation pipeline constructs prompts, generates CIFs, post-processes the results, and ranks candidate structures.

Each stage is exposed as a standalone command. The corresponding module docstrings provide representative `Usage:` commands.

The `_load_and_generate` command combines the generation workflow and is documented on the [CLI entry points](cli.md) page.

## Prompts

::: _utils._generating.make_prompts

## Generating CIFs

::: _utils._generating.generate_cifs

## Post-processing

::: _utils._generating.postprocess

## Ranking candidates

::: _utils._generating.scoring_methods