# Conventions

Symbols on these pages are documented in two different shapes. This page says which shape applies
when.

## Two docstring formats

| Format | Used when | Looks like |
|---|---|---|
| Parameter table | An argument is a tensor or array whose shape you need, or the symbol is on these pages and takes 4+ parameters that need more than their name | `Args:` / `Returns:` blocks, with a shape on every tensor line |
| Paragraph | Everything else, whatever the parameter count | Prose describing behaviour and return value |

The test is whether a caller can get it wrong, not how many parameters there are.
[`generate_on_gpu`](generation.md#_utils._generating.workers.generate_on_gpu) has a long argument
list and gets a paragraph: it runs inside a worker pool and nothing calls it directly.
[`run_generation_pool`](generation.md#_utils._generating.generate_cifs.run_generation_pool) gets a
table, because it is the programmatic entry point.

Internal helpers, meaning anything not listed on these pages, keep a one-line summary or a
comment. They never carry `Args:` blocks.

## Command-line scripts

The package ships no console scripts, so each CLI is its `.py` file. The module docstring carries
the contract and one runnable `Usage:` command, and `main()` gets a single line. Listing the flags
in both places guarantees they drift apart, and `argparse --help` is already the authority.

Those usage commands are written as raw string literals. A trailing backslash inside a normal
literal is a Python line continuation, which would collapse a wrapped shell command onto one line
and drop the backslash.

## Module naming

Inside `_utils/`, module files are lowercase (acronyms included), carry no leading underscore, and
never end in `_utils.py`. Package directories keep the leading underscore, which marks the
boundary once so the files inside need not restate it. Two exceptions:

- `_models/` keeps CapWords filenames (`PKV_model.py`, `PrefixXRD_model.py`). They track the
  checkpoint families they load, which is worth more here than PEP 8 casing.
- Root scripts (`_train.py`, `_load_and_generate.py`, `_api.py`) keep their leading underscore.
  With no console scripts these files are the CLI, and the underscore marks them as entry points
  rather than an importable public surface.

## Legacy models

`PKV_model.py` and `Slider_model.py` are load-only. Released checkpoints still load into them.
They carry a module-level deprecation note and nothing else. Their successors are `Prefix_model.py` and
`Residual_model.py`, which is what new work should use.

The two generations are never unified. A `PKV` checkpoint loads into `Prefix` with no error and
then produces different outputs, silently.

## What the tests check

Three registered tests fail the build when this drifts:

| Test | Checks |
|---|---|
| `module_docstring_format` | Every module docstring follows the house shape |
| `api_manifest` | Every symbol listed on these pages exists and has a docstring |
| `module_naming` | The naming rules above, plus an `__init__.py` in every package directory |

Run them with `python -m tests.local.runner --cpu --offline`.
