"""Local test section: source conventions for module docstrings, naming and annotations.

Fails the suite when a module docstring drifts from the expected format, and when a symbol listed on
the docs API pages carries no docstring of its own. Those pages double as the manifest of what
counts as public, since Python has no export list to check against.

Nothing here imports the code it checks. Everything is read with ast instead, because `_train`
imports wandb and the offline CI tier does not install it, so an import-based check would fail the
gate rather than the docstring.

Inspired by: https://github.com/Frost-group/PolaronMobility.jl/blob/main/test/docs_smoke.jl
"""

import ast
import builtins
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SKIP_DIRS = {"build", "outputs", "data", "__pycache__", ".git", ".tmp", "wandb",
             "model_ckpts", "crystallm_pi.egg-info", "site", "dist", "HF-databases"}


def _python_files():
    """Every tracked Python file, skipping build artefacts and generated trees."""
    return [p for p in sorted(REPO_ROOT.rglob("*.py"))
            if not any(part in SKIP_DIRS for part in p.parts)]


def _is_cli_entry_point(tree):
    """True when the module has a real `if __name__ == "__main__":` guard.

    Matched on the parsed tree, not by substring: this very module mentions `__main__` inside a
    string literal, and a substring check would call every such file a CLI.
    """
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name) and test.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in test.comparators)):
            return True
    return False


def _docstring_source(path, tree):
    """The module docstring exactly as written, before Python interprets any escapes."""
    node = tree.body[0]
    lines = path.read_text().splitlines(keepends=True)
    return "".join(lines[node.lineno - 1: node.end_lineno])


def _wraps_a_command(path, tree):
    """True when the docstring source ends a line with a shell continuation backslash."""
    return any(line.rstrip("\n").endswith("\\")
               for line in _docstring_source(path, tree).splitlines())


def _is_raw_literal(path, tree):
    """True when the docstring is written as an r-string.

    A lone backslash-newline inside a normal literal is a Python line continuation, so the wrapped
    shell command collapses onto one line and loses the backslash when rendered. Raw literals keep
    both, which is what a docs site and help() need to show a runnable command.
    """
    return _docstring_source(path, tree).lstrip().startswith(("r\"\"\"", "r'''"))


class SourceConventionTests:
    """Check module docstring format, module naming and annotation hygiene."""

    def __init__(self, temp_dir, test_data):
        self.temp_dir = temp_dir
        self.test_data = test_data

    def test_module_docstrings_follow_the_house_format(self):
        """Every module docstring matches the house format in the docstring style guide."""
        problems = []
        for path in _python_files():
            rel = path.relative_to(REPO_ROOT)
            tree = ast.parse(path.read_text())
            raw = ast.get_docstring(tree, clean=False)
            if raw is None:
                problems.append(f"{rel}: no module docstring")
                continue
            if raw.startswith("\n"):
                problems.append(f"{rel}: summary must start on the opening triple quote")
                continue
            lines = raw.strip().split("\n")
            summary = lines[0].rstrip()
            if len(summary) > 95:
                problems.append(f"{rel}: summary is {len(summary)} chars, limit is 95")
            if not summary.endswith((".", "!")):
                problems.append(f"{rel}: summary must end in a period")
            if len(lines) > 1 and lines[1].strip():
                problems.append(f"{rel}: needs a blank line after the summary")
            if _is_cli_entry_point(tree) and "Usage:" not in raw:
                problems.append(f"{rel}: CLI entry point needs a Usage: block")
            if _wraps_a_command(path, tree) and not _is_raw_literal(path, tree):
                problems.append(f'{rel}: docstring wraps a command, so the literal must be r"""')
        assert not problems, "module docstring format:\n  " + "\n  ".join(problems)

    def test_no_legacy_typing_generics(self):
        """No module spells generics the old way, since the 3.10 floor makes builtins available."""
        names = ["L" + "ist", "D" + "ict", "T" + "uple", "S" + "et", "T" + "ype",
                 "O" + "ptional", "U" + "nion"]
        legacy = re.compile(r"from typing import[^\n]*\b(" + "|".join(names) + r")\b")
        offenders = [str(p.relative_to(REPO_ROOT)) for p in _python_files()
                     if p.name not in ("PKV_model.py", "Slider_model.py")
                     and legacy.search(p.read_text())]
        assert not offenders, ("use builtin generics and | unions, not typing: " + ", ".join(offenders))

    def test_annotations_do_not_break_imports(self):
        """Every annotation resolves at import time.

        Annotating with a name the module never imports, or one defined further down the file,
        raises NameError on import rather than at call time. Both happened during the annotation
        pass, so this compiles each module and evaluates its annotations.
        """
        problems = []
        for path in _python_files():
            tree = ast.parse(path.read_text())
            defined = {n.name for n in ast.walk(tree)
                       if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            imported = {a.asname or a.name.split(".")[0]
                        for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            imported |= {a.asname or a.name
                         for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                args = node.args
                anns = [a.annotation for a in args.posonlyargs + args.args + args.kwonlyargs
                        if a.annotation is not None]
                if node.returns is not None:
                    anns.append(node.returns)
                for ann in anns:
                    if isinstance(ann, ast.Constant):      # quoted forward reference, fine
                        continue
                    for name in {n.id for n in ast.walk(ann) if isinstance(n, ast.Name)}:
                        if hasattr(builtins, name) or name in defined or name in imported:
                            continue
                        problems.append(f"{path.relative_to(REPO_ROOT)}::{node.name} -> {name}")
        assert not problems, "annotations name something unresolvable:\n  " + "\n  ".join(sorted(set(problems)))


    def test_module_names_follow_the_house_convention(self):
        """Module files under `_utils/` are lowercase, unprefixed and free of a `_utils` suffix.

        `_models/` is exempt: its filenames track the checkpoint families they load (`PKV_model.py`,
        `PrefixXRD_model.py`), which is worth more than PEP 8 casing. Root scripts are exempt too,
        since the leading underscore there marks the package's own entry points.
        """
        utils = REPO_ROOT / "_utils"
        sources = [p for p in sorted(utils.rglob("*.py"))
                   if not any(part in SKIP_DIRS for part in p.parts)]
        problems = []
        for path in sources:
            rel, name = path.relative_to(REPO_ROOT), path.name
            if name == "__init__.py":
                continue
            if name.lower() != name:
                problems.append(f"{rel}: uppercase in filename, acronyms go lowercase")
            if name.startswith("_"):
                problems.append(f"{rel}: leading underscore, the package directory already marks it private")
            if name.endswith("_utils.py"):
                problems.append(f"{rel}: `_utils` suffix stutters with the package name")

        # A subpackage without __init__.py is invisible to find_packages(), so an installed wheel
        # silently ships _utils without it while a clone keeps working.
        for directory in sorted({p.parent for p in sources}):
            if not (directory / "__init__.py").exists():
                problems.append(f"{directory.relative_to(REPO_ROOT)}: no __init__.py, find_packages() skips it")

        assert not problems, "module naming violations:\n  " + "\n  ".join(problems)

    def test_api_pages_list_documented_symbols(self):
        """Every `:::` directive on the docs API pages resolves to a symbol with a docstring.

        These pages double as the manifest of what counts as public, since Python has no export list
        to check against. A directive that names a moved or renamed symbol would build a page with a
        hole in it, and mkdocs only catches that when the docs toolchain is installed, which the
        offline tier does not have.
        """
        api_dir = REPO_ROOT / "docs" / "api"
        assert api_dir.is_dir(), "docs/api is missing, the API reference pages are the manifest"

        seen, problems = {}, []
        for page in sorted(api_dir.glob("*.md")):
            for dotted in re.findall(r"^::: ([A-Za-z_][A-Za-z0-9_.]*)", page.read_text(), re.M):
                if dotted in seen:
                    problems.append(f"{dotted}: listed on both {seen[dotted]} and {page.name}")
                    continue
                seen[dotted] = page.name

                # longest dotted prefix that is a real module, remainder is the symbol chain
                parts = dotted.split(".")
                module = chain = None
                for cut in range(len(parts), 0, -1):
                    candidate = REPO_ROOT.joinpath(*parts[:cut]).with_suffix(".py")
                    if candidate.is_file():
                        module, chain = candidate, parts[cut:]
                        break
                if module is None:
                    problems.append(f"{dotted}: no module on disk ({page.name})")
                    continue

                node = ast.parse(module.read_text())
                for name in chain:
                    node = next((c for c in ast.iter_child_nodes(node)
                                 if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                                 and c.name == name), None)
                    if node is None:
                        problems.append(f"{dotted}: `{name}` not found in {module.relative_to(REPO_ROOT)}")
                        break
                else:
                    if not ast.get_docstring(node):
                        problems.append(f"{dotted}: listed on {page.name} but has no docstring")

        assert seen, "no ::: directives found, the API pages are empty"
        assert not problems, "API reference manifest problems:\n  " + "\n  ".join(problems)
