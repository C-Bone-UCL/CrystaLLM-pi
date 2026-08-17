"""Entry point for local test suites."""

import sys
from pathlib import Path

# Running this file by path puts tests/local on sys.path instead of the repo root, so
# "tests.local.runner" would resolve to any other installed distribution that ships a
# top-level tests package. Put the repo root first so the local suite always wins.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.local.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
