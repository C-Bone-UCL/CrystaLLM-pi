"""Entry point for the API test suite.

Routing tests check that every endpoint responds. The integration tier drives generation and metrics
end to end, and takes considerably longer.

Usage:
    python -m tests.api.suite --docker_url http://localhost:8000
"""

from tests.api.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
