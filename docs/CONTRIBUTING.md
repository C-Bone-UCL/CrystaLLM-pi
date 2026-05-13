# Contributing to CrystaLLM-pi

Thank you for considering contributing to CrystaLLM-pi. We welcome contributions from the community, whether they are bug reports, feature requests, documentation improvements, or code modifications.

## How to Contribute

### Reporting Bugs
If you find a bug in the source code, you can help us by submitting an issue to our GitHub Repository. Even better, you can submit a Pull Request with a fix.

### Suggesting Enhancements
If you have an idea for an enhancement, please submit an issue explaining your idea, why it would be useful, and how it might be implemented.

### Pull Requests
1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (see the Testing section below).
5. Issue that pull request.

## Development Setup
To set up the project locally for development:

1. Clone your fork of the repository.
2. Install the required dependencies as pre the `README.md`.
3. Add your Credentials for Hugging-Face and WandB (may need to create accounts).
4. Run the local tests to ensure your environment is configured correctly.

## Testing and Continuous Integration (CI)
To ensure stability and reproducibility, this project relies on a three-tier testing strategy. All code contributions must pass the relevant CI checks before being merged.

### Tier 1: Automated Offline CPU Tests
* **Trigger:** Runs automatically on all Pull Requests and pushes to `main`.
* **Description:** This is the core testing suite. It runs entirely offline using CPU resources to verify base functionality, algorithms, and local processing logic. 
* **Requirement:** Must pass for any Pull Request to be accepted.

### Tier 2: Manual Secrets-Backed Tests
* **Trigger:** Triggered manually via GitHub Actions (`workflow_dispatch`) by repository maintainers.
* **Description:** Exercises code paths requiring external APIs (e.g., Hugging Face, Weights & Biases). If the required API keys (`HF_KEY`, `WANDB_KEY`) are present in the repository secrets, the full suite runs. If not, it falls back to the offline suite.

### Tier 3: Manual Docker API & Integration Tests
* **Trigger:** Triggered manually via GitHub Actions (`workflow_dispatch`).
* **Description:** Builds the Docker container and runs API smoke tests to verify the deployment environment. An optional flag can be set during dispatch to run the slower, full integration suite. Note: This runs on standard GitHub-hosted runners (CPU), while production deployments would need GPU environments.

## Governance and Support
CrystaLLM-pi is primarily developed and maintained by Cyprien Bone (PhD Student @ UCL). 

Please note that this repository is maintained on a **best-effort basis**. While we value community input and strive to review pull requests, investigate bug reports, and answer questions in a timely manner, we cannot guarantee immediate responses or resolution. Support and maintenance are provided as time and research resources permit.

## License
By contributing to CrystaLLM-pi, you agree that your contributions will be licensed under the MIT License.