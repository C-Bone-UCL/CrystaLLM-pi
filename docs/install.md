# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.12 if needed
- Hugging Face and Weights & Biases accounts should be set up
- (Optional but recommended) NVIDIA GPU. Linux GPU installs need driver 560+ and use CUDA 12.6 wheels for x86 and aarch64


## Setup

```bash
# Clone the repository
git clone https://github.com/C-Bone-UCL/CrystaLLM-pi.git
cd CrystaLLM-pi

# Create .venv and install CrystaLLM-pi and all its dependencies
uv sync --extra all

# Activate the environment, or run commands with `uv run`
source .venv/bin/activate
```

### Choosing what to install

`--extra all` installs everything. If you only need part of the toolkit, install just that part instead. Extras can be combined, for example `uv sync --extra train --extra metrics`.

| Command | Gives you |
|---|---|
| `uv sync` | generating structures with released models |
| `uv sync --extra train` | + training and fine-tuning your own |
| `uv sync --extra api` | + the containerised HTTP service |
| `uv sync --extra metrics` | + VUN, stability and property scoring |
| `uv sync --extra notebooks` | + the tutorial notebooks |
| `uv sync --extra all` | everything above |

To keep the environment outside the repository, set `UV_PROJECT_ENVIRONMENT=/path/to/venv` before running `uv sync`.

### API Keys Configuration

Create `API_keys.jsonc` in the root directory for HuggingFace and Weights & Biases integration:

```jsonc
// filepath: API_keys.jsonc
{
  "HF_key": "your_hf_key_here", // Hugging Face token
  "wandb_key": "your_wandb_api_key_here" // Weights & Biases key
}
```

<br>
<br>
