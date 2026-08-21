# Installation

## Prerequisites

- Python 3.10+
- PyTorch 2.1+
- Conda for environment management
- Hugging Face and Weights & Biases accounts should be set up
- (Optional but recommended) CUDA-compatible GPU


## Setup

```bash
# Clone the repository
git clone https://github.com/C-Bone-UCL/CrystaLLM-pi.git
cd CrystaLLM-pi

# Create virtual environment
conda create -n CrystaLLM-pi_env python=3.10
conda activate CrystaLLM-pi_env

# Install CrystaLLM-pi and all its dependencies
pip install -e ".[all]"
```

### Choosing what to install

`[all]` installs everything. If you only need part of the toolkit, install just that part
instead. It is much faster and avoids building DeepSpeed.

| Command | Gives you |
|---|---|
| `pip install -e .` | generating structures with released models |
| `pip install -e ".[train]"` | + training and finetuning your own |
| `pip install -e ".[api]"` | + the containerised HTTP service |
| `pip install -e ".[metrics]"` | + VUN, stability and property scoring |
| `pip install -e ".[notebooks]"` | + the tutorial notebooks |
| `pip install -e ".[all]"` | everything above |

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
