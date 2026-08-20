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

# Install dependencies and setup package
pip install -r requirements.txt
# material-hasher package needs to be installed via
pip install git+https://github.com/lematerial/material-hasher.git
# muon optimizer (addition in development, install required)
pip install git+https://github.com/KellerJordan/Muon
# Install CrystaLLM-pi in editable mode
pip install -e .
```

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
