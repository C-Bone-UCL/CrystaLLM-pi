# API

!!! tip "Run it in a notebook"
    [`T3_API_density_example.ipynb`](https://github.com/C-Bone-UCL/CrystaLLM-pi/blob/main/notebooks/T3_API_density_example.ipynb) drives the containerised API from Python, no curl needed.

Containerized API provides REST endpoints for preprocessing, training, generation, and metrics.

Current API parity notes:

- `/generate/direct` accepts exactly one output target: `output_parquet` or `output_cif_dir`.
- `/preprocessing/clean` exposes `property3_normaliser`, `filter_to`, and `count_tokens`.
- Metrics routes include `/metrics/vun`, `/metrics/ehull`, `/metrics/xrd`, and `/metrics/property`.

First-time host setup (Linux + NVIDIA GPU required):

```bash
# Install NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Quick sanity checks
docker --version
docker compose version
nvidia-smi
# Verify Docker can access your GPUs
# If successful, this will download a test image and print nvidia-smi table
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Optional: run Docker without sudo, but you need to re-login or reboot for membership to apply
sudo usermod -aG docker $USER
```

Setup (first time bringing container up, if requirements.txt or dockerfile or system dependencies are changed):

```bash
# Make the env file

## In command line
### Copy the template to your local, git-ignored .env file (only edit the .env)
cp docker/.env.example docker/.env

### inject your current host machine's UID and GID into the .env file
sed -i "s/^UID=.*/UID=$(id -u)/" docker/.env
sed -i "s/^GID=.*/GID=$(id -g)/" docker/.env

## In the .env file
### pick GPUs exposed to the API container
### default in template is 0,1 (can do that or all)
NVIDIA_VISIBLE_DEVICES=0,1
DOCKER_GPUS=all

### your API keys to the .env
HF_KEY=your_hf_token_here
WANDB_KEY=your_wandb_key_here

# back in Command line
## make dirs needed for the api
mkdir -p data outputs

## OR Dev Mode
### If Docker still requires sudo:
sudo --preserve-env=HF_KEY,WANDB_KEY,UID,GID make api-up-dev-build
### else
make api-up-dev-build

## Production mode
make api-up-build
```

### Usage Modes (CLI)

| Command | Mode | Description |
| --- | --- | --- |
| `make api-up-dev` | **Dev** | Uses `uvicorn --reload`. Restarts on file changes. Best for rapid development. |
| `make api-up` | **Prod** | No auto-reload. More stable. Recommended for long-running generation jobs. |

```bash
# Start the API
make api-up-dev # Development mode
# OR
make api-up # Production mode

# Utilities
make api-health # Check server status (wait a couple mins before this will work)
make api-logs # Follow logs
make api-down # Stop and cleanup
```

### Running Tests

Ensure your API container is running, then run the test suites to verify the pipeline.

```bash
# Fast Routing Tests (Checks if endpoints respond, 10 secs)
make api-test

# Full Integration Tests (Runs generation & metrics end-to-end, ~10 mins)
make api-test-with-integration

# For Local Unit Tests activate environment
conda activate CrystaLLM-pi_env
# Then run tests (1 min)
python -m tests.local.suite --cpu
```

### Quickstart Generation Examples

See the examples below.

For `/generate/direct`, provide exactly one of `output_parquet` or `output_cif_dir`.

<details>
<summary>Expand for comprehensive API generation examples (curl)</summary>

### Direct generation (Explicit Z, Spacegroup targeting)

```bash
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-pi_base",
    "reduced_formula_list": "TiO2",
    "z_list": "2",
    "spacegroups": "P4_2/mnm",
    "level": "level_4",
    "num_return_sequences": 5,
    "max_return_attempts": 2,
    "output_parquet": "/app/outputs/test_generated_structures.parquet"
  }'
```

### Direct generation (SLME, level_1 so no composition provided)

```bash
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-pi_SLME",
    "condition_lists": ["25.0"],
    "level": "level_1",
    "num_return_sequences": 5,
    "output_parquet": "/app/outputs/solar_screening.parquet"
  }'
```

### Direct generation (Mattergen-XRD, Early-Stopping Z-Search with Spacegroup)

```bash
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
    "reduced_formula_list": "TiO2",
    "spacegroups": "P4_2/mnm",
    "level": "level_4",
    "search_zs": true,
    "xrd_files": ["/app/tests/fixtures/test_rutile_processed.csv"],
    "num_return_sequences": 5,
    "max_return_attempts": 2,
    "target_valid_cifs": 1,
    "scoring_mode": "none",
    "output_parquet": "/app/outputs/xrd_mattergen_early_stop.parquet"
  }'
```

### Direct generation (Chili100K-XRD, LOGP Ranked Z-Search with Raw Wavelength Conversion)

```bash
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-pi_Chili100K-XRD",
    "reduced_formula_list": "TiO2",
    "search_zs": true,
    "xrd_files": ["/app/tests/fixtures/test_rutile_raw.xy"],
    "xrd_wavelength": 0.71073,
    "num_return_sequences": 10,
    "max_return_attempts": 2,
    "target_valid_cifs": 5,
    "scoring_mode": "LOGP",
    "temperature": 1.0,
    "output_cif_dir": "/app/outputs/xrd_chili_logp"
  }'
```

### Direct generation (Mattergen-XRD without xrd_files)

```bash
curl -X POST "http://localhost:8000/generate/direct" \
  -H "Content-Type: application/json" \
  -d '{
    "hf_model_path": "c-bone/CrystaLLM-pi_Mattergen-XRD",
    "reduced_formula_list": "NaCl",
    "search_zs": true,
    "num_return_sequences": 5,
    "max_return_attempts": 1,
    "target_valid_cifs": 1,
    "scoring_mode": "logp",
    "output_parquet": "/app/outputs/mattergen_no_xrd.parquet"
  }'
```

### Virtualise a generated CIF (inline element pairs)

Convert an ordered CIF to a disordered virtual crystal using inline matching pairs arrays:

```bash
curl -X POST "http://localhost:8000/virtualise" \
  -H "Content-Type: application/json" \
  -d '{
    "input_cif": "/app/outputs/Mg3ZnO4.cif",
    "output_cif": "/app/outputs/Mg3ZnO4_virtual.cif",
    "virtual_pairs": [["Mg", "Zn"]],
    "symprec": 0.003,
    "angle_tolerance": 0.5
  }'
```

### Virtualise a generated CIF (YAML config file)

Alternatively, supply a YAML config file:

```bash
curl -X POST "http://localhost:8000/virtualise" \
  -H "Content-Type: application/json" \
  -d '{
    "input_cif": "/app/outputs/FeSbO4_ordered.cif",
    "output_cif": "/app/outputs/FeSbO4_virtual.cif",
    "config_file": "/app/data/virtualiser_config.yaml"
  }'
```

</details>

### API Training GPU Selection

* You can force behavior in requests for training:
* `"multi_gpu": false` forces single-process launch
* `"multi_gpu": true` requests torchrun (only used when 2+ GPUs are visible)
* `"nproc_per_node": N` caps torchrun workers when multi-GPU is active

* For generate, all available GPUs are used

### Troubleshooting

* **API Permission Denied**: Run `chmod 644 API_keys.jsonc` and `chmod -R 775 outputs data`.
* **Cache Failures**: Ensure `outputs/` and `data/` are owned by the current user: `sudo chown -R $USER:$USER outputs data`.
* **Logs**: Job and test logs are stored in `outputs/api_job_logs/` and `outputs/api_test_logs/`.
* **Docs:** Visit `http://localhost:8000/docs` in browser to view the interactive API schema and execute endpoints directly. (needs to be on, or linked to machine where API is running)

### Cancel a job or check status

To cancel a running job:

```bash
curl -X POST "http://localhost:8000/jobs/<job-id>/cancel"
```

To check status of a current job:

```bash
curl "http://localhost:8000/jobs/<job-id>"
```

# Apptainer (Production Build)

Use this when you want the API packaged as a portable `.sif` (e.g. for HPC / no-Docker environments).

### 1) Build the production Docker image

```bash
# Builds the docker image so we can make a .sif file from it, this command doesnt boot up the container.
make api-build
```

### 2) Build Apptainer image from Docker daemon (latest tag)

```bash
# this compresses to about 8GB and took me 15 min to build
make api-apptainer-build
```

### 3) Run the API from Apptainer (GPU + mounted data/output)

> Apptainer does not read `.env` automatically, so we export the two API keys:

```bash
# If you ran make api-up-build, the container may be up and running
# Run this command to shut it down to clear up the :8000 port for apptainer image
make api-down

# Export the keys Apptainer needs
set -a; source <(grep -E '^(HF_KEY|WANDB_KEY)=' docker/.env); set +a

# warning about api_keys.jsonc is harmless here
make api-apptainer-run
```

Leave this terminal open. Health checks, tests, and curl commands are identical to the Docker flow (see `Running Tests`).

#### You can also override names/tags:

```bash
make api-apptainer-build APPTAINER_SIF=my-api.sif APPTAINER_DOCKER_IMAGE=crystallm-api APPTAINER_DOCKER_SOURCE_TAG=local APPTAINER_DOCKER_TAG=latest
```
