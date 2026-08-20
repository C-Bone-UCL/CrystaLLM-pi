r"""Upload a loadable CrystaLLM-pi checkpoint to the Hugging Face Hub.

The upload includes the checkpoint, configuration, and tokenizer files required by load-and-generate, allowing the repository to be supplied directly as `--hf_model_path`.

Usage:
    ```bash
    python _utils/_preprocessing/save_model_to_hf.py --checkpoint model_ckpts/run/checkpoint-1500 \
        --repo c-bone/CrystaLLM-pi_my-model
    ```
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _utils import load_api_keys


def upload_custom_model_to_hub(
    checkpoint_path: str | Path,
    repo_name: str,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    """Upload the files needed to reload a CrystaLLM checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {checkpoint_path}")

    if commit_message is None:
        commit_message = "Upload model checkpoint"

    # Exclude optimizer state while retaining everything needed to reload the model.
    essential_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "vocabulary.json",
        "spacegroups.txt",
        "training_args.json",
    ]

    api = HfApi()
    api.create_repo(repo_name, private=private, exist_ok=True)

    print("Uploading files to Hub...")
    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_name,
        allow_patterns=essential_files,
        commit_message=commit_message,
    )
    print(f"Successfully uploaded model to: https://huggingface.co/{repo_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload CrystaLLM models to Hugging Face Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (directory)")
    parser.add_argument("--repo", required=True, help="HF repo name (username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--message", help="Commit message")

    args = parser.parse_args()

    hf_token = load_api_keys("API_keys.jsonc")["HF_key"]
    login(token=hf_token)

    upload_custom_model_to_hub(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo,
        private=args.private,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
