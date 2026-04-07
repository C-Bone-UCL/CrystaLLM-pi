#!/bin/bash
set -e

# Extract API_KEY_PATH from _train.py
API_KEY_FILE=$(python3 << 'PYTHON'
import re
with open("/app/_train.py") as f:
    content = f.read()
match = re.search(r'API_KEY_PATH\s*=\s*["\']([^"\']+)["\']', content)
print(match.group(1) if match else "API_keys.jsonc")
PYTHON
)

# Make it an absolute path if it is not already
if [[ "$API_KEY_FILE" != /* ]]; then
  API_KEY_FILE="/app/$API_KEY_FILE"
fi

FALLBACK_API_KEY_FILE="/tmp/API_keys.jsonc"

write_api_key_file() {
  local target_file="$1"
  mkdir -p "$(dirname "$target_file")"
  cat > "$target_file" << EOF
{
  "HF_key": "${HF_KEY}",
  "wandb_key": "${WANDB_KEY}"
}
EOF
}

if [[ -n "${HF_KEY:-}" && -n "${WANDB_KEY:-}" ]]; then
  if mkdir -p "$(dirname "$API_KEY_FILE")" 2>/dev/null && touch "$API_KEY_FILE" 2>/dev/null; then
    write_api_key_file "$API_KEY_FILE"
  else
    echo "Warning: Cannot write API keys to $API_KEY_FILE, using $FALLBACK_API_KEY_FILE instead."
    API_KEY_FILE="$FALLBACK_API_KEY_FILE"
    write_api_key_file "$API_KEY_FILE"
  fi
else
  echo "Warning: HF_KEY and/or WANDB_KEY not set. Commands requiring authentication may fail."
fi

export API_KEY_PATH="$API_KEY_FILE"

# Export tokens for libs that read env vars directly
export HF_TOKEN="${HF_KEY:-}"
export HF_HUB_TOKEN="${HF_KEY:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_KEY:-}"
export WANDB_API_KEY="${WANDB_KEY:-}"

# Use writable caches/temp; prefer /app/outputs when writable, else fallback to /tmp
CACHE_BASE="/app/outputs"
if ! mkdir -p "$CACHE_BASE/.tmp" 2>/dev/null; then
  CACHE_BASE="/tmp/crystallm_api"
fi

export HF_HOME="$CACHE_BASE/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export XDG_CACHE_HOME="$CACHE_BASE/.cache"
export TMPDIR="$CACHE_BASE/.tmp"
export HOME="$CACHE_BASE"
export TORCH_HOME="$CACHE_BASE/.cache/torch"
export API_JOB_LOG_DIR="$CACHE_BASE/api_job_logs"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR" "$TORCH_HOME" "$HOME" "$API_JOB_LOG_DIR" 2>/dev/null || true

exec "$@"
