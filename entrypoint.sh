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

# Create the directory if it does not exist
mkdir -p "$(dirname "$API_KEY_FILE")"

# Generate the API keys file
cat > "$API_KEY_FILE" << EOF
{
  "HF_key": "${HF_KEY}",
  "wandb_key": "${WANDB_KEY}"
}
EOF

exec "$@"
