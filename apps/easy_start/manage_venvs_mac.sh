#!/bin/bash

# ========== 🛠️ INSTRUCTIONS ==========
echo ""
echo "🛠️ INSTRUCTIONS:"
echo "1. Open ../../the_venvs/venv_info.json"
echo "2. Set 'enabled': true for any venv you want to create or install."
echo "3. Then run this script."
echo ""

# ========== VARIABLES ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../../the_venvs"
INFO_FILE="$VENV_DIR/venv_info.json"

# ========== CHECK FILE EXISTS ==========
if [ ! -f "$INFO_FILE" ]; then
  echo "❌ venv_info.json not found at $INFO_FILE"
  exit 1
fi

read -p "❓ Do you want to create venvs and install requirements for enabled environments? (y/n): " CONTINUE

if [[ "$CONTINUE" != "y" ]]; then
  echo "🚫 Aborted by user. No actions taken."
  exit 0
fi

echo "🔄 Processing enabled virtual environments..."

# ========== MAIN LOOP ==========
jq -c 'to_entries[]' "$INFO_FILE" | while read -r entry; do
  NAME=$(echo "$entry" | jq -r '.key')
  ENABLED=$(echo "$entry" | jq -r '.value.enabled')
  VENV_PATH=$(echo "$entry" | jq -r '.value.venv_path')
  PYTHON_EXEC=$(echo "$entry" | jq -r '.value.python_exec')
  REQUIREMENTS=$(echo "$entry" | jq -r '.value.requirements_file')

  if [[ "$ENABLED" == "true" ]]; then
    ABS_VENV_PATH="$SCRIPT_DIR/../../$VENV_PATH"
    ABS_REQS_FILE="$SCRIPT_DIR/../../$REQUIREMENTS"

    if [ ! -d "$ABS_VENV_PATH" ]; then
      echo "🛠️  Creating venv for [$NAME]..."
      python3 -m venv "$ABS_VENV_PATH"
    else
      echo "✅ [$NAME] venv already exists. Skipping creation."
    fi

    if [[ "$REQUIREMENTS" != "null" && -s "$ABS_REQS_FILE" ]]; then
      echo "📦 Installing from $REQUIREMENTS..."
      "$ABS_VENV_PATH/bin/python" -m pip install -r "$ABS_REQS_FILE"
    else
      echo "⚠️  [$NAME] No requirements to install or file is empty."
    fi
  fi
done

echo "✅ All enabled environments processed."