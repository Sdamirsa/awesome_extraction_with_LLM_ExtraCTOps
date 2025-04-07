import os
import sys
import json
import subprocess
from pathlib import Path
import platform

# 🛠️ INSTRUCTIONS
print("""
🛠️ INSTRUCTIONS:
1. Open the_venvs/venv_info.json
2. Set "enabled": true for any venv you want to create or install.
3. Then run this script.
""")

# Determine paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VENV_DIR = PROJECT_ROOT / "the_venvs"
VENV_INFO_PATH = VENV_DIR / "venv_info.json"

# Detect platform
IS_WINDOWS = platform.system() == "Windows"

# Check for venv_info.json
if not VENV_INFO_PATH.exists():
    print(f"❌ venv_info.json not found at {VENV_INFO_PATH}")
    sys.exit(1)

# Confirm with user
proceed = input("❓ Do you want to create venvs and install requirements for enabled environments? (y/n): ").strip().lower()
if proceed != "y":
    print("🚫 Aborted by user. No actions taken.")
    sys.exit(0)

# Load JSON
with open(VENV_INFO_PATH, "r") as f:
    venv_data = json.load(f)

# Process entries
for name, entry in venv_data.items():
    if not entry.get("enabled", False):
        continue

    venv_path = PROJECT_ROOT / entry["venv_path"]
    requirements_file = entry.get("requirements_file")
    python_exec = venv_path / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

    # Create venv if missing
    if not venv_path.exists():
        print(f"🛠️  Creating venv for [{name}] at {venv_path}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    else:
        print(f"✅ [{name}] venv already exists. Skipping creation.")

    # Install requirements
    if requirements_file:
        req_path = PROJECT_ROOT / requirements_file
        if req_path.exists() and req_path.stat().st_size > 0:
            print(f"📦 Installing from {requirements_file}...")
            subprocess.run([str(python_exec), "-m", "pip", "install", "-r", str(req_path)], check=True)
        else:
            print(f"⚠️  [{name}] Requirements file is missing or empty. Skipping.")
    else:
        print(f"🔹 [{name}] No requirements file specified.")

print("\n✅ All enabled environments processed.")