import os
import sys
import json
import subprocess
from pathlib import Path
import platform

# Add project root to path for config imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.settings import VenvSettings, PROJECT_ROOT as CONFIG_PROJECT_ROOT, get_all_settings
    USE_CONFIG = True
except ImportError:
    print("⚠️  Config module not available, using fallback paths")
    USE_CONFIG = False

# 🛠️ INSTRUCTIONS
print("""
🛠️ INSTRUCTIONS:
1. Open config/venv_info.json
2. Set "enabled": true for any venv you want to create or install.
3. Then run this script.
""")

# Determine paths - use centralized config/venv_info.json
CONFIG_VENV_INFO_PATH = PROJECT_ROOT / "config" / "venv_info.json"

# Use the centralized configuration file
VENV_INFO_PATH = CONFIG_VENV_INFO_PATH
print(f"📍 Using centralized venv_info.json: {VENV_INFO_PATH}")

VENV_DIR = PROJECT_ROOT / "the_venvs"

# Detect platform
IS_WINDOWS = platform.system() == "Windows"

# Check for venv_info.json
if not VENV_INFO_PATH.exists():
    print(f"❌ venv_info.json not found at {VENV_INFO_PATH}")
    print("💡 Make sure the centralized configuration file exists in the config/ folder")
    print("💡 You can create it or copy from another location if needed")
    sys.exit(1)

# Confirm with user
proceed = input("❓ Do you want to create venvs and install requirements for enabled environments? (y/n): ").strip().lower()
if proceed != "y":
    print("🚫 Aborted by user. No actions taken.")
    sys.exit(0)

# Load JSON
with open(VENV_INFO_PATH, "r") as f:
    venv_data = json.load(f)

print(f"📊 Found {len(venv_data)} virtual environment configurations")

# Show configuration summary if using config system
if USE_CONFIG:
    try:
        settings = get_all_settings()
        print(f"📁 Project root: {settings['project_info']['project_root']}")
        print(f"🗂️  Venv base directory: {settings['venv_settings']['base_dir']}")
        if settings['project_info']['debug']:
            print("🐛 Debug mode: ON")
    except Exception as e:
        print(f"⚠️  Could not load full configuration: {e}")

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