import os
import sys
import subprocess
from pathlib import Path
import platform

# 🎯 CONFIG
IS_WINDOWS = platform.system() == "Windows"
VENV_EXEC = Path("the_venvs/venv_streamlit/Scripts/python.exe" if IS_WINDOWS else "the_venvs/venv_streamlit/bin/python")

# 📁 Project structure
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # e.g., /project/apps
ROOT_DIR = PARENT_DIR.parent    # e.g., /project/

# 🔍 Discover apps with app.py
apps = {}
for folder in PARENT_DIR.iterdir():
    if folder.is_dir():
        app_file = folder / "app.py"
        if app_file.exists():
            apps[folder.name] = app_file

if not apps:
    print("❌ No Streamlit apps found in subfolders.")
    sys.exit(1)

# 👤 Ask user to select one
print("📦 Available Streamlit apps:")
for i, name in enumerate(apps.keys(), 1):
    print(f"{i}. {name}")

try:
    choice = int(input("Select an app to run by number: ").strip())
    selected_app = list(apps.values())[choice - 1]
except (ValueError, IndexError):
    print("❌ Invalid selection.")
    sys.exit(1)

# 🚀 Run the Streamlit app
if not VENV_EXEC.exists():
    print(f"❌ Streamlit virtual environment not found at: {VENV_EXEC}")
    sys.exit(1)

print(f"🚀 Launching Streamlit app: {selected_app}")
subprocess.run([str(VENV_EXEC), "-m", "streamlit", "run", str(selected_app)])