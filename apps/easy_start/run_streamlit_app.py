import os
import sys
import subprocess
from pathlib import Path
import platform

# Add project root to path for config imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # e.g., /project/apps
ROOT_DIR = PARENT_DIR.parent    # e.g., /project/
sys.path.insert(0, str(ROOT_DIR))

try:
    from config.settings import VenvSettings, get_all_settings
    USE_CONFIG = True
    print("✅ Using configuration from config.settings")
except ImportError:
    print("⚠️  Config module not available, using fallback configuration")
    USE_CONFIG = False

# 🎯 CONFIG
IS_WINDOWS = platform.system() == "Windows"

# Determine virtual environment path
if USE_CONFIG:
    # Use configuration system
    venv_streamlit_path = VenvSettings.VENV_BASE_DIR / "venv_streamlit"
    VENV_EXEC = venv_streamlit_path / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
else:
    # Fallback to hardcoded path
    VENV_EXEC = ROOT_DIR / ("the_venvs/venv_streamlit/Scripts/python.exe" if IS_WINDOWS else "the_venvs/venv_streamlit/bin/python")

print(f"📍 Virtual environment path: {VENV_EXEC}")

# 📁 Project structure - no changes needed here

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
    
    # Try to provide helpful guidance
    if USE_CONFIG:
        print("💡 Try running the manage_venvs.py script first to set up virtual environments:")
        print("   python manage_venvs.py")
    else:
        print("💡 Make sure you have created the virtual environment:")
        print(f"   python -m venv {VENV_EXEC.parent}")
        print(f"   {VENV_EXEC} -m pip install streamlit")
    
    sys.exit(1)

print(f"🚀 Launching Streamlit app: {selected_app}")

# Enhanced command with better error handling
try:
    result = subprocess.run([str(VENV_EXEC), "-m", "streamlit", "run", str(selected_app)], 
                          check=True, 
                          cwd=str(ROOT_DIR))
except subprocess.CalledProcessError as e:
    print(f"❌ Failed to run Streamlit app: {e}")
    print("💡 Make sure Streamlit is installed in the virtual environment:")
    print(f"   {VENV_EXEC} -m pip install streamlit")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n👋 Streamlit app stopped by user")
    sys.exit(0)