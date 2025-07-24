#!/usr/bin/env python3
"""
Setup script for generators testing environment.

This script ensures that all test dependencies are installed in the 
virtual environment configured for generators.
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from generators.config import get_venv_python_path, get_test_dependencies
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Config system not available, using fallback configuration")
    CONFIG_AVAILABLE = False

def get_venv_python():
    """Get the Python executable from the configured virtual environment."""
    if CONFIG_AVAILABLE:
        venv_python = get_venv_python_path()
    else:
        # Fallback to default
        project_root = Path(__file__).parent.parent
        venv_python = project_root / "the_venvs" / "venv_ollama" / "bin" / "python"
    
    if venv_python.exists():
        return str(venv_python)
    else:
        print(f"❌ Virtual environment not found at: {venv_python}")
        print("Please set up the virtual environment first.")
        print("Check the config system and venv_info.json configuration.")
        return None

def get_test_packages():
    """Get the list of test packages to install."""
    if CONFIG_AVAILABLE:
        return get_test_dependencies()
    else:
        # Fallback test dependencies
        return [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0"
        ]

def install_test_dependencies():
    """Install test dependencies in the configured virtual environment."""
    venv_python = get_venv_python()
    if not venv_python:
        return 1
    
    # Get test packages from config
    test_packages = get_test_packages()
    
    print(f"Using Python: {venv_python}")
    print("Installing test dependencies...")
    
    for package in test_packages:
        print(f"Installing {package}...")
        try:
            result = subprocess.run([
                venv_python, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return 1
    
    print("\n✅ All test dependencies installed successfully!")
    return 0

def verify_installation():
    """Verify that pytest can be imported and run."""
    venv_python = get_venv_python()
    if not venv_python:
        return 1
    
    print("Verifying pytest installation...")
    try:
        result = subprocess.run([
            venv_python, "-c", "import pytest; print(f'pytest version: {pytest.__version__}')"
        ], check=True, capture_output=True, text=True)
        print(f"✅ {result.stdout.strip()}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to verify pytest: {e}")
        return 1

def main():
    """Main setup function."""
    print("Setting up generators testing environment...\n")
    
    if install_test_dependencies() != 0:
        sys.exit(1)
    
    if verify_installation() != 0:
        sys.exit(1)
    
    print("\n🎉 Setup complete! You can now run tests with:")
    print("python generators/tests/run_tests.py")
    
    # Ask if user wants to run tests now
    run_now = input("\nRun tests now? (y/n): ").lower().strip()
    if run_now in ['y', 'yes']:
        test_runner = Path(__file__).parent / "tests" / "run_tests.py"
        subprocess.run([sys.executable, str(test_runner)])

if __name__ == "__main__":
    main()
