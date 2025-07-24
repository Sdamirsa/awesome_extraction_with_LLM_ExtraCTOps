"""
Test runner for generators package.

This script runs all tests in the generators package using the same
virtual environment as the generators, configured through the config system.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from generators.config import get_venv_python_path
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Config system not available, falling back to default paths")
    CONFIG_AVAILABLE = False

def get_venv_python():
    """Get the Python executable from the configured virtual environment."""
    if CONFIG_AVAILABLE:
        venv_python = get_venv_python_path()
    else:
        # Fallback to default path
        project_root = Path(__file__).parent.parent.parent
        venv_python = project_root / "the_venvs" / "venv_ollama" / "bin" / "python"
    
    if venv_python.exists():
        return str(venv_python)
    else:
        print(f"❌ Virtual environment not found at: {venv_python}")
        print("Please ensure the virtual environment is set up.")
        print("You can run: python generators/setup_tests.py")
        return None

def run_tests():
    """Run all tests in the generators package using the Ollama venv."""
    venv_python = get_venv_python()
    if not venv_python:
        return 1
    
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Run pytest with the venv Python
    pytest_args = [
        venv_python, "-m", "pytest",
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        "-x",  # Stop on first failure
    ]
    
    print(f"Using Python: {venv_python}")
    print("Running generators package tests...")
    
    try:
        result = subprocess.run(pytest_args, env=env, cwd=str(project_root))
        exit_code = result.returncode
        
        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code {exit_code}")
        
        return exit_code
        
    except FileNotFoundError:
        print(f"❌ Could not run pytest with {venv_python}")
        print("Make sure pytest is installed in the virtual environment:")
        print(f"{venv_python} -m pip install pytest pytest-asyncio pytest-cov pytest-mock")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
