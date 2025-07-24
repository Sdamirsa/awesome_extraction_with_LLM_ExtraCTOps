#!/usr/bin/env python3
"""
Kernel Setup Helper for ExtraCTOps Jupyter Notebooks

This script helps verify and configure Jupyter kernels for the ExtraCTOps project.
Run this to check that your virtual environments are properly registered as Jupyter kernels.
"""

import subprocess
import sys
import os
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False, e.stderr

def check_venv_python(venv_name):
    """Check if a virtual environment's Python is working."""
    project_root = get_project_root()
    python_path = project_root / "the_venvs" / venv_name / "bin" / "python"
    
    if not python_path.exists():
        print(f"❌ Python executable not found: {python_path}")
        return False
    
    success, output = run_command([str(python_path), "--version"], f"Checking {venv_name} Python version")
    return success

def check_jupyter_kernels():
    """Check available Jupyter kernels."""
    project_root = get_project_root()
    jupyter_path = project_root / "the_venvs" / "venv_main" / "bin" / "jupyter"
    
    if not jupyter_path.exists():
        print(f"❌ Jupyter not found: {jupyter_path}")
        return False
    
    success, output = run_command([str(jupyter_path), "kernelspec", "list"], "Listing available Jupyter kernels")
    return success, output

def test_notebook_imports(venv_name):
    """Test key imports in a virtual environment."""
    project_root = get_project_root()
    python_path = project_root / "the_venvs" / venv_name / "bin" / "python"
    
    test_imports = [
        "import pandas",
        "import pydantic",
        "import sys; print(f'Python path: {sys.executable}')",
        "from generators.config import GeneratorConfig",
        "from the_pydantics.EchoReport import EchoReport",
        "from utils.ExtraCTOps_loops import ExtraCTOpsProcessor",
    ]
    
    print(f"\n🧪 Testing imports in {venv_name}")
    print("-" * 50)
    
    for test_import in test_imports:
        cmd = [str(python_path), "-c", test_import]
        success, output = run_command(cmd, f"Testing: {test_import}")
        if not success:
            print(f"⚠️  Import failed: {test_import}")
        else:
            if "Python path:" in output:
                print(f"✅ {output.strip()}")

def main():
    """Main function to check all kernel setups."""
    print("🚀 ExtraCTOps Kernel Setup Verification")
    print("=" * 60)
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Check virtual environments
    venvs = ["venv_main", "venv_ollama"]
    for venv in venvs:
        print(f"\n📦 Checking {venv}")
        check_venv_python(venv)
    
    # Check Jupyter kernels
    print(f"\n🔬 Checking Jupyter Kernels")
    success, output = check_jupyter_kernels()
    
    if success:
        # Check for ExtraCTOps kernels
        extractops_kernels = [line for line in output.split('\n') if 'extractops' in line.lower()]
        if extractops_kernels:
            print("✅ Found ExtraCTOps kernels:")
            for kernel in extractops_kernels:
                print(f"  {kernel}")
        else:
            print("⚠️  No ExtraCTOps-specific kernels found")
    
    # Test imports
    for venv in venvs:
        test_notebook_imports(venv)
    
    print(f"\n📝 Summary")
    print("=" * 60)
    print("If you see '✅' for most tests, your kernels are properly configured!")
    print("In VS Code:")
    print("1. Open the_scripts/echo_extraction_production.ipynb")
    print("2. Click 'Select Kernel' in the top right")
    print("3. Choose 'ExtraCTOps Main (Python 3.12)' or 'ExtraCTOps Ollama (Python 3.12)'")
    print("4. Run the notebook cells")
    
    print(f"\n💡 Troubleshooting:")
    print("- If kernels don't appear in VS Code, restart VS Code")
    print("- Make sure the Python extension is installed in VS Code")
    print("- Check that Jupyter extension is installed in VS Code")

if __name__ == "__main__":
    main()
