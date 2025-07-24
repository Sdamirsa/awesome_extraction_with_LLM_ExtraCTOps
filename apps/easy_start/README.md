# 🚀 Easy Start Tools

This folder contains utility scripts to help you quickly set up and manage your ExtraCTOps environment.

## 📁 Files in this folder

### Core Scripts

- **`manage_venvs.py`** - Create and manage virtual environments
- **`run_streamlit_app.py`** - Launch Streamlit applications easily

### Configuration Files

- **`venv_info.json`** - Local virtual environment configuration (optional override)
- **`manage_venvs_mac.sh`** - macOS/Linux shell script for venv management
- **`manage_venvs_windows.bat`** - Windows batch script for venv management

**Note:** The main configuration files are now located in the `config/` folder:
- `config/venv_info.json` - Primary virtual environment configuration
- `config/config_helper.py` - Configuration management utility

## 🛠️ Quick Setup

### 1. Set up Virtual Environments

First, ensure your virtual environments are configured:

```bash
# View current venv configuration
python config_helper.py venvs

# Create/update virtual environments
python manage_venvs.py
```

### 2. Configure Environment Variables

Copy the environment template and customize it:

```bash
# Generate .env template (if not exists)
python config_helper.py template

# Copy and edit the environment file
cp ../../.env.example ../../.env
# Edit .env with your actual values
```

### 3. Launch Applications

Run any Streamlit app in the project:

```bash
python run_streamlit_app.py
```

This will show you a menu of available apps to run.

## 📊 Configuration Management

### View Configuration

```bash
# Show full configuration (from config folder)
python ../../config/config_helper.py show

# Show just paths
python ../../config/config_helper.py paths

# Validate configuration
python ../../config/config_helper.py validate

# Check virtual environments
python ../../config/config_helper.py venvs
```

### Virtual Environment Management

The virtual environment configuration is now managed in `config/venv_info.json`. You can also create a local override in this folder if needed.

**Priority order:**
1. `config/venv_info.json` (primary configuration)
2. `apps/easy_start/venv_info.json` (local override)
3. `the_venvs/venv_info.json` (fallback) 

**Structure:**
```json
{
  "venv_name": {
    "type": "main|script-specific",
    "script": "path/to/script.py",
    "enabled": true|false,
    "venv_path": "the_venvs/venv_name",
    "python_exec": "the_venvs/venv_name/bin/python",
    "requirements_file": "the_venvs/requirements_file.txt"
  }
}
```

**To add a new virtual environment:**

1. Edit `venv_info.json`
2. Add your configuration
3. Set `"enabled": true`
4. Run `python manage_venvs.py`

## 🔧 Configuration System Integration

These scripts integrate with the main configuration system (`config/settings.py`):

- **Automatic path detection** - Uses configured directories
- **Environment variable support** - Respects .env settings  
- **Fallback behavior** - Works even without full config system
- **Validation** - Checks for common configuration issues

## 📝 Examples

### Example 1: Set up new environment for a custom generator

1. Edit `config/venv_info.json` (or create local `venv_info.json`):
```json
{
  "venv_my_generator": {
    "type": "script-specific",
    "script": "generators/my_generator.py",
    "enabled": true,
    "venv_path": "the_venvs/venv_my_generator",
    "python_exec": "the_venvs/venv_my_generator/bin/python",
    "requirements_file": "the_venvs/requirements_my_generator.txt"
  }
}
```

2. Create requirements file:
```bash
echo "torch>=1.9.0" > ../../the_venvs/requirements_my_generator.txt
echo "transformers>=4.0.0" >> ../../the_venvs/requirements_my_generator.txt
```

3. Create and install:
```bash
python manage_venvs.py
```

### Example 2: Debug configuration issues

```bash
# Check what's wrong (from config folder)
python ../../config/config_helper.py validate

# View all paths
python ../../config/config_helper.py paths

# Check venv status
python ../../config/config_helper.py venvs
```

## 🆘 Troubleshooting

### Virtual Environment Issues

- **"venv not found"**: Run `python manage_venvs.py` first
- **"requirements missing"**: Check if requirements file exists and has content
- **"permission denied"**: Make sure shell scripts are executable (`chmod +x *.sh`)

### Configuration Issues

- **"Config module not available"**: The scripts will use fallback paths
- **"API keys not set"**: Copy `.env.example` to `.env` and fill in values
- **"Directory not found"**: Run `python config_helper.py validate` for guidance

### Streamlit Issues

- **"No apps found"**: Make sure there are `app.py` files in sibling directories
- **"Streamlit not installed"**: Check if requirements include streamlit
- **"Import errors"**: Verify virtual environment has all dependencies

## 🔗 Related Documentation

- [Main Configuration Documentation](../../config/README.md)
- [Virtual Environment Management](../../the_venvs/Readme.md)
- [Project Structure](../../README.md)

---

💡 **Tip**: Run `python ../../config/config_helper.py help` for a quick reference of available commands.
