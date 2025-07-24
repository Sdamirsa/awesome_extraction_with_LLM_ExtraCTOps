# Configuration Management for ExtraCTOps

This directory contains the centralized configuration system for the ExtraCTOps project. All locations, variables, API endpoints, and settings are managed here.

## Files Overview

### `settings.py` - Main Configuration
- **Primary configuration file** with all core settings
- Environment variable integration (`.env` file support)
- Path management for data, models, logs, uploads, exports, temp
- LLM provider settings (OpenAI, Azure OpenAI, Ollama)
- Performance, security, and file handling settings
- Database and external services configuration
- Application-specific settings for each app
- Virtual environment management settings
- Configuration validation and debugging utilities

### `profiles.py` - Environment Profiles
- **Configuration profiles** for different deployment scenarios
- Development, Testing, Staging, and Production configurations
- Profile-specific database, performance, and security settings
- Easy profile switching via environment variables
- Profile validation and information display

### `locations.py` - Path Management
- **Centralized path and URL management**
- All project directory paths and file locations
- Virtual environment path management
- Application directory mapping
- URL patterns for APIs and services
- File pattern definitions for common files
- Path validation and directory creation utilities

### `.env.example` - Environment Template
- **Template for environment variables**
- API keys for LLM providers (OpenAI, Azure, Ollama)
- Database connection settings
- File storage locations
- Performance and security settings
- Development and testing flags

### `example_usage.py` - Usage Examples
- **Complete demonstration** of how to use the configuration system
- Examples for all configuration components
- Workflow demonstrations
- Best practices and usage patterns

### `config_helper.py` - Configuration Management Utility
- **Interactive configuration management tool**
- View and validate current configuration
- Generator-specific configuration management
- Virtual environment status and validation
- Template generation and configuration debugging
- Check virtual environment status
- Generate environment templates
- Troubleshoot configuration issues

### `MIGRATION.md` - Migration Guide
- **Migration instructions** for moving from old configuration locations
- Steps to centralize virtual environment configuration
- Troubleshooting common migration issues

### `venv_info.json` - Virtual Environment Configuration
- **Central virtual environment management**
- Defines which venvs to create and manage
- Requirements file associations
- Enable/disable individual environments

## Quick Start

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your API keys and settings in `.env`**

3. **Set your environment profile:**
   ```bash
   export EXTRACTOPS_PROFILE=development  # or testing, staging, production
   ```

4. **Use configuration management tools:**
   ```bash
   # View current configuration
   python config/config_helper.py show
   
   # Set up virtual environments  
   python apps/easy_start/manage_venvs.py
   
   # Launch Streamlit apps
   python apps/easy_start/run_streamlit_app.py
   
   # Validate configuration
   python config/config_helper.py validate
   ```

5. **Use in your code:**
   ```python
   from config.settings import *
   from config.locations import locations, get_data_dir
   from config.profiles import get_config_profile
   
   # Access paths
   data_dir = get_data_dir()
   app_path = locations.get_app_path("manual_extraction")
   
   # Access settings
   ollama_url = LLMSettings.OLLAMA['base_url']
   max_file_size = FileSettings.MAX_FILE_SIZE_BYTES
   
   # Get profile-specific config
   config = get_config_profile()
   debug_mode = config.get('debug', False)
   ```

## Configuration Categories

### 📁 **Paths & Directories**
- Project structure paths
- Data storage locations
- Virtual environment paths
- Application directories
- Temporary and cache directories

### 🤖 **LLM Provider Settings**
- OpenAI API configuration
- Azure OpenAI settings
- Ollama local setup
- Request timeouts and limits
- Default models and parameters

### 🗄️ **Database & Storage**
- Database connection URLs
- File storage paths
- Upload and export directories
- Temporary file handling

### ⚡ **Performance Settings**
- Concurrent request limits
- File size limits
- Batch processing sizes
- Request timeouts

### 🔒 **Security Settings**
- API key management
- Session configuration
- CORS and HTTPS settings
- Encryption keys

### 🔧 **Generator Integration**
- Generator-specific configuration
- Virtual environment management for generators
- Test dependency management
- LLM provider integration
- Automatic configuration validation

### 🧪 **Environment Profiles**
- Development settings
- Testing configuration
- Staging environment
- Production deployment

## Environment Variables

Key environment variables you can set:

```bash
# LLM APIs
OPENAI_API_KEY=your_key_here
AZURE_OPENAI_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434/v1

# Paths
DATA_STORAGE_PATH=./data
LOGS_STORAGE_PATH=./logs
UPLOADS_PATH=./uploads

# Settings
DEBUG=true
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=100
REQUEST_TIMEOUT=60

# Profile
EXTRACTOPS_PROFILE=development
```

## Usage Examples

### Basic Configuration Access
```python
from config.settings import *

# Get directory paths
data_path = get_data_dir()
models_path = get_models_dir()

# Access LLM settings
ollama_model = LLMSettings.OLLAMA['default_model']
openai_key = LLMSettings.OPENAI['api_key']

# File handling settings
max_size = FileSettings.MAX_FILE_SIZE_BYTES
supported_types = FileSettings.SUPPORTED_TEXT_EXTENSIONS
```

### Path Management
```python
from config.locations import locations

# Get application paths
manual_app = locations.get_app_path("manual_extraction")
conv_app = locations.get_app_path("conversational_generator")

# Get specific files
settings_file = locations.get_file_path("settings")
venv_info = locations.get_file_path("venv_info")

# Get virtual environment Python
python_path = locations.get_python_executable("ollama")
```

### Profile Management
```python
from config.profiles import get_config_profile, set_profile

# Set development profile
set_profile("development")

# Get current configuration
config = get_config_profile()
debug_mode = config['debug']
db_url = config['database']['url']
```

### Configuration Validation
```python
from config.settings import validate_configuration, print_config_summary

# Check for configuration issues
warnings = validate_configuration()
if warnings:
    for warning in warnings:
        print(f"⚠️  {warning}")

# Print complete configuration summary
print_config_summary()
```

## Best Practices

1. **Use environment variables** for sensitive data (API keys, passwords)
2. **Set appropriate profiles** for different environments
3. **Validate configuration** before deploying
4. **Use centralized paths** instead of hardcoded paths
5. **Keep `.env` file out of version control** (it's in `.gitignore`)
6. **Update `.env.example`** when adding new environment variables

## Integration with Existing Code

The configuration system is designed to work with your existing codebase:

- Import `settings.py` in any module that needs configuration
- Use `locations.py` for all path operations
- Set profiles via environment variables or code
- Environment variables automatically override default settings

## Testing the Configuration

Run the example script to test your configuration:

```bash
cd /path/to/project
python config/example_usage.py
```

This will show you all available settings, validate your configuration, and demonstrate usage patterns.
