# Virtual Environment Configuration Migration Guide

## New Centralized Configuration

As of the latest updates, virtual environment configuration has been centralized to improve project organization and reduce duplication.

### Location Change

**Old locations (no longer used):**
- `apps/easy_start/venv_info.json`
- `the_venvs/venv_info.json`

**New centralized location:**
- `config/venv_info.json` ✅

### Migration Steps

If you have existing `venv_info.json` files in old locations, follow these steps:

1. **Check if you need to migrate:**
   ```bash
   python config/config_helper.py validate
   ```

2. **Create default configuration (if needed):**
   ```bash
   python config/config_helper.py create-venv-config
   ```

3. **Manual migration (if you have custom settings):**
   ```bash
   # Copy your existing configuration
   cp apps/easy_start/venv_info.json config/venv_info.json
   # OR
   cp the_venvs/venv_info.json config/venv_info.json
   ```

4. **Verify the migration:**
   ```bash
   python config/config_helper.py venvs
   ```

5. **Test virtual environment management:**
   ```bash
   cd apps/easy_start
   python manage_venvs.py
   ```

### Benefits of Centralization

- **Single source of truth** - No more duplicate configurations
- **Better organization** - All configuration in the `config/` folder
- **Easier maintenance** - Update settings in one place
- **Version control friendly** - Clear configuration history
- **Reduced errors** - No conflicts between multiple config files

### Configuration File Format

The `config/venv_info.json` file format remains the same:

```json
{
  "venv_name": {
    "type": "main|script-specific",
    "script": "path/to/script.py",
    "enabled": true,
    "venv_path": "the_venvs/venv_name",
    "python_exec": "the_venvs/venv_name/bin/python",
    "requirements_file": "the_venvs/requirements_file.txt"
  }
}
```

### Tools and Commands

All tools now reference the centralized configuration:

```bash
# View configuration
python config/config_helper.py show
python config/config_helper.py venvs

# Validate setup
python config/config_helper.py validate

# Manage virtual environments
python apps/easy_start/manage_venvs.py

# Run applications
python apps/easy_start/run_streamlit_app.py
```

### Troubleshooting

**Problem:** `venv_info.json not found`
**Solution:** 
```bash
python config/config_helper.py create-venv-config
```

**Problem:** Old configuration files exist
**Solution:** Remove them after migrating:
```bash
rm -f apps/easy_start/venv_info.json
rm -f the_venvs/venv_info.json
```

**Problem:** Scripts can't find configuration
**Solution:** Make sure you're running from the project root or the configuration exists:
```bash
ls -la config/venv_info.json
```

For more help, run:
```bash
python config/config_helper.py help
```
