#!/usr/bin/env python3
"""
Configuration Management Helper Script for ExtraCTOps

This script provides utilities to:
- View current configuration
- Validate configuration
- Generate configuration templates
- Manage environment variables

Usage:
    python config_helper.py [command]

Commands:
    show                - Display current configuration
    validate            - Validate configuration and show warnings
    template            - Generate .env template
    paths               - Show all configured paths
    venvs               - Show virtual environment configuration
    generators          - Show generator-specific configuration
    create-venv-config  - Create default venv_info.json configuration
    help                - Show this help message
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.settings import (
        get_all_settings, 
        validate_configuration, 
        print_config_summary,
        VenvSettings,
        get_data_dir,
        get_models_dir,
        get_logs_dir,
        get_uploads_dir,
        get_exports_dir,
        get_temp_dir
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import configuration: {e}")
    CONFIG_AVAILABLE = False
    sys.exit(1)

def show_configuration():
    """Display the current configuration."""
    print("=" * 60)
    print("🔧 ExtraCTOps Configuration")
    print("=" * 60)
    
    try:
        settings = get_all_settings()
        
        # Project info
        print("\n📁 Project Information:")
        for key, value in settings["project_info"].items():
            print(f"   {key}: {value}")
        
        # Directories
        print("\n📂 Directory Paths:")
        for key, value in settings["directories"].items():
            status = "✅" if Path(value).exists() else "❌"
            print(f"   {status} {key}: {value}")
        
        # Performance settings
        print("\n⚡ Performance Settings:")
        for key, value in settings["performance"].items():
            print(f"   {key}: {value}")
        
        # LLM settings (without API keys)
        print("\n🤖 LLM Configuration:")
        for provider, config in settings["llm"].items():
            print(f"   {provider}:")
            for key, value in config.items():
                if "api_key" not in key.lower():
                    print(f"     {key}: {value}")
                else:
                    print(f"     {key}: {'***SET***' if value else '***NOT SET***'}")
        
        # Virtual environments
        print("\n🐍 Virtual Environment Settings:")
        venv_settings = settings["venv_settings"]
        print(f"   Base directory: {venv_settings['base_dir']}")
        print(f"   Info file: {venv_settings['info_file']}")
        print("   Default environments:")
        for name, config in venv_settings["default_venvs"].items():
            print(f"     {name}: {config['path']}")
        
        # Generator settings
        print("\n🔧 Generator Settings:")
        generator_settings = settings["app_settings"].get("generators", {})
        if generator_settings:
            print(f"   Default venv: {generator_settings.get('default_venv', 'Not set')}")
            print(f"   Supported generators: {generator_settings.get('supported_generators', [])}")
            print(f"   Test dependencies: {len(generator_settings.get('test_dependencies', []))} packages")
        else:
            print("   No generator-specific settings found")
        
    except Exception as e:
        print(f"❌ Error displaying configuration: {e}")

def validate_config():
    """Validate the configuration and show warnings."""
    print("🔍 Validating Configuration...")
    print("-" * 40)
    
    try:
        warnings = validate_configuration()
        
        if not warnings:
            print("✅ Configuration validation passed - no issues found!")
        else:
            print(f"⚠️  Found {len(warnings)} configuration warning(s):")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
        
        # Additional checks
        print("\n🔍 Additional Checks:")
        
        # Check if .env file exists
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            print("   ✅ .env file found")
        else:
            print("   ⚠️  .env file not found - using defaults")
        
        # Check venv_info.json file
        config_venv_info = SCRIPT_DIR / "venv_info.json"
        
        if config_venv_info.exists():
            print(f"   ✅ venv_info.json found: {config_venv_info}")
        else:
            print(f"   ❌ venv_info.json not found: {config_venv_info}")
            print("      💡 This file should contain virtual environment configuration")
        
        # Check generator configuration
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from generators.config import validate_generator_config
            
            print("\n🔧 Generator Configuration:")
            generator_issues = validate_generator_config()
            if not generator_issues:
                print("   ✅ Generator configuration is valid")
            else:
                for issue in generator_issues:
                    print(f"   ⚠️  {issue}")
        except ImportError:
            print("\n🔧 Generator Configuration:")
            print("   ⚠️  Generator config system not available")
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")

def show_paths():
    """Show all configured paths."""
    print("📁 Configured Paths")
    print("-" * 30)
    
    try:
        paths = {
            "Data Directory": get_data_dir(),
            "Models Directory": get_models_dir(),
            "Logs Directory": get_logs_dir(),
            "Uploads Directory": get_uploads_dir(),
            "Exports Directory": get_exports_dir(),
            "Temp Directory": get_temp_dir(),
            "Project Root": PROJECT_ROOT,
            "Config Directory": SCRIPT_DIR,
            "Venv Base": VenvSettings.VENV_BASE_DIR,
        }
        
        for name, path in paths.items():
            status = "✅" if path.exists() else "❌"
            print(f"   {status} {name}: {path}")
            
    except Exception as e:
        print(f"❌ Error showing paths: {e}")

def show_venvs():
    """Show virtual environment configuration."""
    print("🐍 Virtual Environment Configuration")
    print("-" * 40)
    
    try:
        # Check the centralized venv_info.json file
        config_venv_info = SCRIPT_DIR / "venv_info.json"
        
        if config_venv_info.exists():
            print(f"\n📄 Configuration from: {config_venv_info}")
            with open(config_venv_info, 'r') as f:
                venv_data = json.load(f)
            
            for name, config in venv_data.items():
                enabled = "✅" if config.get("enabled", False) else "❌"
                venv_path = PROJECT_ROOT / config.get("venv_path", "")
                exists = "✅" if venv_path.exists() else "❌"
                
                print(f"   {name}:")
                print(f"     Enabled: {enabled}")
                print(f"     Exists: {exists}")
                print(f"     Path: {venv_path}")
                print(f"     Type: {config.get('type', 'unknown')}")
                if config.get("requirements_file"):
                    req_path = PROJECT_ROOT / config["requirements_file"]
                    req_exists = "✅" if req_path.exists() else "❌"
                    print(f"     Requirements: {req_exists} {req_path}")
                print()
        else:
            print(f"\n❌ Central venv_info.json not found: {config_venv_info}")
            print("💡 This file should contain all virtual environment configurations")
            print("💡 You can create it or copy from the old location if it exists")
                
    except Exception as e:
        print(f"❌ Error showing venv configuration: {e}")

def generate_env_template():
    """Generate a .env template file."""
    print("📝 Generating .env template...")
    
    try:
        template_path = PROJECT_ROOT / ".env.example"
        env_path = PROJECT_ROOT / ".env"
        
        if template_path.exists():
            print(f"✅ .env.example already exists at: {template_path}")
            
            if not env_path.exists():
                print("💡 Copy it to .env and fill in your values:")
                print(f"   cp {template_path} {env_path}")
            else:
                print(f"✅ .env file already exists at: {env_path}")
        else:
            print(f"❌ .env.example not found at: {template_path}")
            print("💡 You may need to create it manually or check the config directory")
            
    except Exception as e:
        print(f"❌ Error generating template: {e}")

def show_help():
    """Show help information."""
    print(__doc__)

def show_generators():
    """Show generator-specific configuration and status."""
    print("🔧 Generator Configuration")
    print("-" * 40)
    
    try:
        # Import generator config
        sys.path.insert(0, str(PROJECT_ROOT))
        from generators.config import GeneratorConfig, validate_generator_config
        
        config = GeneratorConfig()
        
        print("Generator Settings:")
        gen_settings = config.generator_settings
        if gen_settings:
            for key, value in gen_settings.items():
                print(f"   {key}: {value}")
        else:
            print("   No generator-specific settings found")
        
        print("\nVirtual Environment:")
        venv_path = config.get_venv_python_path()
        status = "✅" if venv_path.exists() else "❌"
        print(f"   {status} Ollama Python: {venv_path}")
        
        openai_venv_path = config.get_venv_python_path(generator_type="openai")
        if openai_venv_path != venv_path:
            status = "✅" if openai_venv_path.exists() else "❌"
            print(f"   {status} OpenAI Python: {openai_venv_path}")
        else:
            print(f"   OpenAI uses same venv as Ollama")
        
        print("\nLLM Configurations:")
        for provider in ["ollama", "openai", "azure_openai"]:
            settings = getattr(config, f"{provider}_settings")
            model = settings.get("default_model", "Not set")
            api_key = settings.get("api_key", "")
            key_status = "✅" if api_key else "❌"
            print(f"   {provider}: {model} {key_status if 'api_key' in settings else ''}")
        
        print("\nTest Configuration:")
        test_deps = config.get_test_dependencies()
        print(f"   Dependencies: {len(test_deps)} packages")
        print(f"   Testing enabled: {config.is_testing_enabled()}")
        
        # Validation
        print("\nValidation:")
        issues = validate_generator_config()
        if not issues:
            print("   ✅ All checks passed")
        else:
            for issue in issues:
                print(f"   ⚠️  {issue}")
        
    except ImportError as e:
        print(f"❌ Could not import generator configuration: {e}")
        print("Make sure the generators package is properly set up")

def create_default_venv_config():
    """Create a default venv_info.json file if it doesn't exist."""
    print("📝 Creating default venv_info.json...")
    
    try:
        config_venv_info = SCRIPT_DIR / "venv_info.json"
        
        if config_venv_info.exists():
            print(f"✅ venv_info.json already exists at: {config_venv_info}")
            return
        
        # Default configuration
        default_config = {
            "venv_main": {
                "type": "main",
                "script": None,
                "enabled": True,
                "venv_path": "the_venvs/venv_main",
                "python_exec": "the_venvs/venv_main/bin/python",
                "requirements_file": "the_venvs/requirements_venv_main.txt"
            },
            "venv_ollama": {
                "type": "script-specific",
                "script": "generators/generator_Ollama.py",
                "enabled": True,
                "venv_path": "the_venvs/venv_ollama",
                "python_exec": "the_venvs/venv_ollama/bin/python",
                "requirements_file": "the_venvs/requirements_venv_ollama.txt"
            },
            "venv_streamlit": {
                "type": "script-specific",
                "script": "apps/",
                "enabled": True,
                "venv_path": "the_venvs/venv_streamlit",
                "python_exec": "the_venvs/venv_streamlit/bin/python",
                "requirements_file": "the_venvs/venv_streamlit.txt"
            }
        }
        
        with open(config_venv_info, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✅ Created default venv_info.json at: {config_venv_info}")
        print("💡 You can now edit this file to customize your virtual environments")
        
    except Exception as e:
        print(f"❌ Error creating default venv configuration: {e}")

def main():
    """Main entry point."""
    if not CONFIG_AVAILABLE:
        print("❌ Configuration system not available")
        sys.exit(1)
    
    # Get command from arguments
    command = sys.argv[1] if len(sys.argv) > 1 else "show"
    
    commands = {
        "show": show_configuration,
        "validate": validate_config,
        "template": generate_env_template,
        "paths": show_paths,
        "venvs": show_venvs,
        "generators": show_generators,
        "create-venv-config": create_default_venv_config,
        "help": show_help,
    }
    
    if command in commands:
        commands[command]()
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Available commands:", ", ".join(commands.keys()))
        show_help()

if __name__ == "__main__":
    main()
