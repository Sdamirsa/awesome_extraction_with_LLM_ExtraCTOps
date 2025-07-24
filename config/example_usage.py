"""
Example script showing how to use the configuration system.
Run this script to see how to access and use all the configuration components.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

# Import configuration modules
from config.settings import *
from config.profiles import get_config_profile, print_profile_info, set_profile
from config.locations import locations, get_data_dir, get_app_path

def demonstrate_basic_settings():
    """Demonstrate basic settings usage."""
    print("=" * 60)
    print("1. BASIC SETTINGS USAGE")
    print("=" * 60)
    
    # Access basic paths
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {get_data_dir()}")
    print(f"Models Directory: {get_models_dir()}")
    print(f"Logs Directory: {get_logs_dir()}")
    print()
    
    # Access LLM settings
    print("LLM Settings:")
    print(f"  Ollama Base URL: {LLMSettings.OLLAMA['base_url']}")
    print(f"  Ollama Default Model: {LLMSettings.OLLAMA['default_model']}")
    print(f"  Request Timeout: {LLMSettings.OLLAMA['request_timeout']}s")
    print()
    
    # Access file settings
    print("File Settings:")
    print(f"  Max File Size: {FileSettings.MAX_FILE_SIZE_BYTES} bytes")
    print(f"  Supported Text Extensions: {FileSettings.SUPPORTED_TEXT_EXTENSIONS}")
    print(f"  Default Upload Path: {FileSettings.DEFAULT_UPLOAD_PATH}")
    print()

def demonstrate_profiles():
    """Demonstrate configuration profiles."""
    print("=" * 60)
    print("2. CONFIGURATION PROFILES")
    print("=" * 60)
    
    # Show current profile
    print("Current profile:")
    print_profile_info()
    print()
    
    # Switch to different profiles and show differences
    profiles_to_demo = ["development", "testing", "production"]
    
    for profile in profiles_to_demo:
        print(f"--- {profile.upper()} PROFILE ---")
        config = get_config_profile(profile)
        print(f"Debug Mode: {config.get('debug', 'Not set')}")
        print(f"Log Level: {config.get('log_level', 'Not set')}")
        print(f"Database URL: {config.get('database', {}).get('url', 'Not set')}")
        print()

def demonstrate_locations():
    """Demonstrate location management."""
    print("=" * 60)
    print("3. LOCATION MANAGEMENT")
    print("=" * 60)
    
    # Show project structure
    print("Key Project Locations:")
    print(f"  Manual Extraction App: {get_app_path('manual_extraction')}")
    print(f"  Conversational Generator: {get_app_path('conversational_generator')}")
    print(f"  Data to Pydantic: {get_app_path('data_to_pydantic')}")
    print()
    
    # Show virtual environment paths
    print("Virtual Environment Paths:")
    for venv_name in ["main", "ollama", "streamlit"]:
        venv_path = locations.get_venv_path(venv_name)
        python_path = locations.get_python_executable(venv_name)
        print(f"  {venv_name.capitalize()}: {venv_path}")
        print(f"    Python: {python_path}")
    print()
    
    # Show file patterns
    print("Common File Patterns:")
    important_files = ["settings", "venv_info", "manual_extraction_app", "generator_ollama"]
    for file_key in important_files:
        file_path = locations.get_file_path(file_key)
        exists = "✅" if file_path.exists() else "❌"
        print(f"  {file_key}: {file_path} {exists}")
    print()
    
    # Show URLs
    print("Important URLs:")
    url_keys = ["localhost_streamlit", "localhost_ollama_api", "github_repo"]
    for url_key in url_keys:
        url = locations.get_url(url_key)
        print(f"  {url_key}: {url}")
    print()

def demonstrate_environment_variables():
    """Demonstrate environment variable usage."""
    print("=" * 60)
    print("4. ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    # Show how to check for API keys
    api_keys = {
        "OpenAI": LLMSettings.OPENAI['api_key'],
        "Azure OpenAI": LLMSettings.AZURE_OPENAI['api_key'],
    }
    
    print("API Key Status:")
    for service, key in api_keys.items():
        status = "✅ Set" if key else "❌ Not Set"
        print(f"  {service}: {status}")
    print()
    
    # Show environment overrides
    print("Environment Variable Examples:")
    print("  Set OLLAMA_BASE_URL to override Ollama URL")
    print("  Set LOG_LEVEL to change logging level")
    print("  Set MAX_FILE_SIZE_MB to change file size limit")
    print("  Set DEBUG=true to enable debug mode")
    print()
    
    # Show current environment settings
    print("Current Environment Settings:")
    print(f"  DEBUG: {DEBUG}")
    print(f"  LOG_LEVEL: {LOG_LEVEL}")
    print(f"  MAX_FILE_SIZE_MB: {MAX_FILE_SIZE_MB}")
    print()

def demonstrate_validation():
    """Demonstrate configuration validation."""
    print("=" * 60)
    print("5. CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Run validation
    warnings = validate_configuration()
    
    if warnings:
        print("Configuration Issues Found:")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("✅ Configuration validation passed - no issues found")
    print()

def demonstrate_app_specific_settings():
    """Demonstrate app-specific configuration."""
    print("=" * 60)
    print("6. APP-SPECIFIC SETTINGS")
    print("=" * 60)
    
    # Manual extraction settings
    print("Manual Extraction App Settings:")
    manual_settings = AppSettings.MANUAL_EXTRACTION
    for key, value in manual_settings.items():
        print(f"  {key}: {value}")
    print()
    
    # Conversational generator settings
    print("Conversational Generator Settings:")
    conv_settings = AppSettings.CONVERSATIONAL_GENERATOR
    print(f"  Max History: {conv_settings['max_conversation_history']}")
    print(f"  Auto Backup: {conv_settings['auto_backup_interval']}s")
    print("  Default Pricing (per 1M tokens):")
    for model, pricing in conv_settings['default_pricing'].items():
        if model != "default":
            print(f"    {model}: ${pricing['prompt']:.2f} input, ${pricing['completion']:.2f} output")
    print()

def demonstrate_complete_workflow():
    """Demonstrate a complete configuration workflow."""
    print("=" * 60)
    print("7. COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # 1. Set development profile
    print("Step 1: Setting development profile...")
    set_profile("development")
    
    # 2. Ensure directories exist
    print("Step 2: Ensuring directories exist...")
    locations.ensure_directories_exist()
    
    # 3. Get application-specific settings
    print("Step 3: Getting app settings...")
    config = get_config_profile()
    app_settings = AppSettings.MANUAL_EXTRACTION
    
    # 4. Construct file paths
    print("Step 4: Constructing file paths...")
    upload_dir = get_uploads_dir()
    export_dir = get_exports_dir()
    log_file = get_logs_dir() / "app.log"
    
    # 5. Show the complete setup
    print("\nComplete Configuration Setup:")
    print(f"  Profile: {get_config_profile.__name__ if hasattr(get_config_profile, '__name__') else 'development'}")
    print(f"  Debug Mode: {config.get('debug', False)}")
    print(f"  Upload Directory: {upload_dir}")
    print(f"  Export Directory: {export_dir}")
    print(f"  Log File: {log_file}")
    print(f"  Dashboard Height: {app_settings['dashboard_height']}px")
    print(f"  Max File Size: {FileSettings.MAX_FILE_SIZE_BYTES} bytes")
    print()

def main():
    """Run all configuration demonstrations."""
    print("ExtraCTOps Configuration System Demonstration")
    print("=" * 80)
    print()
    
    try:
        demonstrate_basic_settings()
        demonstrate_profiles()
        demonstrate_locations()
        demonstrate_environment_variables()
        demonstrate_validation()
        demonstrate_app_specific_settings()
        demonstrate_complete_workflow()
        
        print("=" * 80)
        print("Configuration demonstration completed successfully!")
        print()
        print("Next Steps:")
        print("1. Copy .env.example to .env and fill in your API keys")
        print("2. Set EXTRACTOPS_PROFILE environment variable to your desired profile")
        print("3. Use 'from config.settings import *' in your applications")
        print("4. Use 'from config.locations import locations' for path management")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
