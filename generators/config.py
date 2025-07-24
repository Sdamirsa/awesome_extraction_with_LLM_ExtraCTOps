"""
Generator-specific configuration and utilities.

This module provides configuration management specifically for the generators package,
integrating with the centralized config system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path for config imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.settings import LLMSettings, AppSettings, VenvSettings
    from config.profiles import get_active_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import config system: {e}")
    CONFIG_AVAILABLE = False

class GeneratorConfig:
    """Configuration manager for generators."""
    
    def __init__(self):
        self._profile_config = get_active_config() if CONFIG_AVAILABLE else {}
        self._generator_settings = AppSettings.GENERATORS if CONFIG_AVAILABLE else {}
    
    @property
    def ollama_settings(self) -> Dict[str, Any]:
        """Get Ollama-specific settings."""
        if CONFIG_AVAILABLE:
            return LLMSettings.OLLAMA
        return {
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3.1:8b",
            "default_temperature": 0.1,
            "default_max_tokens": 2000,
            "default_system_message": "You are a helpful assistant that extracts structured information from text.",
            "request_timeout": 60,
        }
    
    @property
    def openai_settings(self) -> Dict[str, Any]:
        """Get OpenAI-specific settings."""
        if CONFIG_AVAILABLE:
            return LLMSettings.OPENAI
        return {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "default_temperature": 0.1,
            "default_max_tokens": 2000,
            "default_system_message": "You are a helpful assistant that extracts structured information from text.",
            "request_timeout": 60,
        }
    
    @property
    def azure_openai_settings(self) -> Dict[str, Any]:
        """Get Azure OpenAI-specific settings."""
        if CONFIG_AVAILABLE:
            return LLMSettings.AZURE_OPENAI
        return {
            "api_key": "",
            "endpoint": "",
            "api_version": "2023-05-15",
            "default_model": "gpt-4o-mini",
            "default_temperature": 0.1,
            "default_max_tokens": 2000,
            "request_timeout": 60,
        }
    
    @property
    def test_settings(self) -> Dict[str, Any]:
        """Get testing-specific settings."""
        if CONFIG_AVAILABLE:
            return LLMSettings.TESTING
        return {
            "run_llm_tests": False,
            "run_ollama_tests": False,
        }
    
    @property
    def venv_settings(self) -> Dict[str, Any]:
        """Get virtual environment settings."""
        if CONFIG_AVAILABLE:
            return {
                "base_dir": str(VenvSettings.VENV_BASE_DIR),
                "info_file": str(VenvSettings.VENV_INFO_FILE),
                "default_venvs": VenvSettings.DEFAULT_VENVS,
                "generator_venv": self._generator_settings.get("default_venv", "venv_ollama")
            }
        return {
            "base_dir": str(PROJECT_ROOT / "the_venvs"),
            "generator_venv": "venv_ollama"
        }
    
    @property
    def generator_settings(self) -> Dict[str, Any]:
        """Get generator-specific settings."""
        return self._generator_settings
    
    def get_venv_python_path(self, venv_name: Optional[str] = None, generator_type: Optional[str] = None) -> Path:
        """Get the Python executable path for a virtual environment."""
        if venv_name is None:
            if generator_type == "openai":
                venv_name = self._generator_settings.get("openai_venv", "venv_main")
            else:
                venv_name = self.venv_settings.get("generator_venv", "venv_ollama")
        
        venv_base = Path(self.venv_settings["base_dir"])
        return venv_base / venv_name / "bin" / "python"
    
    def get_test_dependencies(self) -> list:
        """Get the list of test dependencies."""
        return self._generator_settings.get("test_dependencies", [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0"
        ])
    
    def is_testing_enabled(self) -> bool:
        """Check if testing is enabled."""
        return self.test_settings.get("run_llm_tests", False)

# Global instance
generator_config = GeneratorConfig()

# Convenience functions for backward compatibility
def get_ollama_settings():
    """Get Ollama settings."""
    return generator_config.ollama_settings

def get_openai_settings():
    """Get OpenAI settings."""
    return generator_config.openai_settings

def get_azure_openai_settings():
    """Get Azure OpenAI settings."""
    return generator_config.azure_openai_settings

def get_venv_python_path(venv_name: Optional[str] = None, generator_type: Optional[str] = None):
    """Get virtual environment Python path."""
    return generator_config.get_venv_python_path(venv_name, generator_type)

def get_test_dependencies():
    """Get test dependencies."""
    return generator_config.get_test_dependencies()

# Settings validation
def validate_generator_config():
    """Validate generator configuration."""
    issues = []
    
    # Check if config system is available
    if not CONFIG_AVAILABLE:
        issues.append("Config system not available - using defaults")
    
    # Check virtual environment paths
    venv_python = generator_config.get_venv_python_path()
    if not venv_python.exists():
        issues.append(f"Generator virtual environment not found: {venv_python}")
    
    # Check OpenAI venv if different
    openai_venv_python = generator_config.get_venv_python_path(generator_type="openai")
    if openai_venv_python != venv_python and not openai_venv_python.exists():
        issues.append(f"OpenAI virtual environment not found: {openai_venv_python}")
    
    # Check API keys for enabled services
    if generator_config.is_testing_enabled():
        openai_key = generator_config.openai_settings.get("api_key")
        if not openai_key:
            issues.append("OpenAI API key not set but testing is enabled")
    
    return issues

if __name__ == "__main__":
    # Test the configuration
    print("Generator Configuration Test")
    print("=" * 40)
    
    config = GeneratorConfig()
    
    print(f"Ollama model: {config.ollama_settings['default_model']}")
    print(f"OpenAI model: {config.openai_settings['default_model']}")
    print(f"Venv path: {config.get_venv_python_path()}")
    print(f"Test dependencies: {len(config.get_test_dependencies())}")
    
    # Validate configuration
    issues = validate_generator_config()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ Configuration validation passed")
