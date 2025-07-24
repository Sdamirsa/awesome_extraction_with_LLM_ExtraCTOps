"""
Centralized configuration module for the ExtraCTOps project.
This file contains all modifiable parameters in one place to make configuration easier.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Project paths
PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_DIR = PROJECT_ROOT / os.environ.get("DATA_STORAGE_PATH", "data")
MODELS_DIR = PROJECT_ROOT / os.environ.get("MODELS_STORAGE_PATH", "models")
LOGS_DIR = PROJECT_ROOT / os.environ.get("LOGS_STORAGE_PATH", "logs")
UPLOADS_DIR = PROJECT_ROOT / os.environ.get("UPLOADS_PATH", "uploads")
EXPORTS_DIR = PROJECT_ROOT / os.environ.get("EXPORTS_PATH", "exports")
TEMP_DIR = PROJECT_ROOT / os.environ.get("TEMP_PATH", "temp")

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR, EXPORTS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Development settings
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
DEVELOPMENT_MODE = os.environ.get("DEVELOPMENT_MODE", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "extractops.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance settings
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "60"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "100"))
BATCH_PROCESSING_SIZE = int(os.environ.get("BATCH_PROCESSING_SIZE", "50"))

# Security settings
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY", "your-encryption-key-change-in-production")
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))

# Database configuration
class DatabaseSettings:
    """Database configuration settings."""
    URL = os.environ.get("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/extractops.db")
    HOST = os.environ.get("DATABASE_HOST", "localhost")
    PORT = int(os.environ.get("DATABASE_PORT", "5432"))
    NAME = os.environ.get("DATABASE_NAME", "extractops")
    USER = os.environ.get("DATABASE_USER", "")
    PASSWORD = os.environ.get("DATABASE_PASSWORD", "")

# External services configuration
class ExternalServices:
    """Configuration for external services."""
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

# LLM parameters
class LLMSettings:
    """Configuration settings for LLM generators."""
    
    # OpenAI settings
    OPENAI = {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "default_model": "gpt-4o-mini",
        "default_temperature": 0.1,
        "default_max_tokens": 2000,
        "default_system_message": "You are a helpful assistant that extracts structured information from text.",
        "request_timeout": REQUEST_TIMEOUT,
    }
    
    # Azure OpenAI settings
    AZURE_OPENAI = {
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        "default_model": "gpt-4o-mini",
        "default_temperature": 0.1,
        "default_max_tokens": 2000,
        "request_timeout": REQUEST_TIMEOUT,
    }
    
    # Ollama settings
    OLLAMA = {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "default_model": os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.1:8b"),
        "default_temperature": 0.1,
        "default_max_tokens": 2000,
        "default_system_message": "You are a helpful assistant that extracts structured information from text.",
        "request_timeout": REQUEST_TIMEOUT,
    }
    
    # Testing settings
    TESTING = {
        "run_llm_tests": os.environ.get("RUN_LLM_TESTS", "false").lower() == "true",
        "run_ollama_tests": os.environ.get("OLLAMA_TEST", "false").lower() == "true",
    }

# File handling settings
class FileSettings:
    """File handling and processing settings."""
    
    # Supported file types
    SUPPORTED_TEXT_EXTENSIONS = [".txt", ".md", ".py", ".json", ".csv"]
    SUPPORTED_DOC_EXTENSIONS = [".pdf", ".docx", ".doc"]
    SUPPORTED_DATA_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]
    
    # Processing limits
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_TEXT_LENGTH = 100000  # characters
    
    # Default file locations
    DEFAULT_UPLOAD_PATH = UPLOADS_DIR
    DEFAULT_EXPORT_PATH = EXPORTS_DIR
    DEFAULT_TEMP_PATH = TEMP_DIR

# Application-specific settings
class AppSettings:
    """Settings specific to different applications in the project."""
    
    # Manual extraction app
    MANUAL_EXTRACTION = {
        "dashboard_height": 600,
        "default_text_column": "text",
        "auto_save_interval": 300,  # seconds
        "max_extractions_per_session": 1000,
    }
    
    # Conversational Pydantic generator
    CONVERSATIONAL_GENERATOR = {
        "max_conversation_history": 50,
        "auto_backup_interval": 600,  # seconds
        "default_pricing": {
            "gpt-4o": {"prompt": 5000.0, "completion": 15000.0},
            "gpt-4o-mini": {"prompt": 1500.0, "completion": 5000.0},
            "gpt-4": {"prompt": 10000.0, "completion": 30000.0},
            "gpt-3.5-turbo": {"prompt": 500.0, "completion": 1500.0},
            "default": {"prompt": 1000.0, "completion": 2000.0}
        }
    }
    
    # Data to Pydantic mapping
    DATA_TO_PYDANTIC = {
        "max_rows_preview": 100,
        "default_sample_size": 50,
        "supported_formats": ["csv", "xlsx", "json"],
    }
    
    # Generator settings
    GENERATORS = {
        "default_venv": "venv_ollama",
        "test_dependencies": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.0.0", "pytest-mock>=3.10.0"],
        "supported_generators": ["ollama", "openai", "azure_openai"],
        "default_output_format": "json",
        "max_retries": 3,
        "default_timeout": 60,
        "openai_venv": "venv_main",  # OpenAI uses main venv or can share with ollama
        "preferred_generator": "ollama",  # Default generator to use
    }

# Virtual environment settings
class VenvSettings:
    """Virtual environment management settings."""
    
    VENV_BASE_DIR = PROJECT_ROOT / "the_venvs"
    VENV_INFO_FILE = PROJECT_ROOT / "config" / "venv_info.json"
    
    # Default venv configurations
    DEFAULT_VENVS = {
        "main": {
            "path": "venv_main",
            "requirements": "requirements_venv_main.txt",
            "python_version": "3.12"
        },
        "ollama": {
            "path": "venv_ollama", 
            "requirements": "requirements_venv_ollama.txt",
            "python_version": "3.12"
        },
        "streamlit": {
            "path": "venv_streamlit",
            "requirements": "venv_streamlit.txt",
            "python_version": "3.12"
        }
    }

# Environment overrides function (enhanced)
def get_env_override(prefix: str, settings_dict: dict) -> dict:
    """Override settings with environment variables."""
    result = settings_dict.copy()
    for key in settings_dict:
        env_var = f"{prefix}_{key}".upper()
        if env_var in os.environ:
            # Convert environment variable value to appropriate type
            env_value = os.environ[env_var]
            if isinstance(settings_dict[key], bool):
                result[key] = env_value.lower() == "true"
            elif isinstance(settings_dict[key], int):
                result[key] = int(env_value)
            elif isinstance(settings_dict[key], float):
                result[key] = float(env_value)
            else:
                result[key] = env_value
    return result

# Apply environment overrides
for llm_provider in [LLMSettings.OPENAI, LLMSettings.AZURE_OPENAI, LLMSettings.OLLAMA, LLMSettings.TESTING]:
    if isinstance(llm_provider, dict):
        provider_name = [k for k, v in LLMSettings.__dict__.items() if v is llm_provider][0]
        llm_provider.update(get_env_override(f"LLM_{provider_name}", llm_provider))

# Validation functions
def validate_configuration() -> List[str]:
    """Validate the current configuration and return any warnings or errors."""
    warnings = []
    
    # Check if required directories exist
    required_dirs = [DATA_DIR, MODELS_DIR, LOGS_DIR]
    for directory in required_dirs:
        if not directory.exists():
            warnings.append(f"Required directory does not exist: {directory}")
    
    # Check API keys for enabled services
    if LLMSettings.TESTING["run_llm_tests"]:
        if not LLMSettings.OPENAI["api_key"]:
            warnings.append("OpenAI API key not set, but LLM tests are enabled")
        if not LLMSettings.AZURE_OPENAI["api_key"]:
            warnings.append("Azure OpenAI API key not set, but LLM tests are enabled")
    
    # Check file size limits
    if MAX_FILE_SIZE_MB > 500:
        warnings.append(f"Large file size limit set: {MAX_FILE_SIZE_MB}MB - may cause memory issues")
    
    return warnings

def get_all_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        "project_info": {
            "project_root": str(PROJECT_ROOT),
            "debug": DEBUG,
            "development_mode": DEVELOPMENT_MODE,
        },
        "directories": {
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "logs_dir": str(LOGS_DIR),
            "uploads_dir": str(UPLOADS_DIR),
            "exports_dir": str(EXPORTS_DIR),
            "temp_dir": str(TEMP_DIR),
        },
        "logging": {
            "log_level": LOG_LEVEL,
            "log_file": str(LOG_FILE),
            "log_format": LOG_FORMAT,
            "log_date_format": LOG_DATE_FORMAT,
        },
        "performance": {
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "request_timeout": REQUEST_TIMEOUT,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "batch_processing_size": BATCH_PROCESSING_SIZE,
        },
        "security": {
            "session_timeout": SESSION_TIMEOUT,
            # Note: Don't include actual keys in output for security
            "secret_key_set": bool(SECRET_KEY),
            "encryption_key_set": bool(ENCRYPTION_KEY),
        },
        "database": {
            "url": DatabaseSettings.URL,
            "host": DatabaseSettings.HOST,
            "port": DatabaseSettings.PORT,
            "name": DatabaseSettings.NAME,
        },
        "external_services": {
            "redis_url": ExternalServices.REDIS_URL,
            "elasticsearch_url": ExternalServices.ELASTICSEARCH_URL,
        },
        "llm": {
            "openai": {k: v for k, v in LLMSettings.OPENAI.items() if k != "api_key"},
            "azure_openai": {k: v for k, v in LLMSettings.AZURE_OPENAI.items() if k != "api_key"},
            "ollama": LLMSettings.OLLAMA,
            "testing": LLMSettings.TESTING,
        },
        "file_settings": {
            "supported_extensions": {
                "text": FileSettings.SUPPORTED_TEXT_EXTENSIONS,
                "documents": FileSettings.SUPPORTED_DOC_EXTENSIONS,
                "data": FileSettings.SUPPORTED_DATA_EXTENSIONS,
            },
            "limits": {
                "max_file_size_bytes": FileSettings.MAX_FILE_SIZE_BYTES,
                "max_text_length": FileSettings.MAX_TEXT_LENGTH,
            }
        },
        "app_settings": {
            "manual_extraction": AppSettings.MANUAL_EXTRACTION,
            "conversational_generator": AppSettings.CONVERSATIONAL_GENERATOR,
            "data_to_pydantic": AppSettings.DATA_TO_PYDANTIC,
            "generators": AppSettings.GENERATORS,
        },
        "venv_settings": {
            "base_dir": str(VenvSettings.VENV_BASE_DIR),
            "info_file": str(VenvSettings.VENV_INFO_FILE),
            "default_venvs": VenvSettings.DEFAULT_VENVS,
        }
    }

# Quick access functions
def get_data_dir() -> Path:
    """Get the data directory path."""
    return DATA_DIR

def get_models_dir() -> Path:
    """Get the models directory path.""" 
    return MODELS_DIR

def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return LOGS_DIR

def get_uploads_dir() -> Path:
    """Get the uploads directory path."""
    return UPLOADS_DIR

def get_exports_dir() -> Path:
    """Get the exports directory path."""
    return EXPORTS_DIR

def get_temp_dir() -> Path:
    """Get the temporary files directory path."""
    return TEMP_DIR

# Configuration summary for debugging
def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 50)
    print("ExtraCTOps Configuration Summary")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Development Mode: {DEVELOPMENT_MODE}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Max File Size: {MAX_FILE_SIZE_MB}MB")
    print(f"Request Timeout: {REQUEST_TIMEOUT}s")
    
    # Check for warnings
    warnings = validate_configuration()
    if warnings:
        print("\nConfiguration Warnings:")
        for warning in warnings:
            print(f"⚠️  {warning}")
    else:
        print("\n✅ Configuration validation passed")
    print("=" * 50)

# Run validation when module is imported (only in development)
if DEVELOPMENT_MODE:
    warnings = validate_configuration()
    if warnings:
        print("Configuration warnings detected:")
        for warning in warnings:
            print(f"⚠️  {warning}")
