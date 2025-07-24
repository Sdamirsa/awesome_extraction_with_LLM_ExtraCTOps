"""
Application-specific configuration profiles for different deployment scenarios.
Use this file to define configuration profiles for development, testing, and production.
"""

from typing import Dict, Any
import os
from pathlib import Path

# Base configuration that applies to all profiles
BASE_CONFIG = {
    "app_name": "ExtraCTOps",
    "version": "1.0.0",
    "author": "Seyed Amir Ahmad Safavi-Naini",
    "license": "MIT",
    "description": "LLM-powered structured data extraction toolkit",
}

# Development configuration
DEVELOPMENT_CONFIG = {
    **BASE_CONFIG,
    "debug": True,
    "log_level": "DEBUG",
    "auto_reload": True,
    "database": {
        "url": "sqlite:///./dev_extractops.db",
        "echo": True,  # SQL query logging
    },
    "llm": {
        "rate_limiting": False,
        "cache_responses": True,
        "mock_api_calls": False,
    },
    "file_limits": {
        "max_file_size_mb": 50,
        "max_concurrent_uploads": 5,
    },
    "performance": {
        "request_timeout": 30,
        "max_concurrent_requests": 5,
        "batch_size": 10,
    }
}

# Testing configuration
TESTING_CONFIG = {
    **BASE_CONFIG,
    "debug": True,
    "log_level": "WARNING",
    "database": {
        "url": "sqlite:///:memory:",  # In-memory database for tests
        "echo": False,
    },
    "llm": {
        "rate_limiting": False,
        "cache_responses": False,
        "mock_api_calls": True,  # Use mocked responses
    },
    "file_limits": {
        "max_file_size_mb": 10,
        "max_concurrent_uploads": 2,
    },
    "performance": {
        "request_timeout": 10,
        "max_concurrent_requests": 2,
        "batch_size": 5,
    }
}

# Production configuration
PRODUCTION_CONFIG = {
    **BASE_CONFIG,
    "debug": False,
    "log_level": "INFO",
    "database": {
        "url": os.environ.get("PRODUCTION_DATABASE_URL", "postgresql://user:pass@localhost/extractops"),
        "echo": False,
        "pool_size": 20,
        "max_overflow": 30,
    },
    "llm": {
        "rate_limiting": True,
        "cache_responses": True,
        "mock_api_calls": False,
    },
    "file_limits": {
        "max_file_size_mb": 100,
        "max_concurrent_uploads": 20,
    },
    "performance": {
        "request_timeout": 60,
        "max_concurrent_requests": 50,
        "batch_size": 100,
    },
    "security": {
        "require_https": True,
        "enable_cors": False,
        "session_cookie_secure": True,
    }
}

# Staging configuration (similar to production but with more logging)
STAGING_CONFIG = {
    **PRODUCTION_CONFIG,
    "debug": False,
    "log_level": "DEBUG",
    "database": {
        **PRODUCTION_CONFIG["database"],
        "url": os.environ.get("STAGING_DATABASE_URL", "postgresql://user:pass@staging/extractops"),
        "echo": True,
    },
    "security": {
        **PRODUCTION_CONFIG["security"],
        "require_https": False,  # Allow HTTP in staging
        "enable_cors": True,     # Allow CORS for testing
    }
}

# Configuration profiles mapping
PROFILES = {
    "development": DEVELOPMENT_CONFIG,
    "dev": DEVELOPMENT_CONFIG,
    "testing": TESTING_CONFIG,
    "test": TESTING_CONFIG,
    "staging": STAGING_CONFIG,
    "production": PRODUCTION_CONFIG,
    "prod": PRODUCTION_CONFIG,
}

def get_config_profile(profile_name: str = None) -> Dict[str, Any]:
    """
    Get configuration for a specific profile.
    
    Args:
        profile_name: Name of the configuration profile.
                     If None, uses EXTRACTOPS_PROFILE env var or defaults to 'development'
    
    Returns:
        Configuration dictionary for the specified profile
    """
    if profile_name is None:
        profile_name = os.environ.get("EXTRACTOPS_PROFILE", "development")
    
    profile_name = profile_name.lower()
    
    if profile_name not in PROFILES:
        available_profiles = list(PROFILES.keys())
        raise ValueError(
            f"Unknown configuration profile: {profile_name}. "
            f"Available profiles: {available_profiles}"
        )
    
    return PROFILES[profile_name]

def get_current_profile_name() -> str:
    """Get the name of the currently active configuration profile."""
    return os.environ.get("EXTRACTOPS_PROFILE", "development").lower()

def set_profile(profile_name: str):
    """Set the active configuration profile via environment variable."""
    if profile_name.lower() not in PROFILES:
        available_profiles = list(PROFILES.keys())
        raise ValueError(
            f"Unknown configuration profile: {profile_name}. "
            f"Available profiles: {available_profiles}"
        )
    
    os.environ["EXTRACTOPS_PROFILE"] = profile_name.lower()

def print_profile_info(profile_name: str = None):
    """Print information about a configuration profile."""
    if profile_name is None:
        profile_name = get_current_profile_name()
    
    config = get_config_profile(profile_name)
    
    print(f"Configuration Profile: {profile_name.upper()}")
    print("=" * 50)
    print(f"App Name: {config['app_name']}")
    print(f"Version: {config['version']}")
    print(f"Debug Mode: {config.get('debug', False)}")
    print(f"Log Level: {config.get('log_level', 'INFO')}")
    
    if 'database' in config:
        print(f"Database: {config['database'].get('url', 'Not configured')}")
    
    if 'performance' in config:
        perf = config['performance']
        print(f"Request Timeout: {perf.get('request_timeout', 'Not set')}s")
        print(f"Max Concurrent Requests: {perf.get('max_concurrent_requests', 'Not set')}")
        print(f"Batch Size: {perf.get('batch_size', 'Not set')}")
    
    print("=" * 50)

# Convenience function to get current active configuration
def get_active_config() -> Dict[str, Any]:
    """Get the currently active configuration profile."""
    return get_config_profile()

# Example usage function
def example_usage():
    """Example of how to use configuration profiles."""
    print("Configuration Profiles Example")
    print("=" * 40)
    
    # Set development profile
    set_profile("development")
    dev_config = get_active_config()
    print(f"Development Debug Mode: {dev_config.get('debug', False)}")
    
    # Set production profile
    set_profile("production")
    prod_config = get_active_config()
    print(f"Production Debug Mode: {prod_config.get('debug', False)}")
    
    # Print current profile info
    print_profile_info()

if __name__ == "__main__":
    example_usage()
