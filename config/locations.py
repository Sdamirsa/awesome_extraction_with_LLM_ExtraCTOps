"""
Location and path management for the ExtraCTOps project.
This module provides centralized management of all file paths, URLs, and locations used throughout the project.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

class ProjectLocations:
    """Central management of all project locations and paths."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize project locations.
        
        Args:
            project_root: Override for project root directory. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (assuming this file is in config/)
            self.project_root = Path(__file__).parents[1].resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self._init_directories()
        self._init_file_patterns()
        self._init_urls()
    
    def _init_directories(self):
        """Initialize all directory paths."""
        # Core directories
        self.config_dir = self.project_root / "config"
        self.utils_dir = self.project_root / "utils"
        self.generators_dir = self.project_root / "generators"
        self.evaluators_dir = self.project_root / "evaluators"
        self.internal_models_dir = self.project_root / "internal_models"
        self.the_pydantics_dir = self.project_root / "the_pydantics"
        self.tests_dir = self.project_root / "tests"
        
        # Data directories
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.uploads_dir = self.project_root / "uploads"
        self.exports_dir = self.project_root / "exports"
        self.temp_dir = self.project_root / "temp"
        
        # Virtual environments
        self.venvs_dir = self.project_root / "the_venvs"
        self.venv_main = self.venvs_dir / "venv_main"
        self.venv_ollama = self.venvs_dir / "venv_ollama"
        self.venv_streamlit = self.venvs_dir / "venv_streamlit"
        
        # Applications
        self.apps_dir = self.project_root / "apps"
        self.manual_extraction_dir = self.apps_dir / "manual_extraction"
        self.conversational_generator_dir = self.apps_dir / "conversational_pydantic_generator"
        self.data_to_pydantic_dir = self.apps_dir / "Data2Pydantic_Map"
        self.manual_evaluation_dir = self.apps_dir / "manual_evaluation"
        self.easy_start_dir = self.apps_dir / "easy_start"
        
        # Third party and licenses
        self.third_party_dir = self.project_root / "third_party_licenses"
        self.report_performance_dir = self.project_root / "report_performance"
        
        # Log subdirectories
        self.inference_logs_dir = self.utils_dir / "inference_logs"
        self.utils_logs_dir = self.utils_dir / "logs"
    
    def _init_file_patterns(self):
        """Initialize common file patterns and naming conventions."""
        self.file_patterns = {
            # Configuration files
            "env_file": self.project_root / ".env",
            "env_example": self.project_root / ".env.example",
            "gitignore": self.project_root / ".gitignore",
            "readme": self.project_root / "README.md",
            "license": self.project_root / "LICENSE",
            "coding_standard": self.project_root / "Coding_standard.md",
            
            # Virtual environment files
            "venv_info": self.config_dir / "venv_info.json",
            "requirements_main": self.venvs_dir / "requirements_venv_main.txt",
            "requirements_ollama": self.venvs_dir / "requirements_venv_ollama.txt",
            "requirements_streamlit": self.venvs_dir / "venv_streamlit.txt",
            
            # Configuration files
            "settings": self.config_dir / "settings.py",
            "profiles": self.config_dir / "profiles.py",
            "config_helper": self.config_dir / "config_helper.py",
            
            # Application entry points
            "manual_extraction_app": self.manual_extraction_dir / "app.py",
            "conversational_app": self.conversational_generator_dir / "app.py",
            "data_to_pydantic_app": self.data_to_pydantic_dir / "app.py",
            "manual_evaluation_app": self.manual_evaluation_dir / "app.py",
            
            # Utility scripts
            "manage_venvs_python": self.easy_start_dir / "manage_venvs.py",
            "run_streamlit": self.easy_start_dir / "run_streamlit_app.py",
            
            # Generator files
            "generator_ollama": self.generators_dir / "generator_Ollama.py",
            "generator_models": self.generators_dir / "generators_models.py",
            "utils_messages": self.generators_dir / "utils_messages.py",
            "utils_pydantic": self.generators_dir / "utils_pydantic.py",
            
            # Test files
            "test_generator_ollama": self.tests_dir / "test_generator_ollama.py",
            
            # Logger utility
            "logger": self.utils_dir / "the_logger.py",
            
            # Pydantic schemas
            "echo_report": self.the_pydantics_dir / "EchoReport.py",
            "mre_schema": self.the_pydantics_dir / "MRE_schema.py",
            "example_schema": self.the_pydantics_dir / "example_schema.py",
            "example_data": self.the_pydantics_dir / "example_MRE_data.xlsx",
        }
    
    def _init_urls(self):
        """Initialize URL patterns and endpoints."""
        self.urls = {
            # Local development URLs
            "localhost_streamlit": "http://localhost:8501",
            "localhost_ollama": "http://localhost:11434",
            "localhost_ollama_api": "http://localhost:11434/v1",
            
            # API endpoints
            "openai_base": "https://api.openai.com/v1",
            "azure_openai_base": "https://{resource}.openai.azure.com",
            
            # Documentation and repository
            "github_repo": "https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps",
            "documentation": "https://github.com/Sdamirsa/awesome_extraction_with_LLM_ExtraCTOps/blob/main/README.md",
        }
    
    def ensure_directories_exist(self, directories: Optional[List[str]] = None):
        """
        Ensure specified directories exist, creating them if necessary.
        
        Args:
            directories: List of directory names to create. If None, creates all core directories.
        """
        if directories is None:
            # Create all core directories
            directories_to_create = [
                self.data_dir, self.models_dir, self.logs_dir, self.uploads_dir,
                self.exports_dir, self.temp_dir, self.inference_logs_dir, self.utils_logs_dir
            ]
        else:
            directories_to_create = [getattr(self, f"{dir_name}_dir") for dir_name in directories if hasattr(self, f"{dir_name}_dir")]
        
        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_file_path(self, file_key: str) -> Path:
        """
        Get a file path by its key.
        
        Args:
            file_key: Key from the file_patterns dictionary
            
        Returns:
            Path object for the requested file
            
        Raises:
            KeyError: If the file key is not found
        """
        if file_key not in self.file_patterns:
            available_keys = list(self.file_patterns.keys())
            raise KeyError(f"File key '{file_key}' not found. Available keys: {available_keys}")
        
        return self.file_patterns[file_key]
    
    def get_url(self, url_key: str, **kwargs) -> str:
        """
        Get a URL by its key, with optional formatting.
        
        Args:
            url_key: Key from the urls dictionary
            **kwargs: Formatting parameters for the URL
            
        Returns:
            Formatted URL string
            
        Raises:
            KeyError: If the URL key is not found
        """
        if url_key not in self.urls:
            available_keys = list(self.urls.keys())
            raise KeyError(f"URL key '{url_key}' not found. Available keys: {available_keys}")
        
        url_template = self.urls[url_key]
        
        if kwargs:
            return url_template.format(**kwargs)
        
        return url_template
    
    def get_app_path(self, app_name: str) -> Path:
        """
        Get the path to an application directory.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Path to the application directory
        """
        app_mapping = {
            "manual_extraction": self.manual_extraction_dir,
            "conversational_generator": self.conversational_generator_dir,
            "conversational_pydantic_generator": self.conversational_generator_dir,
            "data_to_pydantic": self.data_to_pydantic_dir,
            "data2pydantic_map": self.data_to_pydantic_dir,
            "manual_evaluation": self.manual_evaluation_dir,
            "easy_start": self.easy_start_dir,
        }
        
        if app_name.lower() not in app_mapping:
            available_apps = list(app_mapping.keys())
            raise ValueError(f"Unknown app '{app_name}'. Available apps: {available_apps}")
        
        return app_mapping[app_name.lower()]
    
    def get_venv_path(self, venv_name: str) -> Path:
        """
        Get the path to a virtual environment.
        
        Args:
            venv_name: Name of the virtual environment
            
        Returns:
            Path to the virtual environment
        """
        venv_mapping = {
            "main": self.venv_main,
            "ollama": self.venv_ollama,
            "streamlit": self.venv_streamlit,
        }
        
        if venv_name.lower() not in venv_mapping:
            available_venvs = list(venv_mapping.keys())
            raise ValueError(f"Unknown venv '{venv_name}'. Available venvs: {available_venvs}")
        
        return venv_mapping[venv_name.lower()]
    
    def get_python_executable(self, venv_name: str) -> Path:
        """
        Get the Python executable path for a specific virtual environment.
        
        Args:
            venv_name: Name of the virtual environment
            
        Returns:
            Path to the Python executable
        """
        venv_path = self.get_venv_path(venv_name)
        
        # Check platform to determine the correct executable path
        if os.name == 'nt':  # Windows
            python_path = venv_path / "Scripts" / "python.exe"
        else:  # Unix-like (macOS, Linux)
            python_path = venv_path / "bin" / "python"
        
        return python_path
    
    def list_available_files(self) -> List[str]:
        """Get a list of all available file keys."""
        return list(self.file_patterns.keys())
    
    def list_available_urls(self) -> List[str]:
        """Get a list of all available URL keys."""
        return list(self.urls.keys())
    
    def print_summary(self):
        """Print a summary of all configured locations."""
        print("ExtraCTOps Project Locations Summary")
        print("=" * 50)
        print(f"Project Root: {self.project_root}")
        print()
        
        print("Core Directories:")
        print(f"  Data: {self.data_dir}")
        print(f"  Models: {self.models_dir}")
        print(f"  Logs: {self.logs_dir}")
        print(f"  Uploads: {self.uploads_dir}")
        print(f"  Exports: {self.exports_dir}")
        print(f"  Temp: {self.temp_dir}")
        print()
        
        print("Virtual Environments:")
        print(f"  Main: {self.venv_main}")
        print(f"  Ollama: {self.venv_ollama}")
        print(f"  Streamlit: {self.venv_streamlit}")
        print()
        
        print("Applications:")
        print(f"  Manual Extraction: {self.manual_extraction_dir}")
        print(f"  Conversational Generator: {self.conversational_generator_dir}")
        print(f"  Data to Pydantic: {self.data_to_pydantic_dir}")
        print(f"  Manual Evaluation: {self.manual_evaluation_dir}")
        print()
        
        print("Key URLs:")
        print(f"  Streamlit: {self.urls['localhost_streamlit']}")
        print(f"  Ollama API: {self.urls['localhost_ollama_api']}")
        print(f"  GitHub: {self.urls['github_repo']}")
        print("=" * 50)

# Global instance for easy access
locations = ProjectLocations()

# Convenience functions for common paths
def get_data_dir() -> Path:
    """Get the data directory path."""
    return locations.data_dir

def get_models_dir() -> Path:
    """Get the models directory path."""
    return locations.models_dir

def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return locations.logs_dir

def get_uploads_dir() -> Path:
    """Get the uploads directory path."""
    return locations.uploads_dir

def get_exports_dir() -> Path:
    """Get the exports directory path."""
    return locations.exports_dir

def get_temp_dir() -> Path:
    """Get the temporary directory path."""
    return locations.temp_dir

def get_project_root() -> Path:
    """Get the project root directory path."""
    return locations.project_root

def get_app_path(app_name: str) -> Path:
    """Get the path to an application directory."""
    return locations.get_app_path(app_name)

def get_venv_python(venv_name: str) -> Path:
    """Get the Python executable for a virtual environment."""
    return locations.get_python_executable(venv_name)

def ensure_data_directories():
    """Ensure all data directories exist."""
    locations.ensure_directories_exist()

# Example usage
if __name__ == "__main__":
    locations.print_summary()
    print()
    print("Available file keys:", locations.list_available_files()[:10], "...")
    print("Available URL keys:", locations.list_available_urls())
