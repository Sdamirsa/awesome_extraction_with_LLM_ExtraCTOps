"""
Tests for the Ollama generator function using standard unittest module.

Served by: Direct call

Path to venv: "the_venvs/venv_ollama/bin/python"

Libraries: unittest, asyncio, pydantic, subprocess
"""

###################
####  imports  ####
###################
import subprocess
import sys
import os

libraries = [
    "pydantic",
    "asyncio"
]

for lib in libraries:
    try:
        __import__(lib.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

import asyncio
import unittest
from pydantic import BaseModel, Field
from typing import List, Optional

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

###################
####  logging  ####
###################
try:
    from utils import the_logger
    # Log the test initialization
    the_logger.info("Initializing Ollama generator tests")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    the_logger = logging.getLogger(__name__)
    the_logger.info("Using basic logger as utils.the_logger was not found")

# Import configuration settings
from config.settings import LLMSettings

# Import the functions to test
from generators.generator_Ollama import (
    generator_Ollama,
    extract_structured_data,
    extract_structured_data_sync
)
from internal_models.generators_models import MessageConfig, ModelConfig

###################################
####  1. Test Data and Models  ####
###################################

# Define a test Pydantic model
class TestPersonModel(BaseModel):
    """Test model for a person's information."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: Optional[str] = Field(None, description="The person's occupation")
    skills: List[str] = Field(default_factory=list, description="List of skills the person has")

# Sample test data
TEST_TEXT = """
John Smith is a 35-year-old software engineer who specializes in Python development.
His skills include Python, JavaScript, Docker, and cloud computing technologies.
"""

###################################
####  2. Test Case Definition  ####
###################################

class OllamaGeneratorTests(unittest.TestCase):
    """Test class for Ollama generator functions"""
    
    def setUp(self):
        """Set up test environment."""
        if not LLMSettings.TESTING["run_ollama_tests"]:
            self.skipTest("Skipping Ollama tests. Set OLLAMA_TEST=true to run.")
    
    async def async_test_generator_ollama(self):
        """Test the main async generator function."""
        # Create configuration objects
        message_config = MessageConfig(
            system_message="Extract structured information about a person from the text.",
            pre_prompt="Return a JSON object with the person's details according to the schema.",
            text=TEST_TEXT,
            pydantic_model=TestPersonModel
        )
        
        model_config = ModelConfig(
            model_name=LLMSettings.OLLAMA["default_model"],
            temperature=LLMSettings.OLLAMA["default_temperature"],
            max_tokens=LLMSettings.OLLAMA["default_max_tokens"]
        )
        
        # Call the generator function
        result = await generator_Ollama(message_config, model_config)
        
        # Verify the result structure
        self.assertTrue(result.generation_success)
        self.assertIsInstance(result.execution_time, float)
        self.assertGreater(result.execution_time, 0)
        
        # Check if parsing was successful
        if result.parsing_success:
            self.assertIsInstance(result.parsed_response, TestPersonModel)
            self.assertTrue(hasattr(result.parsed_response, "name"))
            self.assertTrue(hasattr(result.parsed_response, "age"))
            
            # Display the parsed output
            print("\n\n=== Parsed Output from generator_Ollama ===")
            print(f"Name: {result.parsed_response.name}")
            print(f"Age: {result.parsed_response.age}")
            print(f"Occupation: {result.parsed_response.occupation}")
            print(f"Skills: {', '.join(result.parsed_response.skills)}")
            print(f"Raw model: {result.parsed_response.model_dump_json(indent=2)}")
        else:
            print("\n\n=== Parsing Failed ===")
            print(f"Raw response: {result.raw_response}")
        
        # Verify raw response exists
        self.assertIsInstance(result.raw_response, str)
        self.assertGreater(len(result.raw_response), 0)
    
    async def async_test_extract_structured_data(self):
        """Test the simplified async extraction function."""
        parsed_model, raw_response, success = await extract_structured_data(
            text=TEST_TEXT,
            pydantic_model=TestPersonModel,
        )
        
        # Verify the result
        self.assertIsInstance(raw_response, str)
        self.assertGreater(len(raw_response), 0)
        
        # If parsing succeeded, verify the model structure
        if success:
            self.assertIsNotNone(parsed_model)
            self.assertIsInstance(parsed_model, TestPersonModel)
            self.assertTrue(hasattr(parsed_model, "name"))
            self.assertTrue(hasattr(parsed_model, "age"))
            
            # Display the parsed output
            print("\n\n=== Parsed Output from extract_structured_data ===")
            print(f"Name: {parsed_model.name}")
            print(f"Age: {parsed_model.age}")
            print(f"Occupation: {parsed_model.occupation}")
            print(f"Skills: {', '.join(parsed_model.skills)}")
            print(f"Raw model: {parsed_model.model_dump_json(indent=2)}")
        else:
            print("\n\n=== Parsing Failed ===")
            print(f"Raw response: {raw_response}")
    
    def test_extract_structured_data_sync(self):
        """Test the synchronous extraction function."""
        parsed_model, raw_response, success = extract_structured_data_sync(
            text=TEST_TEXT,
            pydantic_model=TestPersonModel,
            model_name=LLMSettings.OLLAMA["default_model"],
            temperature=LLMSettings.OLLAMA["default_temperature"]
        )
        
        # Verify the result
        self.assertIsInstance(raw_response, str)
        self.assertGreater(len(raw_response), 0)
        
        # If parsing succeeded, verify the model structure
        if success:
            self.assertIsNotNone(parsed_model)
            self.assertIsInstance(parsed_model, TestPersonModel)
            self.assertTrue(hasattr(parsed_model, "name"))
            self.assertTrue(hasattr(parsed_model, "age"))
            
            # Display the parsed output
            print("\n\n=== Parsed Output from extract_structured_data_sync ===")
            print(f"Name: {parsed_model.name}")
            print(f"Age: {parsed_model.age}")
            print(f"Occupation: {parsed_model.occupation}")
            print(f"Skills: {', '.join(parsed_model.skills)}")
            print(f"Raw model: {parsed_model.model_dump_json(indent=2)}")
        else:
            print("\n\n=== Parsing Failed ===")
            print(f"Raw response: {raw_response}")
    
    async def async_test_error_handling(self):
        """Test error handling with invalid configurations."""
        # Test with non-existent model
        message_config = MessageConfig(
            system_message="Extract information",
            text=TEST_TEXT,
            pydantic_model=TestPersonModel
        )
        
        model_config = ModelConfig(
            model_name="non_existent_model_123",
            temperature=LLMSettings.OLLAMA["default_temperature"]
        )
        
        result = await generator_Ollama(message_config, model_config)
        
        # Verify error handling
        self.assertFalse(result.generation_success)
        self.assertFalse(result.parsing_success)
        self.assertIsNotNone(result.error)

    def test_generator_ollama(self):
        """Runner for async generator test."""
        asyncio.run(self.async_test_generator_ollama())
    
    def test_extract_data(self):
        """Runner for async extract data test."""
        asyncio.run(self.async_test_extract_structured_data())
    
    def test_error_handling(self):
        """Runner for async error handling test."""
        asyncio.run(self.async_test_error_handling())

###################################
####  3. Subprocess Utilities  ####
###################################

def run_tests_with_custom_subprocess(venv_path=None, specific_test=None):
    """
    Run the tests using subprocess with an optional virtual environment.
    
    Args:
        venv_path (str, optional): Path to the virtual environment's Python executable.
                                  Default is None, which uses the current Python interpreter.
        specific_test (str, optional): Name of a specific test to run. Default is None, which runs all tests.
    
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    # Use the specified venv Python or the current Python interpreter
    python_executable = venv_path if venv_path else sys.executable
    
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Construct the command
    cmd = [python_executable, "-m", "unittest", __file__]
    
    # Add specific test if provided
    if specific_test:
        cmd[-1] = f"{__file__}.OllamaGeneratorTests.{specific_test}"
    
    # Run the tests
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        
        the_logger.info(f"Running tests with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        the_logger.info(f"Return code: {return_code}")
        the_logger.info(f"Output:\n{stdout}")
        
        if stderr:
            the_logger.warning(f"Errors:\n{stderr}")
            
        return return_code, stdout, stderr
        
    except Exception as e:
        the_logger.error(f"Error running tests: {e}")
        return 1, "", str(e)

###################################
####      Main function        ####
###################################

def run_all_tests():
    """Run all tests in the test class."""
    unittest.main(verbosity=2)


###################################
####  Example use in terminal  ####
###################################
"""
# Run all tests using the system Python
python tests/test_generator_ollama.py

# Run tests with specific virtual environment
python -c "from tests.test_generator_ollama import run_tests_with_custom_subprocess; run_tests_with_custom_subprocess(venv_path='the_venvs/venv_ollama/bin/python')"

# Run specific test with a virtual environment
python -c "from tests.test_generator_ollama import run_tests_with_custom_subprocess; run_tests_with_custom_subprocess(venv_path='the_venvs/venv_ollama/bin/python', specific_test='test_extract_structured_data_sync')"
"""


###################################
####  Example use in notebook  ####
###################################
"""
# Import the test file
from tests.test_generator_ollama import run_tests_with_custom_subprocess

# Run all tests
return_code, stdout, stderr = run_tests_with_custom_subprocess(
    venv_path="the_venvs/venv_ollama/bin/python"
)

# Run specific test
return_code, stdout, stderr = run_tests_with_custom_subprocess(
    venv_path="the_venvs/venv_ollama/bin/python",
    specific_test="test_extract_structured_data_sync"
)

print(f"Tests completed with return code {return_code}")
print(stdout)
"""

if __name__ == "__main__":
    # Run tests with a specific virtual environment
    run_tests_with_custom_subprocess(venv_path="the_venvs/venv_ollama/bin/python")