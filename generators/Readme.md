# ExtraCTOps Generators

This directory contains generator modules that extract structured data from unstructured text using various Large Language Models (LLMs). The generators are designed to be modular, extensible, and compatible with both synchronous and asynchronous workflows.

## Directory Structure

```
generators/
├── generator_ollama.py          # Ollama LLM generator
├── generator_openai.py          # OpenAI LLM generator
├── config.py                    # Configuration integration with central config system
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── utils_messages.py        # Message preparation utilities
│   └── utils_pydantic.py        # Pydantic model utilities  
├── tests/                       # Test modules
│   ├── __init__.py
│   ├── test_generator_ollama.py
│   ├── test_generator_openai.py
│   ├── test_utils_messages.py
│   ├── test_utils_pydantic.py
│   └── run_tests.py
├── __init__.py
├── pytest.ini                   # Test configuration
├── setup_tests.py               # Test environment setup
├── requirements_test.txt        # Test dependencies reference
└── Readme.md                    # This file
```

I'll create a comprehensive table for the generators with the columns you specified, plus a few additional useful columns.

## Generators Comparison Table
| Generator | Script | Specific venv | LLM | Parser | Provider | Async Support | Multimodal | Interface |
|-----------|--------|---------------|-----|--------|----------|--------------|------------|-----------|
| Ollama | generator_ollama.py | venv_ollama | Various (llama3, mixtral, etc.) | JSON parsing with fallbacks | Ollama (Local) | Yes | Yes (images) | Direct import, Sync/Async functions |
| OpenAI | generator_openai.py | venv_main | GPT models (gpt-4o, gpt-4o-mini, etc.) | JSON parsing with fallbacks | OpenAI (API) | Yes | Yes (images) | Direct import, Sync/Async functions |


## Overview

The generators receive:
- Text (required)
- A Pydantic model for extraction (required)
- Optional parameters such as images, prompt instructions, etc.

And using the model_config dictionary with (hyper)parameters:
- model_name
- temperature
- top_p
- top_k
- and other model-specific parameters

They produce:
- Structured data (JSON) matching the provided Pydantic model
- Validation results
- Performance metrics

## Architecture

Each generator follows a consistent pattern:
1. Accept standardized input configurations through `MessageConfig` and `ModelConfig`
2. Process text through an LLM
3. Parse and validate the response against the target Pydantic model
4. Return a standardized `GenerationResult` object

### Key Components

- **Data Models** - Shared configuration and result models in `generators_models.py`
- **Generator Modules** - LLM-specific implementations (Ollama, OpenAI, etc.)
- **Utility Modules** - Shared functionality for message preparation and response parsing

## Data Models

The generators use three primary data models from `generators_models.py`:

### MessageConfig

Configuration for the input message to the LLM:

```python
class MessageConfig(BaseModel):
    system_message: str                  # System instructions for the LLM
    pre_prompt: Optional[str] = None     # Instructions to prepend to text
    few_shot_json_path: Optional[str] = None  # Path to few-shot examples
    image_paths: Optional[List[str]] = None   # Paths to images for multimodal input
    text: str                            # Unstructured text for extraction
    pydantic_model: Union[str, Type[BaseModel]]  # Model for extraction
```

### ModelConfig

Configuration for the LLM parameters:

```python
class ModelConfig(BaseModel):
    model_name: str                      # Name of the LLM to use
    temperature: float = 0.2             # Temperature (0.0-1.0)
    max_tokens: Optional[int] = None     # Maximum tokens to generate
    top_p: Optional[float] = None        # Top-p sampling value
    top_k: Optional[int] = None          # Top-k sampling value
    seed: Optional[int] = None           # Random seed
    logprobs: Optional[int] = None       # Log probabilities
    stop: Optional[List[str]] = None     # Stop sequences
```

### GenerationResult

Standardized result from all generators:

```python
class GenerationResult(BaseModel):
    execution_time: float                # Time taken in seconds
    generation_success: bool             # Whether generation was successful
    parsing_success: bool                # Whether parsing was successful
    raw_response: Optional[str] = None   # Raw text response from LLM
    parsed_response: Optional[BaseModel] = None  # Structured data
    error: Optional[str] = None          # Error message if any
```

## Available Generators

- **generator_ollama.py** - Uses Ollama for local LLM generation
- **generator_openai.py** - Uses OpenAI API for cloud-based LLM generation
- *(Additional generators will be added for other LLM providers)*

## Configuration Integration

The generators package is fully integrated with the centralized configuration system located in the `config/` directory. This provides:

- **Centralized LLM settings** - All LLM provider configurations (Ollama, OpenAI, Azure) are managed centrally
- **Virtual environment management** - Automatic detection and use of the appropriate venv
- **Environment profiles** - Support for development, testing, staging, and production configurations
- **Configuration validation** - Built-in validation of generator settings and dependencies

### Using the Configuration System

```python
from generators.config import generator_config

# Get LLM settings
ollama_settings = generator_config.ollama_settings
openai_settings = generator_config.openai_settings

# Get virtual environment path
venv_python = generator_config.get_venv_python_path()

# Get test dependencies
test_deps = generator_config.get_test_dependencies()

# Validate configuration
from generators.config import validate_generator_config
issues = validate_generator_config()
```

### Configuration Management

Use the config helper to manage generator settings:

```bash
# View generator configuration
python config/config_helper.py generators

# Validate overall configuration
python config/config_helper.py validate

# Show virtual environment status
python config/config_helper.py venvs
```

## Utility Modules
```

- **utils/utils_messages.py** - Prepares input messages for LLMs, including multimodal capabilities
- **utils/utils_pydantic.py** - Handles conversion between Pydantic models and JSON schemas, and parsing LLM outputs

## Testing

The generators package includes comprehensive test coverage using the same virtual environment as the generators (`venv_ollama`).

### Setting Up Tests

First, set up the test environment (this installs pytest and related packages in the Ollama venv):

```bash
cd generators
python setup_tests.py
```

### Running Tests

```bash
# Run all tests using the Ollama venv
python tests/run_tests.py

# Or manually with the venv Python
../the_venvs/venv_ollama/bin/python -m pytest tests/ -v

# Run specific test file
../the_venvs/venv_ollama/bin/python -m pytest tests/test_generator_ollama.py -v

# Run tests with coverage
../the_venvs/venv_ollama/bin/python -m pytest tests/ --cov=generators --cov-report=html
```

### Test Structure

- `test_generator_ollama.py` - Tests for the Ollama generator
- `test_utils_messages.py` - Tests for message utility functions  
- `test_utils_pydantic.py` - Tests for Pydantic utility functions
- `run_tests.py` - Test runner script that uses the Ollama venv
- `setup_tests.py` - Setup script for installing test dependencies

All tests use pytest and include both unit tests and integration tests with mocked dependencies. The tests run in the same virtual environment (`venv_ollama`) as the generators to ensure compatibility.

## Usage Examples

### Basic Usage with Ollama

```python
import asyncio
from generators.generator_ollama import generator_Ollama
from internal_models import MessageConfig, ModelConfig
from the_pydantics.example_schema import ExampleModel

async def extract_with_ollama():
    message_config = MessageConfig(
        system_message="Extract structured data from the following text.",
        pre_prompt="Return information as JSON according to the schema.",
        text="The field1 value is 'sample text' and field2 is 42.",
        pydantic_model=ExampleModel
    )
    
    model_config = ModelConfig(
        model_name="llama3", 
        temperature=0.1,
        max_tokens=1000
    )
    
    result = await generator_Ollama(message_config, model_config)
    
    if result.parsing_success:
        print(f"Successfully extracted: {result.parsed_response}")
    else:
        print(f"Extraction failed. Raw response: {result.raw_response}")

asyncio.run(extract_with_ollama())
```

### Basic Usage with OpenAI

```python
import asyncio
from generators.generator_openai import generator_OpenAI
from internal_models import MessageConfig, ModelConfig
from the_pydantics.example_schema import ExampleModel

async def extract_with_openai():
    message_config = MessageConfig(
        system_message="Extract structured data from the following text.",
        pre_prompt="Return information as JSON according to the schema.",
        text="The field1 value is 'sample text' and field2 is 42.",
        pydantic_model=ExampleModel
    )
    
    model_config = ModelConfig(
        model_name="gpt-4o-mini", 
        temperature=0.1,
        max_tokens=1000
    )
    
    result = await generator_OpenAI(message_config, model_config)
    
    if result.parsing_success:
        print(f"Successfully extracted: {result.parsed_response}")
    else:
        print(f"Extraction failed. Raw response: {result.raw_response}")

asyncio.run(extract_with_openai())
```

### Switching Between Generators

```python
import asyncio
from generators.generator_ollama import generator_Ollama  
from generators.generator_openai import generator_OpenAI
from internal_models import MessageConfig, ModelConfig
from the_pydantics.example_schema import ExampleModel

async def compare_generators():
    message_config = MessageConfig(
        system_message="Extract structured data from the following text.",
        pre_prompt="Return information as JSON according to the schema.",
        text="The field1 value is 'sample text' and field2 is 42.",
        pydantic_model=ExampleModel
    )
    
    # Use Ollama (local)
    ollama_config = ModelConfig(model_name="llama3", temperature=0.1)
    ollama_result = await generator_Ollama(message_config, ollama_config)
    
    # Use OpenAI (API)  
    openai_config = ModelConfig(model_name="gpt-4o-mini", temperature=0.1)
    openai_result = await generator_OpenAI(message_config, openai_config)
    
    print(f"Ollama: {ollama_result.parsing_success}, Time: {ollama_result.execution_time:.2f}s")
    print(f"OpenAI: {openai_result.parsing_success}, Time: {openai_result.execution_time:.2f}s")

asyncio.run(compare_generators())
```



## Environment Management

Each generator has its own virtual environment:
- Path format: `the_venvs/venv_{generator_name}`
- Requirements file: `the_venvs/requirements_{generator_name}.txt`

## Features

- **Isolated Environments**: Each generator has a separate venv stored at `the_venvs/venv_{script_name}` with requirements at `the_venvs/requirements_{script_name}.txt`
- **Flexible Usage**: Available via direct import, subprocess call for notebook integration, or API (FastAPI) for remote use
- **Async Support**: All generators support both synchronous and asynchronous workflows
- **Robust Parsing**: Multiple fallback strategies to handle imperfect LLM outputs
- **Standardized Interface**: Consistent input/output patterns across all LLM providers

## Adding New Generators

To add a new generator for another LLM provider:

1. Create a new Python file named `generator_{ProviderName}.py`
2. Implement the generator using the same interface and data models
3. Create requirements file for specific dependencies
4. Add appropriate tests and examples

## Integration Options

Generators can be used in three ways:
1. **Direct import** for use within Python code
2. **Subprocess call** for notebook integration
3. **API endpoint** when exposed through FastAPI (forthcoming)


