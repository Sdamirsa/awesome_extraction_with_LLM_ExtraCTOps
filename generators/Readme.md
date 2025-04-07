# ExtraCTOps Generators

This directory contains generator modules that extract structured data from unstructured text using various Large Language Models (LLMs). The generators are designed to be modular, extensible, and compatible with both synchronous and asynchronous workflows.

I'll create a comprehensive table for the generators with the columns you specified, plus a few additional useful columns.

## Generators Comparison Table
| Generator | Script | Specific venv | LLM | Parser | Provider | Async Support | Multimodal | Interface |
|-----------|--------|---------------|-----|--------|----------|--------------|------------|-----------|
| Ollama | generator_Ollama.py | venv_generator_Ollama | Various (llama3, mixtral, etc.) | JSON parsing with fallbacks | Ollama (Local) | Yes | Yes (images) | Direct import, Sync/Async functions |


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

- **generator_Ollama.py** - Uses Ollama for local LLM generation
- *(Additional generators will be added for other LLM providers)*

## Utility Modules

- **utils_messages.py** - Prepares input messages for LLMs, including multimodal capabilities
- **utils_pydantic.py** - Handles conversion between Pydantic models and JSON schemas, and parsing LLM outputs

## Usage Examples

### Basic Usage with Ollama

```python
import asyncio
from generators.generator_Ollama import extract_structured_data_sync
from the_pydantics.example_schema import ExampleModel

# Simple synchronous usage
parsed_data, raw_response, success = extract_structured_data_sync(
    text="The field1 value is 'sample text' and field2 is 42.",
    pydantic_model=ExampleModel,
    model_name="llama3",
    temperature=0.0
)

if success:
    print(f"Successfully extracted: {parsed_data}")
else:
    print(f"Extraction failed. Raw response: {raw_response}")
```

### Advanced Usage with Configurations

```python
import asyncio
from generators.generator_Ollama import generator_Ollama
from generators.generators_models import MessageConfig, ModelConfig
from the_pydantics.example_schema import ExampleModel

async def extract_from_text():
    # Create configuration objects
    message_config = MessageConfig(
        system_message="Extract structured data from the following text.",
        pre_prompt="Return information as JSON according to the schema.",
        text="The example contains value 'test data' and number 123.",
        pydantic_model=ExampleModel
    )
    
    model_config = ModelConfig(
        model_name="llama3", 
        temperature=0.1,
        max_tokens=1000
    )
    
    # Call the generator
    result = await generator_Ollama(message_config, model_config)
    
    if result.parsing_success:
        print(f"Extracted data: {result.parsed_response}")
    else:
        print(f"Failed to parse. Raw response: {result.raw_response}")
        
    return result

# Run async function
asyncio.run(extract_from_text())
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


