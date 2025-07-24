"""
Generator function for extracting structured data using Ollama LLMs.
Takes an input text and a Pydantic model, produces structured output according to the model.

Served by: API and direct call

Path to venv, if required: "the_venvs/venv_generator_ollama"

Libraries to import:
- ollama
- pydantic
- time
- asyncio
"""

###################
####  imports  ####
###################
import subprocess
import sys
import json
import os
import asyncio
import time
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from pathlib import Path

libraries = [
    "ollama",
    "pydantic"
]

for lib in libraries:
    try:
        __import__(lib.replace("-", "_"))  # crude guess for pip-import mismatch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

from pydantic import BaseModel, Field

try:
    from ollama import AsyncClient
except ImportError:
    print("Error importing AsyncClient from ollama. Make sure ollama package is installed.")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import settings - using local config for better organization
from generators.config import get_ollama_settings

# Import utility functions from utils modules
from generators.utils.utils_pydantic import advanced_parser_Utils_async
from generators.utils.utils_messages import prepare_messages_Utils_async

###################
####  logging  ####
###################
from utils import the_logger

the_logger.info("Initializing Ollama generator module")

###################################
#### 1. Configuration Models  #####
###################################
# Import models from generators_models.py
from internal_models import MessageConfig, ModelConfig, GenerationResult

# Get default values from settings
ollama_settings = get_ollama_settings()
DEFAULT_MODEL = ollama_settings["default_model"]
DEFAULT_TEMPERATURE = ollama_settings["default_temperature"]
DEFAULT_MAX_TOKENS = ollama_settings["default_max_tokens"]
DEFAULT_SYSTEM_MESSAGE = ollama_settings["default_system_message"]
REQUEST_TIMEOUT = ollama_settings["request_timeout"]

###################################
#### 2. Ollama Client Functions ###
###################################
async def generator_Ollama(
    message_config: MessageConfig,
    model_config: ModelConfig
) -> GenerationResult:
    """
    Extract structured data using Ollama.
    
    Args:
        message_config: Configuration for input message
        model_config: Configuration for Ollama model
        
    Returns:
        GenerationResult with extraction results
    """

    
    start_time = time.time()
    error = None
    
    try:
        # Initialize Ollama client
        client = AsyncClient()
        
        # Resolve Pydantic model
        model_class = message_config.pydantic_model
        
        # Generate options dictionary
        options = model_config.to_options_dict()
        
        # Get the model schema
        schema = model_class.model_json_schema()
        
        the_logger.info(f"Making extraction request to Ollama model {model_config.model_name}")
        
        # Use prepare_messages_Utils_async from utils_messages.py with MessageConfig
        messages = await prepare_messages_Utils_async(message_config)
        
        # Send request to Ollama
        response = await client.chat(
            model=model_config.model_name,
            messages=messages,
            format=schema,
            options=options
        )
        
        generation_success = True
        raw_response = response.message.content
        
        # Use advanced_parser_Utils_async from utils_pydantic.py to parse the response
        parsed_model, parsing_result = await advanced_parser_Utils_async(raw_response, model_class)
        parsing_success = parsed_model is not None
        
        execution_time = time.time() - start_time
        
        # Log results
        if parsing_success:
            the_logger.info(f"Successfully extracted and parsed data in {execution_time:.2f} seconds")
        else:
            the_logger.warning(f"Generated response but failed to parse into {model_class.__name__}")
        
        return GenerationResult(
            execution_time=execution_time,
            generation_success=generation_success,
            parsing_success=parsing_success,
            raw_response=raw_response,
            parsed_response=parsed_model
        )
        
    except Exception as e:
        error_msg = f"Error in Ollama extraction: {str(e)}"
        the_logger.error(error_msg)
        execution_time = time.time() - start_time
        
        return GenerationResult(
            execution_time=execution_time,
            generation_success=False,
            parsing_success=False,
            error=error_msg
        )



###################################
####      Main function        ####
###################################
if __name__ == "__main__":
    import argparse
    from the_pydant ics.example_schema import ExampleModel
    
    async def main():
        parser = argparse.ArgumentParser(description="Extract structured data using Ollama.")
        parser.add_argument("--text", type=str, required=True, help="Text to extract data from")
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
        parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for generation")
        args = parser.parse_args()
        
        message_config = MessageConfig(
            system_message="Extract the requested information from the text as JSON.",
            pre_prompt="Please extract the following fields into a JSON object:",
            text=args.text,
            pydantic_model=ExampleModel
        )
        
        model_config = ModelConfig(
            model_name=args.model,
            temperature=args.temp
        )
        
        result = await generator_Ollama(message_config, model_config)
        
        if result.generation_success:
            print("\nGeneration successful!")
            print(f"Execution time: {result.execution_time:.2f} seconds")
            print("\nRaw response:")
            print(result.raw_response)
            
            if result.parsing_success:
                print("\nParsed result:")
                print(result.parsed_response.model_dump_json(indent=2))
            else:
                print("\nParsing failed. Could not convert to Pydantic model.")
        else:
            print(f"\nGeneration failed: {result.error}")
    
    asyncio.run(main())

###################################
####  Example use in terminal  ####
###################################
"""
python generators/generator_ollama.py --text "Example value is 42 and contains the items: first, second." --model llama3 --temp 0.1
"""

###################################
####  Example use in notebook  ####
###################################
"""
# Async usage - with MessageConfig and ModelConfig
from generators.generator_ollama import generator_Ollama
from generators.generators_models import MessageConfig, ModelConfig
from the_pydantics.example_schema import ExampleModel
import asyncio

async def extract_from_text():
    message_config = MessageConfig(
        system_message="Extract structured data from the following text.",
        pre_prompt="Return information as JSON according to the schema.",
        text="The example value is 'test data' with the number 123 and a list containing a, b, and c.",
        pydantic_model=ExampleModel
    )
    
    model_config = ModelConfig(
        model_name="llama3", 
        temperature=0.1,
        max_tokens=1000
    )
    
    result = await generator_Ollama(message_config, model_config)
    
    if result.parsing_success:
        print(f"Extracted data: {result.parsed_response}")
    else:
        print(f"Failed to parse. Raw response: {result.raw_response}")
        
    return result

# Run async function
extraction_result = asyncio.run(extract_from_text())

# Simple synchronous usage - without needing to import the config models
from generators.generator_ollama import extract_structured_data_sync
from the_pydantics.example_schema import ExampleModel

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
"""
