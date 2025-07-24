"""
Pydantic utility functions for LLM extraction workflows.
Convert between Pydantic models and JSON schema, parse LLM outputs into Pydantic models,
and handle various failure cases gracefully.

Served by: Direct call

Path to venv, if required: "the_venvs/venv_utils_pydantic"

Libraries to import:
- pydantic
- json
- re
- ast
- xml.etree.ElementTree
"""

###################
####  imports  ####
###################
import subprocess
import sys
import json
import re
import ast
import inspect
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_type_hints
import xml.etree.ElementTree as ET
import importlib

libraries = [
    "pydantic",
    "jsonschema"
]

for lib in libraries:
    try:
        __import__(lib.replace("-", "_"))  # crude guess for pip-import mismatch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

from pydantic import BaseModel, ValidationError, Field, create_model
import jsonschema

# Import models from generators_models.py where applicable
try:
    from internal_models import GenerationResult
except ImportError:
    the_logger.debug("Could not import GenerationResult from generators_models.py")

###################
####  logging  ####
###################
from utils import the_logger

the_logger.info("Initializing pydantic utils for LLM extraction")

###############################
#### 1. pydantic to schema  ####
###############################
async def pydantic_to_schema(model: Type[BaseModel], include_examples: bool = True) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a JSON schema with optional field examples.
    
    Args:
        model: The Pydantic model class
        include_examples: Whether to include example values in schema descriptions
    
    Returns:
        Dictionary containing the JSON schema representation
    """
    the_logger.info(f"Converting {model.__name__} to JSON schema")
    
    # Use asyncio.to_thread for CPU-bound operations to avoid blocking the event loop
    def _get_schema():
        # Get the JSON schema from the Pydantic model
        schema = model.model_json_schema()
        
        # Add example values if available and requested
        if include_examples:
            # Check if model has Config class with examples
            if hasattr(model, "Config") and hasattr(model.Config, "schema_extra"):
                try:
                    # Apply Config schema_extra to enhance the schema
                    if callable(model.Config.schema_extra):
                        model.Config.schema_extra(schema)
                    elif isinstance(model.Config.schema_extra, dict):
                        schema.update(model.Config.schema_extra)
                except Exception as e:
                    the_logger.warning(f"Error applying schema_extra: {str(e)}")
        
        return schema
    
    return await asyncio.to_thread(_get_schema)

async def get_pydantic_field_descriptions(model: Type[BaseModel]) -> Dict[str, str]:
    """
    Extract field descriptions from a Pydantic model.
    
    Args:
        model: The Pydantic model class
    
    Returns:
        Dictionary mapping field names to their descriptions
    """
    # Use asyncio.to_thread to avoid blocking for larger models
    def _get_descriptions():
        descriptions = {}
        
        for field_name, field in model.__annotations__.items():
            field_info = model.model_fields.get(field_name)
            if field_info and field_info.description:
                descriptions[field_name] = field_info.description
        
        return descriptions
    
    return await asyncio.to_thread(_get_descriptions)

async def generate_extraction_prompt(model: Type[BaseModel], verbose: bool = False) -> str:
    """
    Generate a prompt for LLM extraction based on a Pydantic model.
    
    Args:
        model: The Pydantic model class
        verbose: Whether to include detailed field information
    
    Returns:
        A string prompt describing the data to extract
    """
    # Use asyncio.to_thread for potentially expensive operations
    schema = await asyncio.to_thread(pydantic_to_schema, model)
    descriptions = await asyncio.to_thread(get_pydantic_field_descriptions, model)
    
    # Start with model name
    prompt = f"Extract the following information as a JSON object matching this schema: {model.__name__}\n\n"
    
    # Add fields with types and descriptions
    prompt += "Fields to extract:\n"
    
    for field_name, field_props in schema.get("properties", {}).items():
        field_type = field_props.get("type", "any")
        description = descriptions.get(field_name) or field_props.get("description", "")
        
        if field_type == "array":
            items = field_props.get("items", {})
            item_type = items.get("type", "any")
            field_type = f"array of {item_type}s"
        
        prompt += f"- {field_name} ({field_type})"
        if description:
            prompt += f": {description}"
        prompt += "\n"
    
    # Add format instructions
    prompt += "\nRespond with a valid JSON object only."
    
    if verbose:
        prompt += f"\n\nFull schema: {json.dumps(schema, indent=2)}"
    
    return prompt

###############################
#### 2. json output parser (to pydantic)  ####
###############################
async def parse_json_to_pydantic_async(
    json_data: Union[str, Dict[str, Any]], 
    model: Type[BaseModel]
) -> Tuple[Optional[BaseModel], Optional[str]]:
    """
    Parse JSON data into a Pydantic model asynchronously.
    
    Args:
        json_data: Either a JSON string or a dictionary
        model: The target Pydantic model class
    
    Returns:
        Tuple of (parsed model instance or None, error message or None)
    """
    the_logger.info(f"Parsing JSON data to {model.__name__} (async)")
    
    def _parse():
        if isinstance(json_data, str):
            try:
                parsed_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON string: {str(e)}"
                the_logger.error(error_msg)
                return None, error_msg
        else:
            parsed_data = json_data
        
        try:
            instance = model(**parsed_data)
            return instance, None
        except ValidationError as e:
            error_msg = f"Validation error: {str(e)}"
            the_logger.warning(error_msg)
            
            # Try to fix common issues
            try:
                # If some fields are missing, use partial model
                present_fields = {k: v for k, v in parsed_data.items() if k in model.__annotations__}
                instance = model(**present_fields)
                the_logger.info(f"Created partial model with available fields: {present_fields.keys()}")
                return instance, "Partial model created (some fields missing)"
            except ValidationError:
                return None, error_msg
    
    return await asyncio.to_thread(_parse)


#######################################
#### 2.1. Manual parser for raw response ####
#######################################
async def validate_and_extract_tool_calls_async(assistant_content):
    """
    Extract structured tool calls from XML-like content in assistant responses asynchronously.
    
    Args:
        assistant_content: Raw string content from LLM
    
    Returns:
        Tuple of (validation_result, tool_calls, error_message)
    """
    def _extract():
        validation_result = False
        tool_calls = []
        error_message = None

        try:
            # wrap content in root element
            xml_root_element = f"<root>{assistant_content}</root>"
            root = ET.fromstring(xml_root_element)

            # extract JSON data
            for element in root.findall(".//tool_call"):
                json_data = None
                try:
                    json_text = element.text.strip()

                    try:
                        # Prioritize json.loads for better error handling
                        json_data = json.loads(json_text)
                    except json.JSONDecodeError as json_err:
                        try:
                            # Fallback to ast.literal_eval if json.loads fails
                            json_data = ast.literal_eval(json_text)
                        except (SyntaxError, ValueError) as eval_err:
                            error_message = f"JSON parsing failed with both json.loads and ast.literal_eval:\n"\
                                            f"- JSON Decode Error: {json_err}\n"\
                                            f"- Fallback Syntax/Value Error: {eval_err}\n"\
                                            f"- Problematic JSON text: {json_text}"
                            the_logger.error(error_message)
                            continue
                except Exception as e:
                    error_message = f"Cannot strip text: {e}"
                    the_logger.error(error_message)

                if json_data is not None:
                    tool_calls.append(json_data)
                    validation_result = True

        except ET.ParseError as err:
            error_message = f"XML Parse Error: {err}"
            the_logger.error(f"XML Parse Error: {err}")

        # Return default values if no valid data is extracted
        return validation_result, tool_calls, error_message
    
    return await asyncio.to_thread(_extract)

# Keep the synchronous version for backwards compatibility
def validate_and_extract_tool_calls(assistant_content):
    """
    Extract structured tool calls from XML-like content in assistant responses.
    
    Args:
        assistant_content: Raw string content from LLM
    
    Returns:
        Tuple of (validation_result, tool_calls, error_message)
    """
    validation_result = False
    tool_calls = []
    error_message = None

    try:
        # wrap content in root element
        xml_root_element = f"<root>{assistant_content}</root>"
        root = ET.fromstring(xml_root_element)

        # extract JSON data
        for element in root.findall(".//tool_call"):
            json_data = None
            try:
                json_text = element.text.strip()

                try:
                    # Prioritize json.loads for better error handling
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        error_message = f"JSON parsing failed with both json.loads and ast.literal_eval:\n"\
                                        f"- JSON Decode Error: {json_err}\n"\
                                        f"- Fallback Syntax/Value Error: {eval_err}\n"\
                                        f"- Problematic JSON text: {json_text}"
                        the_logger.error(error_message)
                        continue
            except Exception as e:
                error_message = f"Cannot strip text: {e}"
                the_logger.error(error_message)

            if json_data is not None:
                tool_calls.append(json_data)
                validation_result = True

    except ET.ParseError as err:
        error_message = f"XML Parse Error: {err}"
        the_logger.error(f"XML Parse Error: {err}")

    # Return default values if no valid data is extracted
    return validation_result, tool_calls, error_message

async def extract_json_from_markdown_async(text):
    """
    Extracts the JSON string from the given text using a regular expression pattern asynchronously.
    
    Args:
        text (str): The input text containing the JSON string.
        
    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    def _extract():
        json_pattern = r'```(?:json)?\r?\n(.*?)\r?\n```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_string = match.group(1)
            try:
                data = json.loads(json_string)
                return data
            except json.JSONDecodeError as e:
                the_logger.error(f"Error decoding JSON string: {e}")
        else:
            the_logger.warning("JSON string not found in the text.")
        return None
    
    return await asyncio.to_thread(_extract)

def extract_json_from_markdown(text):
    """
    Extracts the JSON string from the given text using a regular expression pattern.
    
    Args:
        text (str): The input text containing the JSON string.
        
    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    json_pattern = r'```(?:json)?\r?\n(.*?)\r?\n```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            the_logger.error(f"Error decoding JSON string: {e}")
    else:
        the_logger.warning("JSON string not found in the text.")
    return None

#######################################
#### 3. Advanced JSON extraction  ####
#######################################
async def extract_structured_data_async(content: str) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Advanced extraction of JSON from various text formats asynchronously.
    Tries multiple extraction methods to find valid JSON data.
    
    Args:
        content: Raw text content potentially containing JSON
        
    Returns:
        Extracted JSON data as dict or list, or None if extraction fails
    """
    the_logger.info("Attempting to extract structured data from content (async)")
    
    # If content is already a dict or list, return it directly
    if isinstance(content, (dict, list)):
        return content
    
    # Helper function to check if a string is valid JSON
    def is_valid_json(json_str: str) -> bool:
        """Check if a string is valid JSON."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    # Sort extraction methods from most reliable to most specialized fallbacks
    extraction_methods = [
        # Method 1: Try to parse the whole content as JSON (simplest and most reliable)
        lambda c: json.loads(c) if c.strip().startswith('{') and c.strip().endswith('}') else None,
        
        # Method 2: Extract JSON from code blocks (common in markdown LLM outputs)
        lambda c: extract_json_from_markdown(c),
        
        # Method 3: Extract line by line (for newline-formatted responses)
        lambda c: next((json.loads(line.strip()) for line in c.splitlines() 
                      if line.strip().startswith('{') and line.strip().endswith('}') and is_valid_json(line.strip())), None),
        
        # Method 4: Extract from tool calls (for API-like responses)
        lambda c: validate_and_extract_tool_calls(c)[1][0] if validate_and_extract_tool_calls(c)[0] and validate_and_extract_tool_calls(c)[1] else None,
        
        # Method 5: Find the longest JSON-like string between curly braces (more aggressive parsing)
        lambda c: max([json.loads(m) for m in re.findall(r'({.*?})', c, re.DOTALL) if is_valid_json(m)], key=lambda x: len(str(x)), default=None),
        
        # Method 6: Fallback to ast.literal_eval for Python-like dictionaries (most permissive approach)
        lambda c: (ast.literal_eval(c) if isinstance(ast.literal_eval(c), (dict, list)) else None) 
                if all(char in c for char in ['{', '}']) else None
    ]
    
    # Try each extraction method in order
    for i, method in enumerate(extraction_methods):
        try:
            result = await asyncio.to_thread(method, content)
            if result is not None:
                the_logger.info(f"Successfully extracted data using method {i+1}")
                return result
        except Exception as e:
            the_logger.debug(f"Method {i+1} failed: {str(e)}")
            continue
    
    # Final fallback: log and return None
    the_logger.warning("All extraction attempts failed; returning None")
    return None

#######################################
#### 4. Main extraction functions  ####
#######################################
async def advanced_parser_Utils_async(
    content: Union[str, BaseModel, Dict[str, Any]], model: Type[BaseModel]
) -> Tuple[Optional[BaseModel], Union[str, Dict[str, Any]]]:
    """
    Full pipeline to extract JSON from content and parse it into a Pydantic model asynchronously.
    Tries multiple extraction methods and always returns the original content on failure.
    
    Args:
        content: Raw text content, Pydantic model, or dictionary potentially containing JSON
        model: The target Pydantic model class
    
    Returns:
        Tuple of (parsed model instance or None, original content)
    """
    the_logger.info("Starting full extraction and parsing pipeline (async)")
    
    # Store original content to return on failure
    original_content = content

    # Check if content is already a Pydantic model instance of the requested type
    if isinstance(content, model):
        the_logger.info("Content is already the requested Pydantic model type")
        return content, None
    
    # Check if content is a Pydantic model of a different type - convert to dict first
    if isinstance(content, BaseModel):
        the_logger.info("Content is a Pydantic model, converting to dict first")
        content = content.model_dump()
    
    # Process dict content directly
    if isinstance(content, dict):
        instance, error = await parse_json_to_pydantic_async(content, model)
        if instance is not None:
            the_logger.info("Successfully parsed dictionary to model")
            return instance, None
        else:
            the_logger.warning(f"Failed to validate dictionary: {error}")
            return None, original_content

    # For string content, extract structured data first
    structured_data = await extract_structured_data_async(content)
    if structured_data is not None:
        instance, error = await parse_json_to_pydantic_async(structured_data, model)
        if instance is not None:
            the_logger.info("Successfully parsed structured data to model")
            return instance, None
        else:
            the_logger.warning(f"Failed to validate structured data: {error}")
            return None, original_content
    else:
        the_logger.warning("No structured data could be extracted")
        return None, original_content
    
